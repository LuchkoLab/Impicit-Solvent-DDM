from dataclasses import dataclass, field
from typing import Optional, Union
from copy import copy
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.restraints import RestraintMaker
from implicit_solvent_ddm.simulations import Simulation
from toil.common import FileID
from toil.job import Promise


@dataclass
class SimulationSetup:
    config: Config
    system_type: str
    restraints: Union[RestraintMaker, Promise]
    binding_mode: FileID
    endstate_traj: FileID
    simulations: list = field(default_factory=list)
    topology: Union[FileID, str] = field(init=False)
    dirstruct: str = field(init=False)
    num_cores: float = field(init=False)
    max_conformational_exponent: float = field(init=False)
    max_orientational_exponent: float = field(init=False)

    def __post_init__(self):
        """
        Post-initialization setup for simulation parameters.

        - Computes and stores the maximum conformational and orientational exponents.
        - Sets the topology file, number of cores, and directory structure based on system type.
        - For GPU-enabled systems (`complex` and `receptor`), reduces CPU core allocation to 0.1.
        """
        # Round and store max exponents
        self.max_conformational_exponent = round(
            max(self.config.intermediate_args.exponent_conformational_forces), 3
        )
        self.max_orientational_exponent = round(
            max(self.config.intermediate_args.exponent_orientational_forces), 3
        )

        # Default values
        dirstruct_map = {
            "complex": "dirstruct_halo",
            "receptor": "dirstruct_apo",
            "ligand": "dirstruct_apo"
        }
        self.dirstruct = dirstruct_map.get(self.system_type, "dirstruct_apo")

        # Topology and core count based on system type
        endstate_files = self.config.endstate_files
        ncores_map = self.config.num_cores_per_system

        if self.system_type == "complex":
            self.topology = endstate_files.complex_parameter_filename
            self.num_cores = ncores_map.complex_ncores
        elif self.system_type == "receptor":
            self.topology = endstate_files.receptor_parameter_filename
            self.num_cores = ncores_map.receptor_ncores
        else:
            self.topology = endstate_files.ligand_parameter_filename
            self.num_cores = ncores_map.ligand_ncores

        # Reduce CPU usage for GPU-bound receptor/complex jobs
        if self.config.system_settings.CUDA and self.system_type in {"complex", "receptor"}:
            self.num_cores = 0.1
 

    def setup_post_endstate_simulation(self, flat_bottom: bool = False):
        """
        Sets up a post-endstate simulation for the system (complex, receptor, or ligand),
        with optional flat-bottom restraints for the complex system.

        Parameters
        ----------
        flat_bottom : bool, optional
            Whether to apply flat-bottom restraints for the complex, by default False.
        """
        temp_args = copy(self.apo_endstate_dirstruct)
        restraint_file = self.config.inputs["empty_restraint"]
        dirstruct_type = "post_process_apo"

        # Determine coordinate and settings by system type
        if self.system_type == "complex":
            dirstruct_type = "post_process_halo"
            restraint_file = (
                self.config.inputs["flat_bottom_restraint"]
                if flat_bottom
                else self.config.inputs["empty_restraint"]
            )

            if not flat_bottom:
                temp_args["state_label"] = "no_flat_bottom"

            coordinate = (
                self.config.endstate_files.complex_initial_coordinate
                or self.config.endstate_files.complex_coordinate_filename
            )

        elif self.system_type == "receptor":
            coordinate = (
                self.config.endstate_files.receptor_initial_coordinate
                or self.config.endstate_files.receptor_coordinate_filename
            )

        else:  # ligand
            coordinate = (
                self.config.endstate_files.ligand_initial_coordinate
                or self.config.endstate_files.ligand_coordinate_filename
            )

        temp_args["topology"] = self.topology

        # Append simulation
        self.simulations.append(
            Simulation(
                executable="sander.MPI",
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=1,
                CUDA=self.config.system_settings.CUDA,
                prmtop=self.topology,
                incrd=coordinate,
                input_file=self.config.inputs["post_mdin"],
                restraint_file=restraint_file,
                directory_args=temp_args,
                system_type=self.system_type,
                dirstruct=dirstruct_type,
                inptraj=[self.endstate_traj],
                post_analysis=True,
                working_directory=self.config.system_settings.working_directory,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
            )
        )

    def setup_ligand_charge_simulation(
        self, prmtop: FileID, charge: float, restraint_key: str
    ):
        """
        Set up a molecular dynamics (MD) simulation to scale the ligand charges for alchemical transitions.

        This function prepares an MD production simulation where the ligand charge is scaled 
        from its full value (1.0) toward 0.0 (fully uncharged). The setup differs depending on whether 
        the system is a ligand-only system or a complex (ligand + receptor). Appropriate GB model, dielectric 
        constants, and simulation inputs are selected based on this context.

        Parameters
        ----------
        prmtop : FileID
            The AMBER parameter topology file containing the system topology, including the ligand.
        charge : float
            The fractional charge to assign to the ligand. A value of 1.0 corresponds to full charge, 
            while 0.0 corresponds to a fully uncharged ligand.
        restraint_key : str
            Identifier for the restraint configuration to be applied to this simulation.

        Notes
        -----
        - For `system_type == "complex"`, the simulation is assumed to be in vacuum with GB solvent corrections.
        - For ligand-only systems, GB is fully on (`extdiel = 78.5) to model implicit solvent.
        - Appends a `Simulation` object to `self.simulations`.
        """
        temp_args = copy(self.ligand_charge_args)

        temp_args["charge"] = charge
        temp_args["extdiel"] = 78.5
        temp_args["igb"] = f"igb_{self.config.intermediate_args.igb_solvent}"
        if self.system_type == "complex":
            temp_args["filename"] = "state_7b_prod"
            #temp_args["igb_value"] = self.config.intermediate_args.igb_solvent
            temp_args["orientational_restraints"] = self.max_orientational_exponent
            temp_args["runtype"] = (
                "Running production simulation in state 7b: Turning back on ligand charges, with GB solvent fully on"
            )
            num_acelerators=self.config.system_settings.num_accelerators
            cpu_required=0.1
        # scale ligand charge
        else:
            #mdin_file = self.config.inputs["no_solvent_mdin"]
            #temp_args["igb"] = "igb_6"
            #temp_args["extdiel"] = 0.0
            temp_args["filename"] = "state_4_prod"
            #temp_args["igb_value"] = 6
            temp_args["runtype"] = (
                f"Running production Simulation in state 4 (with GB). Max conformational force: {self.max_conformational_exponent}"
            )
            num_acelerators=None


        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                CUDA=self.config.system_settings.CUDA,
                prmtop=prmtop,
                incrd=self.binding_mode,
                input_file=self.config.inputs["default_mdin"],
                restraint_file=self.restraints,
                directory_args=temp_args,
                dirstruct=self.dirstruct,
                system_type=self.system_type,
                restraint_key=restraint_key,
                working_directory=self.config.system_settings.working_directory,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
                accelerators=num_acelerators,
            )
        )

    def setup_remove_gb_solvent_simulation(self, restraint_key: str, prmtop: FileID):
        """
        Set up a molecular dynamics (MD) simulation with GB solvent removed (vacuum simulation).

        This function prepares a vacuum-phase MD production simulation where the GB implicit 
        solvent model is turned off. The simulation is configured differently depending on the 
        system type (receptor/ligand vs complex), with appropriate directory structures and 
        simulation parameters for vacuum conditions.

        Parameters
        ----------
        restraint_key : str
            Identifier for the restraint configuration to be applied to this simulation.
        prmtop : FileID
            The AMBER parameter topology file containing the system topology.

        Notes
        -----
        - For `system_type` of "receptor" or "ligand", uses `dirstruct_apo` directory structure.
        - For `system_type == "complex"`, uses `dirstruct_halo` directory structure.
        - Sets `igb = 6` (no GB model) and `extdiel = 0.0` (vacuum dielectric) via `no_gb_args`.
        - Appends a `Simulation` object to `self.simulations`.
        """
        temp_args = copy(self.no_gb_args)
        
        if self.system_type != "complex":
            temp_args["state_label"] = "no_gb"
            temp_args["charge"] = 0.0
            temp_args["filename"] = "state_4_prod"
            temp_args["runtype"] = f"Running production simulation in state 4: {self.system_type} only"
            dirstruct_args = "dirstruct_apo"
        
        else:  # complex
            dirstruct_args = "dirstruct_halo"

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                CUDA=self.config.system_settings.CUDA,
                prmtop=prmtop,
                incrd=self.binding_mode,
                input_file=self.config.inputs["no_solvent_mdin"],
                restraint_file=self.restraints,
                directory_args=temp_args,
                system_type=self.system_type,
                dirstruct=dirstruct_args,
                working_directory=self.config.system_settings.working_directory,
                restraint_key=restraint_key,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
                accelerators=self.config.system_settings.num_accelerators,
            )
        )

    def setup_lj_interations_simulation(self, restraint_key: str, prmtop: FileID):
        """
        Set up a molecular dynamics (MD) simulation to restore Lennard-Jones interactions in vacuum.

        This function prepares a vacuum-phase MD production simulation where the Lennard-Jones 
        (LJ) interactions between the receptor and guest (ligand) are turned back on. This 
        simulation represents state 7a in the alchemical thermodynamic cycle, where non-bonded 
        interactions are restored in the absence of solvent.

        Parameters
        ----------
        restraint_key : str
            Identifier for the restraint configuration to be applied to this simulation.
        prmtop : FileID
            The AMBER parameter topology file containing the system topology.

        Notes
        -----
        - This simulation uses vacuum conditions (`igb = 6`, `extdiel = 0.0`) via `no_gb_args`.
        - Always uses `dirstruct_halo` directory structure (complex system).
        - Uses the `no_solvent_mdin` input file for vacuum simulations.
        - Appends a `Simulation` object to `self.simulations`.
        """
        temp_args = copy(self.no_gb_args)
        temp_args["state_label"] = "interactions"
        temp_args["filename"] = "state_7a_prod"
        temp_args["runtype"] = (
            "Running production simulation in state 7a: Turing back on interactions with recetor and guest in vacuum"
        )

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                CUDA=self.config.system_settings.CUDA,
                prmtop=prmtop,
                incrd=self.binding_mode,
                input_file=self.config.inputs["no_solvent_mdin"],
                restraint_file=self.restraints,
                directory_args=temp_args,
                system_type=self.system_type,
                dirstruct="dirstruct_halo",
                working_directory=self.config.system_settings.working_directory,
                restraint_key=restraint_key,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
                accelerators=self.config.system_settings.num_accelerators,
            )
        )

    def setup_gb_external_dielectric(
        self, restraint_key: str, prmtop: FileID, mdin: FileID, extdiel: float
    ):
        """
        Set up a molecular dynamics (MD) simulation with GB solvent at a specific external dielectric constant.

        This function prepares an MD production simulation where the external dielectric constant 
        (extdiel) is varied while using the GB implicit solvent model. This simulation represents 
        state 8 in the alchemical thermodynamic cycle and is used to modulate the strength of 
        electrostatic solvation effects.

        Parameters
        ----------
        restraint_key : str
            Identifier for the restraint configuration to be applied to this simulation.
        prmtop : FileID
            The AMBER parameter topology file containing the system topology.
        mdin : FileID
            The AMBER MD input file containing simulation parameters.
        extdiel : float
            The external dielectric constant to be used in the simulation. Typical values 
            range from 1.0 (vacuum-like) to 78.5 (water-like).

        Notes
        -----
        - For `system_type == "complex"`, uses `dirstruct_halo` and filename `state_8_prod`.
        - For ligand/receptor systems, uses `dirstruct_apo` and filename `state_gb_dielectric_ligand`.
        - Sets charge to 0.0 (uncharged state) while varying dielectric response.
        - Uses the configured GB model via `igb_solvent` setting.
        - Appends a `Simulation` object to `self.simulations`.
        """
        temp_args = copy(self.no_gb_args)

        temp_args["state_label"] = "gb_dielectric"
        temp_args["filename"] = "state_8_prod"
        temp_args["extdiel"] = extdiel
        temp_args["igb"] = f"igb_{self.config.intermediate_args.igb_solvent}"
        temp_args["igb_value"] = f"igb_{self.config.intermediate_args.igb_solvent}"
        temp_args["charge"] = 0.0
        temp_args["runtype"] = (
            f"Running production Simulation in state 8. Changing extdiel to: {extdiel}."
        )
        dirstruct_type = "dirstruct_halo"

        if self.system_type != "complex":
            dirstruct_type = "dirstruct_apo"
            temp_args["filename"] = "state_gb_dielectric_ligand"   

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                CUDA=self.config.system_settings.CUDA,
                prmtop=prmtop,
                incrd=self.binding_mode,
                input_file=mdin,
                restraint_file=self.restraints,
                directory_args=temp_args,
                system_type=self.system_type,
                dirstruct=dirstruct_type,
                working_directory=self.config.system_settings.working_directory,
                restraint_key=restraint_key,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
                accelerators=self.config.system_settings.num_accelerators,
            )
        )

    def setup_apply_restraint_windows(
        self,
        restraint_key: str,
        exponent_conformational: float,
        exponent_orientational: Optional[float] = None,
    ):
        """
        Set up a molecular dynamics (MD) simulation for a restraint window in thermodynamic integration.

        This function prepares an MD production simulation for a specific lambda window where 
        conformational restraints (and optionally orientational restraints) are applied at varying 
        strengths. These simulations are used to calculate the free energy contribution of applying 
        restraints to the system, with GB implicit solvent fully enabled.

        Parameters
        ----------
        restraint_key : str
            Identifier for the restraint configuration to be applied to this simulation.
        exponent_conformational : float
            The exponent (strength) of the conformational restraints to be applied. 
            Higher values correspond to stronger restraints.
        exponent_orientational : Optional[float], optional
            The exponent (strength) of the orientational restraints to be applied.
            If provided, indicates a complex system with both conformational and orientational 
            restraints. Defaults to None.

        Notes
        -----
        - Uses GB implicit solvent (`extdiel = 78.5`, full charge `charge = 1.0`).
        - For ligand-only systems (no orientational restraints), uses `dirstruct_apo` and 
          filename pattern `state_2_{exponent_conformational}_prod`.
        - For complex systems (with orientational restraints), uses `dirstruct_halo` and 
          filename pattern `state_8_{exponent_conformational}_{exponent_orientational}_prod`.
        - GPU acceleration is enabled for non-ligand systems when CUDA is available.
        - Appends a `Simulation` object to `self.simulations`.
        """
        if not self.system_type == "ligand" and self.config.system_settings.CUDA:
            num_accelerators = self.config.system_settings.num_accelerators
        else:
            num_accelerators = None

        temp_args = copy(self.no_gb_args)

        temp_args["state_label"] = "lambda_window"
        temp_args["extdiel"] = 78.5
        temp_args["charge"] = 1.0
        temp_args["filename"] = f"state_2_{exponent_conformational}_prod"
        temp_args["igb"] = f"igb_{self.config.intermediate_args.igb_solvent}"
        temp_args["igb_value"] = self.config.intermediate_args.igb_solvent
        temp_args["runtype"] = (
            f"Running restraint window. Conformational restraint: {exponent_conformational}"
        )
        temp_args["conformational_restraint"] = exponent_conformational
        dirstruct_type = "dirstruct_apo"

        if exponent_orientational is not None:
            temp_args["orientational_restraints"] = exponent_orientational
            temp_args["filename"] = (
                f"state_8_{exponent_conformational}_{exponent_orientational}_prod"
            )
            dirstruct_type = "dirstruct_halo"

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                CUDA=self.config.system_settings.CUDA,
                prmtop=self.topology,
                incrd=self.binding_mode,
                input_file=self.config.inputs["default_mdin"],
                restraint_file=self.restraints,
                directory_args=temp_args,
                system_type=self.system_type,
                dirstruct=dirstruct_type,
                working_directory=self.config.system_settings.working_directory,
                restraint_key=restraint_key,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
                accelerators=num_accelerators,
            )
        )

    @property
    def ligand_charge_args(self) -> dict:
        """
        Base arguments for scaling ligand charges in alchemical simulations.

        Returns
        -------
        dict
            Dictionary containing simulation parameters for ligand charge scaling,
            including topology, state labels, restraints, and GB model settings.
        """

        return {
            "topology": self.topology,
            "state_label": "electrostatics",
            "conformational_restraint": self.max_conformational_exponent,
            # "igb": "igb_6",
            # "extdiel": 0.0,
            # "charge": charge,
            "igb_value": f"igb_{self.config.intermediate_args.igb_solvent}",
            # "filename": "state_4_prod",
            # "runtype": f"Running production Simulation in state 4 (No GB). Max conformational force: {self.max_conformational_exponent} ",
            "topdir": self.config.system_settings.top_directory_path,
        }

    @property
    def no_gb_args(self) -> dict:
        return {
            "topology": self.topology,
            "state_label": "no_interactions",
            "extdiel": 0.0,
            "igb": "igb_6",
            "igb_value": 6,
            "charge": 0.0,
            "conformational_restraint": self.max_conformational_exponent,
            "orientational_restraints": self.max_orientational_exponent,
            "filename": "state_7_prod",
            "topdir": self.config.system_settings.top_directory_path,
            "runtype": "Running production simulation in state 7: No iteractions with receptor/guest and in vacuum",
        }

    @property
    def apo_endstate_dirstruct(self) -> dict:
        """
        Arguments for post-processing endstate simulations with no restraints.

        Returns
        -------
        dict
            Dictionary containing simulation parameters for the unrestrained endstate,
            including full charges (1.0), GB solvent (extdiel = 78.5), zero restraints,
            and trajectory analysis settings for REMD post-processing.
        """
        return {
            "topology": self.topology,
            "state_label": "endstate",
            "charge": 1.0,
            "extdiel": 78.5,
            "igb": f"igb_{self.config.intermediate_args.igb_solvent}",
            "igb_value": self.config.intermediate_args.igb_solvent,
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_charge": 1.0,
            "traj_extdiel": 78.5,
            "topdir": self.config.system_settings.top_directory_path,
            "traj_igb": f"igb_{self.config.intermediate_args.igb_solvent}",
            "filename": "state_2_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
            "trajectory_restraint_orenrest": 0.0,
        }
