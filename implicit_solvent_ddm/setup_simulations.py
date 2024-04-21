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
        self.max_conformational_exponent = float(
            round(max(self.config.intermidate_args.exponent_conformational_forces), 3)
        )
        self.max_orientational_exponent = float(
            round(max(self.config.intermidate_args.exponent_orientational_forces), 3)
        )
        self.dirstruct = "dirstruct_apo"
        if self.system_type == "complex":
            self.topology = self.config.endstate_files.complex_parameter_filename
            self.num_cores = self.config.num_cores_per_system.complex_ncores
            self.dirstruct = "dirstruct_halo"

        elif self.system_type == "receptor":
            self.topology = self.config.endstate_files.receptor_parameter_filename  # type: ignore
            self.num_cores = self.config.num_cores_per_system.receptor_ncores

        else:
            self.topology = self.config.endstate_files.ligand_parameter_filename  # type: ignore
            self.num_cores = self.config.num_cores_per_system.ligand_ncores

    def setup_post_endstate_simulation(self, flat_bottom: bool = False):
        """_summary_

        Args:
            system_type (str): _description_
        """
        temp_args = copy(self.apo_endstate_dirstruct)

        restraint_file = self.config.inputs["empty_restraint"]

        dirstruct_type = "post_process_apo"

        if self.system_type == "complex":
            coordinate = self.config.endstate_files.complex_coordinate_filename
            dirstruct_type = "post_process_halo"
            restraint_file = self.config.inputs["flat_bottom_restraint"]

            if not flat_bottom:
                temp_args["state_label"] = "no_flat_bottom"
                restraint_file = self.config.inputs["empty_restraint"]

            if self.config.endstate_files.complex_initial_coordinate is not None:
                coordinate = self.config.endstate_files.complex_initial_coordinate

        elif self.system_type == "receptor":
            coordinate = self.config.endstate_files.receptor_coordinate_filename

            if self.config.endstate_files.receptor_initial_coordinate is not None:
                coordinate = self.config.endstate_files.receptor_initial_coordinate

        else:
            coordinate = self.config.endstate_files.ligand_coordinate_filename
            if self.config.endstate_files.ligand_initial_coordinate is not None:
                coordinate = self.config.endstate_files.ligand_initial_coordinate

        temp_args["topology"] = self.topology

        self.simulations.append(
            Simulation(
                executable="sander.MPI",
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
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
        """_summary_

        Args:
            prmtop (_type_): _description_
            charge (_type_): _description_
        """
        temp_args = copy(self.ligand_charge_args)

        temp_args["charge"] = charge
        if self.system_type == "complex":
            mdin_file = self.config.inputs["default_mdin"]
            temp_args["igb"] = f"igb_{self.config.intermidate_args.igb_solvent}"
            temp_args["extdiel"] = 78.5
            temp_args["filename"] = "state_7b_prod"
            temp_args["igb_value"] = self.config.intermidate_args.igb_solvent
            temp_args["orientational_restraints"] = self.max_orientational_exponent
            temp_args["runtype"] = (
                "Running production simulation in state 7b: Turning back on ligand charges, still in vacuum"
            )
        # scale ligand charge
        else:
            mdin_file = self.config.inputs["no_solvent_mdin"]
            temp_args["igb"] = "igb_6"
            temp_args["extdiel"] = 0.0
            temp_args["filename"] = "state_4_prod"
            temp_args["igb_value"] = 6
            temp_args["runtype"] = (
                f"Running production Simulation in state 4 (No GB). Max conformational force: {self.max_conformational_exponent}"
            )

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                prmtop=prmtop,
                incrd=self.binding_mode,
                input_file=mdin_file,
                restraint_file=self.restraints,
                directory_args=temp_args,
                dirstruct=self.dirstruct,
                system_type=self.system_type,
                restraint_key=restraint_key,
                working_directory=self.config.system_settings.working_directory,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
            )
        )

    def setup_remove_gb_solvent_simulation(self, restraint_key: str, prmtop: FileID):
        temp_args = copy(self.no_gb_args)

        # Desolvation of receptor

        if self.system_type == "receptor":
            dirstruct_args = "dirstruct_apo"
            temp_args["state_label"] = "no_gb"
            temp_args["charge"] = 1.0
            temp_args["filename"] = "state_4_prod"
            temp_args["runtype"] = (
                "Running production simulation in state 4: Receptor only"
            )

        else:
            dirstruct_args = "dirstruct_halo"

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
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
            )
        )

    def setup_lj_interations_simulation(self, restraint_key: str, prmtop: FileID):
        """_summary_

        Args:
            restraint_key (str): _description_
            prmtop (FileID): _description_
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
            )
        )

    def setup_gb_external_dielectric(
        self, restraint_key: str, prmtop: FileID, mdin: FileID, extdiel: float
    ):
        """_summary_

        Args:
            restraint_key (str): _description_
            dielectric (float): _description_
        """
        temp_args = copy(self.no_gb_args)

        temp_args["state_label"] = "gb_dielectric"
        temp_args["filename"] = "state_8_prod"
        temp_args["extdiel"] = extdiel
        temp_args["igb"] = f"igb_{self.config.intermidate_args.igb_solvent}"
        temp_args["igb_value"] = f"igb_{self.config.intermidate_args.igb_solvent}"
        temp_args["runtype"] = (
            f"Running production Simulation in state 8. Changing extdiel to: {extdiel}."
        )

        self.simulations.append(
            Simulation(
                executable=self.config.system_settings.executable,
                mpi_command=self.config.system_settings.mpi_command,
                num_cores=self.num_cores,
                prmtop=prmtop,
                incrd=self.binding_mode,
                input_file=mdin,
                restraint_file=self.restraints,
                directory_args=temp_args,
                system_type=self.system_type,
                dirstruct="dirstruct_halo",
                working_directory=self.config.system_settings.working_directory,
                restraint_key=restraint_key,
                memory=self.config.system_settings.memory,
                disk=self.config.system_settings.disk,
                sim_debug=self.config.workflow.debug,
            )
        )

    def setup_apply_restraint_windows(
        self,
        restraint_key: str,
        exponent_conformational: float,
        exponent_orientational: Optional[float] = None,
    ):
        temp_args = copy(self.no_gb_args)

        temp_args["state_label"] = "lambda_window"
        temp_args["extdiel"] = 78.5
        temp_args["charge"] = 1.0
        temp_args["filename"] = f"state_2_{exponent_conformational}_prod"
        temp_args["igb"] = f"igb_{self.config.intermidate_args.igb_solvent}"
        temp_args["igb_value"] = self.config.intermidate_args.igb_solvent
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
            )
        )

    @property
    def ligand_charge_args(self) -> dict:
        """_summary_

        Returns:
            dict: _description_
        """

        return {
            "topology": self.topology,
            "state_label": "electrostatics",
            "conformational_restraint": self.max_conformational_exponent,
            # "igb": "igb_6",
            # "extdiel": 0.0,
            # "charge": charge,
            "igb_value": f"igb_{self.config.intermidate_args.igb_solvent}",
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
        """_summary_

        Returns:
            _type_: _description_
        """
        return {
            "topology": self.topology,
            "state_label": "endstate",
            "charge": 1.0,
            "extdiel": 78.5,
            "igb": f"igb_{self.config.intermidate_args.igb_solvent}",
            "igb_value": self.config.intermidate_args.igb_solvent,
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_charge": 1.0,
            "traj_extdiel": 78.5,
            "topdir": self.config.system_settings.top_directory_path,
            "traj_igb": f"igb_{self.config.intermidate_args.igb_solvent}",
            "filename": "state_2_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
            "trajectory_restraint_orenrest": 0.0,
        }
