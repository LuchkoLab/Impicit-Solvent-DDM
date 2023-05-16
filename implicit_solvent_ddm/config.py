import os
import random
import re
import shutil
import string
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

import numpy as np
import parmed as pmd
import pytraj as pt
import yaml
from toil.common import FileID, Toil
from toil.job import Job

WORKDIR = os.getcwd()


@dataclass
class Workflow:
    """Base workflow procedure."""

    setup_workflow: bool = True
    post_treatment: bool = True
    run_endstate_method: bool = True
    gb_extdiel_windows: bool = True
    end_state_postprocess: bool = True
    add_ligand_conformational_restraints: bool = True
    remove_GB_solvent_ligand: bool = True
    remove_ligand_charges: bool = True
    add_receptor_conformational_restraints: bool = True
    remove_GB_solvent_receptor: bool = True
    ignore_receptor_endstate: bool = False
    complex_ligand_exclusions: bool = True
    complex_turn_off_exclusions: bool = True
    complex_turn_on_ligand_charges: bool = True
    complex_turn_on_GB_enviroment: bool = True
    complex_remove_restraint: bool = True
    post_analysis_only: bool = False
    vina_dock: bool = False

    @classmethod
    def from_config(cls: Type["Workflow"], obj: dict):
        if "workflow_jobs" in obj.keys():
            return cls(**obj["workflow_jobs"])
        else:
            return cls()


@dataclass
class NumberOfCoresPerSystem:
    """Number of cores specified for respected
    host, guest and complex simulation.
    User can specify the number of cores required to
    run each system simulation.
    """

    complex_ncores: int
    ligand_ncores: int
    receptor_ncores: int

    @classmethod
    def from_config(cls: Type["NumberOfCoresPerSystem"], obj: dict):
        return cls(
            complex_ncores=obj["complex_ncores"],
            ligand_ncores=obj["ligand_ncores"],
            receptor_ncores=obj["receptor_ncores"],
        )


@dataclass
class SystemSettings:
    """System required parameters.
    Paramters to denoted whether to run
    CPU or GPU jobs. Support HPC schedular
    SLURM.

    Attributes:
    ----------
    mpi_command: str
        HPC schedular command [srun]
    working_directory: str
        User working directory.
    executable: str
        AMBER executable [sander, pmemd]
    CUDA: bool
        Run on GPU
    memory: Optional[Union[int, str]]
        Required memory to run individual MD simulation.
    disk: Optional[Union[int, str]]
        Required disk space needed for individual MD simulation input/output
    """

    mpi_command: str
    working_directory: str = WORKDIR
    executable: str = "sander"
    output_directory_name: str = "mdgb"
    CUDA: bool = field(default=False)
    memory: Optional[Union[int, str]] = field(default="5G")
    disk: Optional[Union[int, str]] = field(default="5G")

    @property
    def top_directory_path(self):

        return os.path.join(self.working_directory, self.output_directory_name)

    @classmethod
    def from_config(cls: Type["SystemSettings"], obj: dict):
        return cls(**obj)


@dataclass
class ParameterFiles:
    """
    AMBER parameter files for workflow.
    Requires a complex parameter file (.parm7) and complex coordinate files (AMBER)
    trajectories (.nc, .ncrst, .rst7 ect)
    """

    complex_parameter_filename: Union[str, FileID]
    complex_coordinate_filename: Union[str, FileID]
    ligand_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    ligand_coordinate_filename: Optional[Union[str, FileID]] = field(default=None)
    receptor_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    receptor_coordinate_filename: Optional[Union[str, FileID]] = field(default=None)

    complex_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)
    ligand_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)
    receptor_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)

    ignore_unique_naming: bool = False

    def __post_init__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        # check complex is a valid structure
        # ASK LUCHKO HOW TO CHECK FOR VALID STRUCTURES
        complex_traj = pt.iterload(
            self.complex_coordinate_filename, self.complex_parameter_filename
        )
        # print(self.complex_traj)
        pt.check_structure(traj=complex_traj)

        # if self.ignore_unique_naming == False:
        #     self._create_unqiue_fileID()

    @classmethod
    def from_config(cls: Type["ParameterFiles"], obj: dict):
        return cls(**obj)

    def get_inital_coordinate(self):

        solu_complex = re.sub(
            r"\..*", "", os.path.basename(self.complex_coordinate_filename)
        )
        solu_receptor = re.sub(r"\..*", "", os.path.basename(self.receptor_coordinate_filename))  # type: ignore
        solu_ligand = re.sub(r"\..*", "", os.path.basename(self.ligand_coordinate_filename))  # type: ignore

        complex_traj = pt.iterload(
            self.complex_coordinate_filename, self.complex_parameter_filename
        )
        print(complex_traj)
        receptor_traj = pt.iterload(
            self.receptor_coordinate_filename, self.receptor_parameter_filename
        )
        print(receptor_traj)
        ligand_traj = pt.iterload(
            self.ligand_coordinate_filename, self.ligand_parameter_filename
        )

        pt.write_traj(
            f"{self.tempdir.name}/{solu_complex}.ncrst",
            complex_traj,
            frame_indices=[0],
        )
        pt.write_traj(
            f"{self.tempdir.name}/{solu_receptor}_.ncrst",
            receptor_traj,
            frame_indices=[0],
        )
        pt.write_traj(
            f"{self.tempdir.name}/{solu_ligand}_.ncrst", ligand_traj, frame_indices=[0]
        )

        self.complex_initial_coordinate = f"{self.tempdir.name}/{solu_complex}.ncrst.1"
        self.receptor_initial_coordinate = (
            f"{self.tempdir.name}/{solu_receptor}_.ncrst.1"
        )
        self.ligand_initial_coordinate = f"{self.tempdir.name}/{solu_ligand}_.ncrst.1"

    def _create_unqiue_fileID(self):
        """
        Creates unique 3-letter Ascii filename for ligand and receptor filenames.
        """
        unique_id = "".join(random.choice(string.ascii_letters) for x in range(3))

        ligand_basename = re.sub(r"\..*", "", os.path.basename(self.ligand_parameter_filename))  # type: ignore
        receptor_basename = re.sub(r"\..*", "", os.path.basename(self.receptor_parameter_filename))  # type: ignore

        if self.ligand_parameter_filename is not None:
            unique_ligand_ID = os.path.join(
                self.tempdir.name, f"{ligand_basename}_{unique_id}.parm7"
            )
            shutil.copyfile(self.ligand_parameter_filename, unique_ligand_ID)  # type: ignore
            self.ligand_parameter_filename = unique_ligand_ID

        if self.receptor_parameter_filename is not None:

            unique_receptor_ID = os.path.join(
                self.tempdir.name,
                f"{receptor_basename}-{ligand_basename}_{unique_id}.parm7",
            )

            shutil.copyfile(self.receptor_parameter_filename, unique_receptor_ID)  # type: ignore

            self.receptor_parameter_filename = unique_receptor_ID

    def toil_import_parmeters(self, toil):

        self.complex_parameter_filename = str(
            toil.import_file(
                "file://" + os.path.abspath(self.complex_parameter_filename)
            )
        )
        self.complex_coordinate_filename = str(
            toil.import_file(
                "file://" + os.path.abspath(self.complex_coordinate_filename)
            )
        )
        if self.ligand_coordinate_filename is not None:
            self.ligand_coordinate_filename = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.ligand_coordinate_filename)
                )
            )
        if self.ligand_parameter_filename is not None:
            self.ligand_parameter_filename = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.ligand_parameter_filename)
                )
            )

        if self.receptor_parameter_filename is not None:
            self.receptor_parameter_filename = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.receptor_parameter_filename)
                )
            )
        if self.receptor_coordinate_filename is not None:
            self.receptor_coordinate_filename = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.receptor_coordinate_filename)
                )
            )
        if self.complex_initial_coordinate is not None:
            self.complex_initial_coordinate = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.complex_initial_coordinate)
                )
            )
        if self.ligand_initial_coordinate is not None:
            self.ligand_initial_coordinate = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.ligand_initial_coordinate)
                )
            )
        if self.receptor_initial_coordinate is not None:
            self.receptor_initial_coordinate = str(
                toil.import_file(
                    "file://" + os.path.abspath(self.receptor_initial_coordinate)
                )
            )


@dataclass
class AmberMasks:
    """AMBER masks to denote receptor/host and guest atoms"""

    receptor_mask: str
    ligand_mask: str

    @classmethod
    def from_config(cls: Type["AmberMasks"], obj: dict):
        return cls(receptor_mask=obj["receptor_mask"], ligand_mask=obj["ligand_mask"])


@dataclass
class REMD:
    """Parameter to run Replica Exchange Molecular Dynamics.

    Attributes:
    -----------
    remd_template_mdin: str
        A template to run REMD simulation.
    equil_template_mdin: str
        A template to run equilibration/relaxtion simulation prior to REMD run.
    temperatures: list[int]
        A list of replica temperatures for REMD runs.
    ngroups: int
        Number of individual simulations
    nthreads_complex: int
        Number of processors will be evenly divided among (number of groups) individual simulation
        for the complex simulations
    nthreads_receptor: int
        Number of processors will be evenly divided among (number of groups) individual simulation
        for the receptor/host simulations
    nthreads_ligand: int
        Number of processors will be evenly divided among (number of groups) individual simulation
        for the ligand/guest simulations
    """

    remd_template_mdin: Union[FileID, str] = "remd.template"
    equil_template_mdin: Union[FileID, str] = "equil.template"
    temperatures: list[int] = field(default_factory=list)
    ngroups: int = field(init=False)
    nthreads_complex: int = 0
    nthreads_receptor: int = 0
    nthreads_ligand: int = 0
    # nthreads: int = 0

    
    def __post_init__(self):
        
        #number of copies should equal to length of temperatures     
        self.ngroups = len(self.temperatures)
    
    
    @classmethod
    def from_config(cls: Type["REMD"], obj: dict):
        return cls(
            remd_template_mdin=obj["endstate_arguments"]["remd_template_mdin"],
            equil_template_mdin=obj["endstate_arguments"]["equilibrate_mdin_template"],
            temperatures=obj["endstate_arguments"]["temperatures"],
            nthreads_complex=obj["endstate_arguments"]["nthreads_complex"],
            nthreads_receptor=obj["endstate_arguments"]["nthreads_receptor"],
            nthreads_ligand=obj["endstate_arguments"]["nthreads_ligand"],
        )

    def toil_import_replica_mdin(self, toil: Toil):

        self.remd_template_mdin = toil.import_file(
            "file://" + os.path.abspath(self.remd_template_mdin)
        )
        self.equil_template_mdin = toil.import_file(
            "file://" + os.path.abspath(self.equil_template_mdin)
        )


@dataclass
class FlatBottomRestraints:
    """Paramters for flat bottom restraints

      Attributes
    ----------
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an
        This can temporarily be left undefined, but ``_missing_parameters()``
        will be called which will define receptor atoms by the provided AMBER masks.
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain.
        This can temporarily be left undefined, but ``_missing_parameters()``
        will be called which will define ligand atoms by the provided AMBER masks.
    flat_bottom_width: float, optional
        The distance r0  at which the harmonic restraint is imposed.
        The well with a square bottom between r2 and r3, with parabolic sides out
        to a defined distance. This has an default value of 5 Å if not provided.
    harmonic_distance: float, optional
        The upper bound parabolic sides out to define distance
        (r1 and r4 for lower and upper bounds, respectively),
        and linear sides beyond that distance. This has an default
        value of 10 Å, if not provided.
    spring_constant: float
        The spring constant K in units compatible
        with kJ/mol*nm^2 f (default is 1 kJ/mol*nm^2).
    flat_bottom_restraints: dict, optional
        User provided {r1, r2, r3, r4, rk2, rk3} restraint
        parameters. This can be temporily left undefined, but
        ``_missing_parameters()`` will be called which which would
        define all the restraint parameters. See example down below.
    """

    flat_bottom_width: float = 5.0
    harmonic_distance: float = 10.0
    spring_constant: float = 1.0
    restrained_receptor_atoms: Optional[List[int]] = None
    restrained_ligand_atoms: Optional[List[int]] = None
    flat_bottom_restraints: Optional[
        dict[str, float]
    ] = None  # {r1: 0, r2: 0, r3: 10, r4: 20, rk2: 0.1, rk3: 0.1}

    @classmethod
    def from_config(cls: Type["FlatBottomRestraints"], obj: dict):
        if "restraints" in obj.keys():
            return cls(**obj["restraints"])
        else:
            return cls()


@dataclass
class EndStateMethod:
    endstate_method_type: Union[str,int]
    remd_args: REMD
    flat_bottom: FlatBottomRestraints

    def __post_init__(self):
        endstate_method_options = ["remd", "md", 0]
        if self.endstate_method_type not in endstate_method_options:
            raise NameError(
                f"'{self.endstate_method_type}' is not a valid endstate method. Options: {endstate_method_options}"
            )

    @classmethod
    def from_config(cls: Type["EndStateMethod"], obj: dict):
        if obj["endstate_method"] == 0:
            return cls(
                endstate_method_type=obj["endstate_method"],
                remd_args=REMD(),
                flat_bottom=FlatBottomRestraints.from_config(obj=obj),
            )
        elif obj["endstate_method"].lower() == "remd":
            return cls(
                endstate_method_type=str(obj["endstate_method"]).lower(),
                remd_args=REMD.from_config(obj=obj),
                flat_bottom=FlatBottomRestraints.from_config(obj=obj),
            )
        else:
            return cls(
                endstate_method_type=obj["endstate_method"],
                remd_args=REMD(),
                flat_bottom=FlatBottomRestraints.from_config(obj=obj),
            )


@dataclass
class IntermidateStatesArgs:
    exponent_conformational_forces: List[float]
    exponent_orientational_forces: List[float]
    restraint_type: int
    igb_solvent: int
    mdin_intermidate_config: str
    temperature: float
    charges_lambda_window: List[float] = field(default_factory=list)
    gb_extdiel_windows: List[float] = field(default_factory=list)

    guest_restraint_template: Optional[str] = None
    receptor_restraint_template: Optional[str] = None
    complex_conformational_template: Optional[str] = None
    complex_orientational_template: Optional[str] = None

    guest_restraint_files: List[Union[str, FileID]] = field(default_factory=list)
    receptor_restraint_files: List[Union[str, FileID]] = field(default_factory=list)
    complex_restraint_files: List[Union[str, FileID]] = field(default_factory=list)

    conformational_restraints_forces: np.ndarray = field(init=False)
    orientational_restriant_forces: np.ndarray = field(init=False)
    max_conformational_restraint: float = field(init=False)
    max_orientational_restraint: float = field(init=False)

    def __post_init__(self):

        # check charges lambda windows have a min value of 0
        if len(self.charges_lambda_window) == 0:
            self.charges_lambda_window = [0.0, 1.0]
        else:
            lower_upper_bound = [0.0, 1.0]
            self.charges_lambda_window = list(
                set(self.charges_lambda_window + lower_upper_bound)
            )

        # charges convert to float
        self.charges_lambda_window = [
            float(charge) for charge in self.charges_lambda_window
        ]

        if len(self.gb_extdiel_windows) > 0:

            if 0 in self.gb_extdiel_windows:
                self.gb_extdiel_windows.remove(0)
            if 1 in self.gb_extdiel_windows:
                self.gb_extdiel_windows.remove(1)

            self.gb_extdiel_windows = [
                float(78.5 * extdiel) for extdiel in self.gb_extdiel_windows
            ]

        self.conformational_restraints_forces = np.exp2(
            self.exponent_conformational_forces
        )

        self.orientational_restriant_forces = np.exp2(
            self.exponent_orientational_forces
        )

        self.max_conformational_restraint = max(self.conformational_restraints_forces)
        self.max_orientational_restraint = max(self.orientational_restriant_forces)

        self.mdin_intermidate_config = os.path.abspath(self.mdin_intermidate_config)

        with open(self.mdin_intermidate_config) as mdin_args:
            self.mdin_intermidate_config = yaml.safe_load(mdin_args)

        if (
            self.guest_restraint_template
            or self.receptor_restraint_template
            or self.complex_orientational_template
        ):

            # self.tempdir = "mdgb/restraints"
            self.tempdir = tempfile.TemporaryDirectory()
            # if not os.path.exists('mdgb/restraints'):
            #     os.makedirs('mdgb/restraints')

            for con_force, orient_force in zip(
                self.conformational_restraints_forces,
                self.orientational_restriant_forces,
            ):
                self.write_ligand_restraint(conformational_force=con_force)
                self.write_receptor_restraints(conformational_force=con_force)
                self.write_complex_restraints(
                    conformational_force=con_force, orientational_force=orient_force
                )

    @classmethod
    def from_config(cls: Type["IntermidateStatesArgs"], obj: dict):
        return cls(**obj)

    def write_ligand_restraint(self, conformational_force):
        filename = re.sub(r"\..*", "", os.path.basename(self.guest_restraint_template))  # type: ignore

        with open(self.guest_restraint_template) as f:  # type: ignore
            ligand_restraints = f.readlines()

        string_template = ""
        for line in ligand_restraints:
            if "frest" in line:
                line = line.replace("frest", str(conformational_force))

            string_template += line

        with open(
            f"{self.tempdir.name}/{filename}_{conformational_force}.RST", "w"
        ) as output:
            output.write(string_template)

        self.guest_restraint_files.append(f"{self.tempdir.name}/{filename}_{conformational_force}.RST")  # type: ignore

    def write_receptor_restraints(self, conformational_force):
        filename = re.sub(r"\..*", "", os.path.basename(self.receptor_restraint_template))  # type: ignore

        with open(self.receptor_restraint_template) as f:  # type: ignore
            receptor_restraints = f.readlines()

        string_template = ""
        for line in receptor_restraints:
            if "frest" in line:
                line = line.replace("frest", str(conformational_force))
            string_template += line

        with open(
            f"{self.tempdir.name}/{filename}_{conformational_force}.RST", "w"
        ) as output:
            output.write(string_template)

        self.receptor_restraint_files.append(f"{self.tempdir.name}/{filename}_{conformational_force}.RST")  # type: ignore

    def write_complex_restraints(self, conformational_force, orientational_force):

        filename = re.sub(r"\..*", "", os.path.basename(self.complex_orientational_template))  # type: ignore

        with open(self.complex_conformational_template) as f:  # type: ignore
            complex_conformational = f.readlines()

        with open(self.complex_orientational_template) as fH:  # type: ignore
            complex_orientational = fH.readlines()

        string_template = ""
        for line in complex_orientational:
            if "drest" in line:
                line = line.replace("drest", str(conformational_force))
            if "arest" in line:
                line = line.replace("arest", str(orientational_force))
            if "trest" in line:
                line = line.replace("trest", str(orientational_force))
            if "&end" in line:
                line = line.replace("&end", "")

            string_template += line

        for line in complex_conformational:
            if "frest" in line:
                line = line.replace("frest", str(conformational_force))
            string_template += line

        with open(
            f"{self.tempdir.name}/{filename}_{conformational_force}_{orientational_force}.RST",
            "w",
        ) as output:
            output.write(string_template)

        self.complex_restraint_files.append(f"{self.tempdir.name}/{filename}_{conformational_force}_{orientational_force}.RST")  # type: ignore

    def toil_import_user_restriants(self, toil: Toil):
        """
        import restraint files into Toil job store
        """
        for i, (guest_rest, receptor_rest, complex_rest) in enumerate(zip(self.guest_restraint_files, self.receptor_restraint_files, self.complex_restraint_files)):  # type: ignore
            self.guest_restraint_files[i] = toil.import_file("file://" + os.path.abspath(guest_rest))  # type: ignore
            self.receptor_restraint_files[i] = toil.import_file(("file://" + os.path.abspath(receptor_rest)))  # type: ignore
            self.complex_restraint_files[i] = toil.import_file(("file://" + os.path.abspath(complex_rest)))  # type: ignore

        # self.tempdir.cleanup()


@dataclass
class Config:
    """Encapsulate a user specified configuration using a data class.

        The configuration settings will be handle through identifiers rather than strings.

    Returns:
        Config: Return an Config object with define the config properties, sub-properties, and types.
    """

    workflow: Workflow
    system_settings: SystemSettings
    endstate_files: ParameterFiles
    num_cores_per_system: NumberOfCoresPerSystem
    amber_masks: AmberMasks
    endstate_method: EndStateMethod
    intermidate_args: IntermidateStatesArgs
    inputs: dict
    restraints: dict
    ignore_receptor: bool = False

    def __post_init__(self):
        self._config_sanitity_check()

        if self.endstate_method.endstate_method_type != 0:
            self.get_receptor_ligand_topologies()

        else:
            self.endstate_files.get_inital_coordinate()
            self.workflow.run_endstate_method = False

        if len(self.intermidate_args.gb_extdiel_windows) == 0:
            self.workflow.gb_extdiel_windows = False

        if self.endstate_files.receptor_coordinate_filename is not None:
            self.ignore_receptor = True 
        
    def _config_sanitity_check(self):
        # check if the amber mask are valid
        self._valid_amber_masks()
        self._check_endstate_method()
        self._check_missing_flatbottom_parameters()
        self._remd_target_temperature()
    def _check_endstate_method(self):

        if self.endstate_method.endstate_method_type == 0:
            if self.endstate_files.ligand_parameter_filename == None:
                raise ValueError(
                    f"user specified to not run an endstate simulation but did not provided ligand_parameter_filename/coordinate endstate files"
                )

    def _valid_amber_masks(self):
        """
        Simple check that the AMBER masks denote recpetor and ligand atoms from the complex file.
        """
        ligand_natoms = pt.strip(
            self.complex_pytraj_trajectory, self.amber_masks.receptor_mask
        ).n_atoms
        receptor_natoms = pt.strip(
            self.complex_pytraj_trajectory, self.amber_masks.ligand_mask
        ).n_atoms
        parm = pmd.amber.AmberFormat(self.endstate_files.complex_parameter_filename)
        # check if sum of ligand & receptor atoms = complex total num of atoms
        if self.complex_pytraj_trajectory.n_atoms != ligand_natoms + receptor_natoms:
            raise RuntimeError(
                f"""The sum of ligand/guest and receptor/host atoms != number of total complex atoms
                                number of ligand atoms: {ligand_natoms} + number of receptor atoms {receptor_natoms} != complex total atoms: {self.complex_pytraj_trajectory.n_atoms}
                                Please check if AMBER masks are correct ligand_mask: "{self.amber_masks.ligand_mask}" receptor_mask: "{self.amber_masks.receptor_mask}"
                                {self.endstate_files.complex_parameter_filename} residue lables are: {parm.parm_data['RESIDUE_LABEL']}"""
            )
    def _remd_target_temperature(self):
        if self.endstate_method.endstate_method_type == 'remd':
            if self.intermidate_args.temperature not in self.endstate_method.remd_args.temperatures:
               raise RuntimeError(
                f"""The specified temperature {self.intermidate_args.temperature} is not 
                in the list of temperatures ({self.endstate_method.remd_args.temperatures}) for running REMD.
                Note the specified temperature {self.intermidate_args.temperature} will be the target temperature 
                during REMD trajectory extraction.
                """
               )
     
        
    @classmethod
    def from_config(cls: Type["Config"], user_config: dict, ignore_unique_naming=False):
        return cls(
            workflow=Workflow.from_config(user_config),
            system_settings=SystemSettings.from_config(
                user_config["hardware_parameters"]
            ),
            endstate_files=ParameterFiles.from_config(
                user_config["endstate_parameter_files"]
            ),
            num_cores_per_system=NumberOfCoresPerSystem.from_config(
                user_config["number_of_cores_per_system"]
            ),
            amber_masks=AmberMasks.from_config(user_config["AMBER_masks"]),
            endstate_method=EndStateMethod.from_config(user_config["workflow"]),
            intermidate_args=IntermidateStatesArgs.from_config(
                user_config["workflow"]["intermidate_states_arguments"]
            ),
            inputs={},
            restraints={},
        )


    def _check_missing_flatbottom_parameters(self):
        pass

    @property
    def complex_pytraj_trajectory(self) -> pt.Trajectory:
        traj = pt.iterload(
            self.endstate_files.complex_coordinate_filename,
            self.endstate_files.complex_parameter_filename,
        )
        return traj

    @property
    def ligand_pytraj_trajectory(self) -> pt.Trajectory:
        return self.complex_pytraj_trajectory[self.amber_masks.ligand_mask]

    @property
    def receptor_pytraj_trajectory(self) -> pt.Trajectory:
        return self.complex_pytraj_trajectory[self.amber_masks.receptor_mask]

    def get_receptor_ligand_topologies(self):
        """
        Splits the complex into the ligand and receptor individual files.

        These parameters files are inputs for ENDSTATE simulation
        """
        self.tempdir = tempfile.TemporaryDirectory()

        if self.endstate_files.receptor_parameter_filename is None:

            receptor_traj = self.complex_pytraj_trajectory[
                self.amber_masks.receptor_mask
            ]
            receptor_filename = os.path.join(
                self.tempdir.name, f"{self.amber_masks.receptor_mask.strip(':')}"
            )
            pt.write_parm(f"{receptor_filename}.parm7", receptor_traj.top)
            pt.write_traj(f"{receptor_filename}.ncrst", receptor_traj)
            self.endstate_files.receptor_parameter_filename = os.path.abspath(
                f"{receptor_filename}.parm7"
            )
            self.endstate_files.receptor_coordinate_filename = os.path.abspath(
                f"{receptor_filename}.ncrst.1"
            )

        if self.endstate_files.ligand_parameter_filename is None:
            # get ligand trajectory coordinate from provided complex.parm7
            ligand_traj = self.complex_pytraj_trajectory[self.amber_masks.ligand_mask]
            ligand_name = self.amber_masks.ligand_mask.strip(":")

            ligand_filename = os.path.join(self.tempdir.name, ligand_name)

            print("lignad_filename", ligand_filename)
            pt.write_parm(f"{ligand_filename}.parm7", ligand_traj.top)
            pt.write_traj(f"{ligand_filename}.ncrst", ligand_traj)

            self.endstate_files.ligand_parameter_filename = os.path.abspath(
                f"{ligand_filename}.parm7"
            )
            self.endstate_files.ligand_coordinate_filename = os.path.abspath(
                f"{ligand_filename}.ncrst.1"
            )

        self.endstate_files._create_unqiue_fileID()


def workflow(job, config: Config):

    tempdir = job.fileStore.getLocalTempDir()

    job.fileStore.readGlobalFile(
        config.endstate_files.complex_coordinate_filename,
        userPath=os.path.join(
            tempdir, os.path.basename(config.endstate_files.complex_coordinate_filename)
        ),
    )
    return 1


if __name__ == "__main__":

    import yaml

    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "OFF"
    options.clean = "always"
    with open("/nas0/ayoub/Impicit-Solvent-DDM/config_files/no_restraints.yaml") as fH:
        yaml_config = yaml.safe_load(fH)

    with Toil(options) as toil:

        config = Config.from_config(yaml_config)
        # # print(config)
        # # config.endstate_files.get_inital_coordinate()
        # if config.endstate_method.endstate_method_type != 0:
        #     config.get_receptor_ligand_topologies()
        # else:
        #     config.endstate_files.get_inital_coordinate()
        #     config.intermidate_args.toil_import_user_restriants(toil=toil)

        # config.endstate_files.toil_import_parmeters(toil=toil)
        # config.endstate_method.remd_args.toil_import_replica_mdin(toil=toil)

        print(config.intermidate_args)
        print(config.endstate_method.remd_args)
        # config.endstate_method.remd_args.toil_import_replica_mdins(toil=toil)
        # boresch_p = list(config.boresch_parameters.__dict__.values())

        # toil.start(Job.wrapJobFn(workflow, config))
        # print(config.intermidate_args)
