"""
Dataclass to make life a little easier, which defines config properties, sub-properties, and types in a config.py file.
Using a dataclass rather a dictionary ensures all key values pairs are read in (YAML config) before initiating the workflow.
"""
import os
import random
import re
import shutil
import string
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union, Dict
from pathlib import Path

import numpy as np
import parmed as pmd
import pytraj as pt
import yaml
from toil.common import FileID, Toil
from toil.job import Job

WORKDIR = os.getcwd()


@dataclass
class Workflow:
    """
    Defines the configuration for a modular computational workflow.

    Each boolean attribute represents whether a particular step or behavior
    in the workflow is enabled. This class can be initialized directly or
    constructed from a configuration dictionary using `from_config()`.
    """
    setup_workflow: bool = True
    consolidate_output: bool = True
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
    run_post_analysis: bool = True
    plot_overlap_matrix: bool = False 
    post_analysis_only: bool = False
    vina_dock: bool = False
    restart: bool = False
    debug: bool = False

    @classmethod
    def from_config(cls: Type["Workflow"], obj: dict):
        """
        Create a Workflow instance from a configuration dictionary.

        Expects the dictionary to have a 'workflow_jobs' key containing a
        sub-dictionary with workflow configuration flags.

        Args:
            obj (dict): Configuration dictionary, typically parsed from a YAML or JSON file.

        Returns:
            Workflow: An instance configured according to the input.
        """
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

    Attributes:
    -----------
    complex_ncores: int
        Number of processors to be used for a single complex intermediate molecular dynamics simulation.
    ligand_ncores: int
        Number of processors to be used for a single ligand intermediate molecular dynamics simulation.
    receptor_ncores: int
        Number of processors to be used for a single receptor intermediate molecular dynamics simulation.
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
    """
    SystemSettings defines system-level parameters for MD simulations.

    This includes runtime settings for hardware (e.g., CPU vs GPU),
    SLURM-based scheduling, executable selection, and resource requirements.

    Attributes
    ----------
    mpi_command : str
        MPI command for parallel execution (e.g., 'srun').
    working_directory : str
        Path to the main working directory.
    cache_directory_output : str
        Path for storing cached output files.
    executable : str
        AMBER executable to use (e.g., 'sander', 'pmemd').
    output_directory_name : str
        Directory name for MD output files.
    CUDA : bool
        If True, run using GPU resources.
    num_accelerators : int
        Number of GPUs requested. If 0 and CUDA is True, auto-detects available GPUs.
    memory : Optional[Union[int, str]]
        Memory required for job (e.g., '5G').
    disk : Optional[Union[int, str]]
        Disk space required for job (e.g., '5G').
    """

    mpi_command: str = field(default=None)
    working_directory: str = WORKDIR
    cache_directory_output: str = WORKDIR
    executable: str = "sander"
    output_directory_name: str = "mdgb"
    CUDA: bool = field(default=False)
    num_accelerators: int = field(default=0)
    memory: Optional[Union[int, str]] = field(default="5G")
    disk: Optional[Union[int, str]] = field(default="5G")
    
    
    def __post_init__(self):
        self.working_directory = os.path.abspath(self.working_directory)
        self.cache_directory_output = os.path.abspath(self.cache_directory_output)
        if self.CUDA and self.num_accelerators == 0:
            # Set default to 1 GPU per job for better distribution
            self.num_accelerators = 1

    @property
    def top_directory_path(self):
        return os.path.join(self.working_directory, self.output_directory_name)

    @classmethod
    def from_config(cls: Type["SystemSettings"], obj: dict):
        return cls(**obj)


@dataclass
class ParameterFiles:
    """
    Stores and manages AMBER parameter and coordinate file paths for MD simulations.

    Includes support for optional ligand and receptor input files, and handles temporary
    coordinate file generation and naming enforcement if requested.
    """
    # Required
    complex_parameter_filename: Union[str, FileID]
    complex_coordinate_filename: Union[str, FileID]

    # Optional components
    ligand_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    ligand_coordinate_filename: Optional[Union[str, FileID]] = field(default=None)
    receptor_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    receptor_coordinate_filename: Optional[Union[str, FileID]] = field(default=None)

    # Internal: Generated coordinates
    complex_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)
    ligand_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)
    receptor_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)

    # Misc
    ignore_unique_naming: bool = False

    def __post_init__(self):
        """
        Initialize a temporary directory for writing coordinate files.
        File loading and structure validation should be done separately,
        not at initialization.
        """
        self.tempdir = tempfile.TemporaryDirectory()
        # check complex is a valid structure
        complex_traj = pt.iterload(
            self.complex_coordinate_filename, self.complex_parameter_filename
        )
        pt.check_structure(traj=complex_traj)


    @classmethod
    def from_config(cls: Type["ParameterFiles"], obj: dict)->"ParameterFiles":
        """
        Instantiate a ParameterFiles object from a configuration dictionary.

        Parameters
        ----------
        obj : dict
            Dictionary with keys matching the attributes of ParameterFiles.

        Returns
        -------
        ParameterFiles
            A fully initialized instance of the class.
        """
        return cls(**obj)

    def get_inital_coordinate(self):
        """
        Extract and write the first frame from complex, receptor, and ligand trajectories
        into coordinate files in a temporary directory.

        Sets the corresponding *_initial_coordinate attributes to the written paths.
        """
        def _get_basename(filename):
            return re.sub(r"\..*", "", Path(str(filename)).name)
        
        def _write_initial_frame(traj, basename, suffix=""):
            output_path = Path(self.tempdir.name) / f"{basename}{suffix}.ncrst"
            pt.write_traj(str(output_path), traj, frame_indices=[0])
            return str(output_path.with_suffix(".ncrst.1"))
        
        # Complex is required
        complex_traj = pt.iterload(str(self.complex_coordinate_filename), str(self.complex_parameter_filename))
        base_complex = _get_basename(self.complex_coordinate_filename)
        self.complex_initial_coordinate = _write_initial_frame(complex_traj, base_complex)


        # Receptor (optional)
        if self.receptor_coordinate_filename and self.receptor_parameter_filename:
            receptor_traj = pt.iterload(str(self.receptor_coordinate_filename), str(self.receptor_parameter_filename))
            base_receptor = _get_basename(self.receptor_coordinate_filename)
            self.receptor_initial_coordinate = _write_initial_frame(receptor_traj, base_receptor, "_")
        
        # Ligand (optional)
        if self.ligand_coordinate_filename and self.ligand_parameter_filename:
            ligand_traj = pt.iterload(str(self.ligand_coordinate_filename), str(self.ligand_parameter_filename))
            base_ligand = _get_basename(self.ligand_coordinate_filename)
            self.ligand_initial_coordinate = _write_initial_frame(ligand_traj, base_ligand, "_")
      


    def _create_unique_fileID(self):
        """
        Generates unique 3-letter ASCII suffixes for ligand and receptor parameter filenames.

        Copies original files into the temporary directory with modified names to avoid collisions,
        and updates the internal parameter filenames to point to the new files.
        """
        def get_basename(path):
            return re.sub(r"\..*", "", Path(str(path)).name)

        unique_id = ''.join(random.choices(string.ascii_letters, k=3))

        if self.ligand_parameter_filename:
            ligand_base = get_basename(self.ligand_parameter_filename)
            ligand_path = Path(self.tempdir.name) / f"{ligand_base}_{unique_id}.parm7"
            shutil.copyfile(str(self.ligand_parameter_filename), ligand_path)
            self.ligand_parameter_filename = str(ligand_path)

        if self.receptor_parameter_filename:
            receptor_base = get_basename(self.receptor_parameter_filename)
            ligand_base = get_basename(self.ligand_parameter_filename) if self.ligand_parameter_filename else "LIG"
            receptor_path = Path(self.tempdir.name) / f"{receptor_base}-{ligand_base}_{unique_id}.parm7"
            shutil.copyfile(str(self.receptor_parameter_filename), receptor_path)
            self.receptor_parameter_filename = str(receptor_path)

    def toil_import_parameters(self, toil:Toil)->None:
        """
        Imports all parameter and coordinate files into the Toil job store and updates
        internal attributes with their corresponding FileStore IDs.
        """
        def import_if_not_none(attr: str):
            value = getattr(self, attr)
            if value is not None:
                file_path = Path(value).resolve()
                imported = toil.import_file(f"file://{file_path}")
                setattr(self, attr, str(imported))

        file_attributes = [
            "complex_parameter_filename",
            "complex_coordinate_filename",
            "ligand_parameter_filename",
            "ligand_coordinate_filename",
            "receptor_parameter_filename",
            "receptor_coordinate_filename",
            "complex_initial_coordinate",
            "ligand_initial_coordinate",
            "receptor_initial_coordinate",
        ]

        for attr in file_attributes:
            import_if_not_none(attr)


@dataclass
class AmberMasks:
    """AMBER masks to denote receptor/host and guest atoms

    Attributes:
    -----------
    receptor_mask: str
        Amber mask syntax to select receptor atoms
    ligand_mask: str
        Amber mask syntax to select ligand atoms
    """

    receptor_mask: str
    ligand_mask: str

    @classmethod
    def from_config(cls: Type["AmberMasks"], obj: dict):
        return cls(receptor_mask=obj["receptor_mask"], ligand_mask=obj["ligand_mask"])


@dataclass
class REMD:
    """
    Parameters for running Replica Exchange Molecular Dynamics (REMD).

    Attributes
    ----------
    remd_template_mdin : str or FileID
        Path to the REMD input template.
    equil_template_mdin : str or FileID
        Path to the equilibration input template used prior to REMD.
    temperatures : list of int
        List of replica temperatures for REMD.
    ngroups : int
        Number of replica groups (set automatically from `temperatures`).
    nthreads_complex : int
        Threads per group for complex simulations.
    nthreads_receptor : int
        Threads per group for receptor/host simulations.
    nthreads_ligand : int
        Threads per group for ligand/guest simulations.
    """
    method_type: str 
    remd_template_mdin: Union[FileID, str] = "remd.template"
    equil_template_mdin: Union[FileID, str] = "equil.template"
    temperatures: List[int] = field(default_factory=list)
    ngroups: int = field(init=False)
    nthreads_complex: int = 0
    nthreads_receptor: int = 0
    nthreads_ligand: int = 0

    def __post_init__(self):
        # Automatically determine number of groups from the temperature list
        self.ngroups = len(self.temperatures)
        if self.method_type == "remd":
            self._mdin_sanity_check()

    @classmethod
    def from_config(cls: Type["REMD"], obj: dict, method_type: str):
        return cls(
            remd_template_mdin=obj["endstate_arguments"]["remd_template_mdin"],
            equil_template_mdin=obj["endstate_arguments"]["equilibrate_mdin_template"],
            temperatures=obj["endstate_arguments"]["temperatures"],
            nthreads_complex=obj["endstate_arguments"]["nthreads_complex"],
            nthreads_receptor=obj["endstate_arguments"]["nthreads_receptor"],
            nthreads_ligand=obj["endstate_arguments"]["nthreads_ligand"],
            method_type=method_type
        )



    def _mdin_sanity_check(self):
        """
        Ensures REMD and equilibration MDIN templates exist
        and contain the required placeholders: $temp and $restraint.

        Raises
        ------
        FileNotFoundError
            If a required MDIN template file is missing or invalid.
        RuntimeError
            If required placeholders are missing in the MDIN templates.
        """
        required_keys = ["$temp", "$restraint"]

        mdin_paths = {
            "REMD template": self.remd_template_mdin,
            "Equilibration template": self.equil_template_mdin,
        }

        for label, mdin_path in mdin_paths.items():
            if not os.path.isfile(mdin_path):
                raise FileNotFoundError(f"{label} file does not exist or is not a valid file: {mdin_path}")

            with open(mdin_path, 'r') as f:
                contents = f.read()

            for key in required_keys:
                if key not in contents:
                    raise RuntimeError(
                        f"\nMissing required key `{key}` in {label}: {mdin_path}\n"
                        f"Please include a line such as `temperature={key}` or `restraint={key}` "
                        f"so the workflow can dynamically insert the correct values."
                    )

    def toil_import_replica_mdin(self, toil: "Toil") -> None:
        """
        Imports the REMD and equilibration template MDIN files into the Toil job store.

        Updates internal attributes with the corresponding FileStore IDs.
        """
        def import_template(path: Union[str, FileID]) -> str:
            return str(toil.import_file(f"file://{Path(path).resolve()}"))

        self.remd_template_mdin = import_template(self.remd_template_mdin)
        self.equil_template_mdin = import_template(self.equil_template_mdin)



@dataclass
class BasicMD:
    """
    Parameters to run a basic Molecular Dynamics (MD) simulation.

    Attributes
    ----------
    md_template_mdin : str or FileID
        Path to the MD input template.
    """

    method_type: str #
    md_template_mdin: Union[FileID, str] = "md.template"
   
    def __post_init__(self):
        if self.method_type == "basic_md":
            self._mdin_sanity_check()

    @classmethod
    def from_config(cls: Type["BasicMD"], obj: dict, method_type: str) -> "BasicMD":
        """
        Instantiate BasicMD from a configuration dictionary.

        Parameters
        ----------
        obj : dict
            Configuration dictionary with nested "endstate_arguments" key.

        Returns
        -------
        BasicMD
            A populated instance of the class.
        """
        return cls(
            md_template_mdin=obj["endstate_arguments"]["md_template_mdin"],
            method_type=method_type
        )

    def _mdin_sanity_check(self):
        """
        Validates that the MDIN template contains the $restraint placeholder.

        Raises
        ------
        RuntimeError
            If `$restraint` is not found in the template file.
        """
        mdin_path = Path(self.md_template_mdin)

        if not mdin_path.is_file():
            raise FileNotFoundError(f"MDIN template not found: {mdin_path}")

        contents = mdin_path.read_text()

        if "$restraint" not in contents:
            raise RuntimeError(
                f"\nMissing '$restraint' placeholder in MDIN template: {mdin_path}\n"
                f"Please include a line such as 'restraint=$restraint' "
                f"so the workflow can dynamically insert the restraint file."
            )

    def toil_import_basic_mdin(self, toil: "Toil") -> None:
        """
        Imports the MD template input file into the Toil job store.

        Updates the internal `md_template_mdin` attribute with the FileStore ID.
        """
        self.md_template_mdin = toil.import_file(
            f"file://{Path(self.md_template_mdin).resolve()}"
        )
@dataclass
class FlatBottomRestraints:
    """
    Parameters for flat-bottom distance restraints between receptor and ligand atoms.

    Attributes
    ----------
    flat_bottom_width : float
        Distance (r0) at which the harmonic restraint begins.
        Default is 5.0 Å.
    harmonic_distance : float
        Upper bound (r4) for the restraint with linear forces beyond this distance.
        Default is 10.0 Å.
    spring_constant : float
        Force constant K in kJ/mol·nm². Default is 1.0.
    restrained_receptor_atoms : list of int, optional
        Atom indices to restrain on the receptor side.
    restrained_ligand_atoms : list of int, optional
        Atom indices to restrain on the ligand side.
    flat_bottom_restraints : dict, optional
        User-defined parameters {r1, r2, r3, r4, rk2, rk3}. If not provided,
        defaults are inferred using `_missing_parameters()`.
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
    def from_config(cls: Type["FlatBottomRestraints"], obj: dict) -> "FlatBottomRestraints":
        """
        Instantiate FlatBottomRestraints from a config dictionary.

        Parameters
        ----------
        obj : dict
            Dictionary containing a "restraints" section.

        Returns
        -------
        FlatBottomRestraints
            Initialized instance with parameters from config or defaults.
        """
        return cls(**obj["restraints"]) if "restraints" in obj else cls()



@dataclass
class EndStateMethod:
    """
    Defines the strategy for simulating endstates: either a long MD simulation or Replica Exchange Molecular Dynamics (REMD).

    Users can choose from the following endstate methods:
    - "basic_md": A single long molecular dynamics simulation.
    - "remd": A replica exchange molecular dynamics (REMD) protocol.
    - 0: A user-defined or fallback mode with minimal setup.

    Attributes
    ----------
    endstate_method_type : str or int
        One of: "remd", "basic_md", or 0 (user-defined).
    remd_args : REMD
        Configuration options specific to REMD (only used if method is "remd").
    basic_md_args : BasicMD
        Configuration options specific to long MD (only used if method is "basic_md").
    flat_bottom : FlatBottomRestraints
        Flat-bottom restraint parameters applied in either method, if used.
    """

    endstate_method_type: Union[str, int]
    remd_args: REMD
    basic_md_args: BasicMD
    flat_bottom: FlatBottomRestraints

    def __post_init__(self):
        valid_options = ["remd", "basic_md", 0]
        if self.endstate_method_type not in valid_options:
            raise NameError(
                f"'{self.endstate_method_type}' is not a valid endstate method. "
                f"Valid options are: {valid_options}"
            )
    @classmethod
    def from_config(cls: Type["EndStateMethod"], obj: dict) -> "EndStateMethod":
        """
        Instantiate an EndStateMethod from a configuration dictionary.

        Parameters
        ----------
        obj : dict
            Configuration dictionary containing the endstate method and arguments.

        Returns
        -------
        EndStateMethod
            A populated instance based on the configuration.
        """
        method = obj.get("endstate_method")

        if method == 0:
            return cls(
                endstate_method_type=0,
                remd_args=REMD(method_type=method),
                basic_md_args=BasicMD(method_type=method),
                flat_bottom=FlatBottomRestraints.from_config(obj),
            )

        method = str(method).lower()
        if method == "remd":
            return cls(
                endstate_method_type=method,
                remd_args=REMD.from_config(obj,method_type=method),
                basic_md_args=BasicMD(method_type=method),
                flat_bottom=FlatBottomRestraints.from_config(obj),
            )
        else:
            return cls(
                endstate_method_type=method,
                remd_args=REMD(method_type=method),
                basic_md_args=BasicMD.from_config(obj, method_type=method),
                flat_bottom=FlatBottomRestraints.from_config(obj),
            )


@dataclass    
class IntermediateStateArgs:
    """
    Parameter arguments used for intermediate simulation steps within the thermodynamic cycle.

    Attributes
    ----------
    exponent_conformational_forces : list of float
        Exponents for scaling conformational restraints (base 2).
    exponent_orientational_forces : list of float
        Exponents for scaling orientational restraints (base 2).
    restraint_type : int
        Type of restraint protocol to apply.
    igb_solvent : int
        Implicit solvent model (e.g., 5 = GBn).
    mdin_intermediate_file : str
        Input MDIN file for the intermediate MD step.
    temperature : float
        Temperature of the simulation (in Kelvin).
    charges_lambda_window : list of float, optional
        Lambda windows for turning on/off partial charges.
    gb_extdiel_windows : list of float, optional
        Scaling factors for the GB external dielectric (normalized 0–1).
    min_degree_overlap : float
        Minimum acceptable overlap value for replica exchange (default 0.03).
    guest_restraint_template, receptor_restraint_template, complex_conformational_template, complex_orientational_template : str, optional
        Restraint template file paths.
    guest_restraint_files, receptor_restraint_files, complex_restraint_files : list of str/FileID
        Output restraint files to be used during simulation.
    """

    exponent_conformational_forces: List[float]
    exponent_orientational_forces: List[float]
    restraint_type: int
    igb_solvent: int
    mdin_intermediate_file: str
    temperature: float

    charges_lambda_window: List[float] = field(default_factory=list)
    gb_extdiel_windows: List[float] = field(default_factory=list)
    min_degree_overlap: float = 0.03

    guest_restraint_template: Optional[str] = None
    receptor_restraint_template: Optional[str] = None
    complex_conformational_template: Optional[str] = None
    complex_orientational_template: Optional[str] = None

    guest_restraint_files: List[Union[str, FileID]] = field(default_factory=list)
    receptor_restraint_files: List[Union[str, FileID]] = field(default_factory=list)
    complex_restraint_files: List[Union[str, FileID]] = field(default_factory=list)

    conformational_restraints_forces: np.ndarray = field(init=False)
    orientational_restraint_forces: np.ndarray = field(init=False)
    max_conformational_restraint: float = field(init=False)
    max_orientational_restraint: float = field(init=False)

    def __post_init__(self):
        # Ensure lambda windows include 0 and 1
        self.charges_lambda_window = list({0.0, 1.0, *self.charges_lambda_window})
        self.charges_lambda_window = [float(charge) for charge in self.charges_lambda_window]
        self.exponent_conformational_forces = [float(force) for force in self.exponent_conformational_forces]
        self.exponent_orientational_forces = [float(force) for force in self.exponent_orientational_forces]
        # Convert GB external dielectric scaling to dielectric values (if present)
        if self.gb_extdiel_windows:
            self.gb_extdiel_windows = [
                float(78.5 * val) for val in self.gb_extdiel_windows if val not in {0, 1}
            ]   

        # Apply 2^x to get force magnitudes
        self.conformational_restraints_forces = np.exp2(self.exponent_conformational_forces)
        self.orientational_restraint_forces = np.exp2(self.exponent_orientational_forces)     

        self.max_conformational_restraint = max(self.conformational_restraints_forces)
        self.max_orientational_restraint = max(self.orientational_restraint_forces)

        # Ensure mdin path is absolute
        self.mdin_intermediate_file = str(Path(self.mdin_intermediate_file).resolve())

        # Write restraint files if templates provided
        if any([
            self.guest_restraint_template,
            self.receptor_restraint_template,
            self.complex_orientational_template
        ]):
            self.tempdir = tempfile.TemporaryDirectory()

            for conf_force, orient_force in zip(
                self.conformational_restraints_forces,
                self.orientational_restraint_forces
            ):
                self.write_ligand_restraint(conformational_force=conf_force)
                self.write_receptor_restraints(conformational_force=conf_force)
                self.write_complex_restraints(
                    conformational_force=conf_force,
                    orientational_force=orient_force
                )

    @classmethod
    def from_config(cls: Type["IntermediateStateArgs"], obj: dict) -> "IntermediateStateArgs":
        """
        Create an IntermediateStateArgs instance from a configuration dictionary.

        Parameters
        ----------
        obj : dict
            Configuration dictionary with fields matching IntermediateStateArgs.

        Returns
        -------
        IntermediateStateArgs
            An initialized instance based on the configuration.
        """
        return cls(**obj)

    def _sanity_check_mdin(self):
        """
        Ensures that the required keywords are present in the user-provided MDIN template file.

        Required keys:
        - igb
        - $restraint
        - $extdiel
        - nmropt

        Raises
        ------
        ValueError
            If any required MDIN keys are missing from the template file.
        """
        required_keys = {"igb", "$restraint", "$extdiel", "nmropt"}

        mdin_path = Path(self.mdin_intermediate_file)

        if not mdin_path.is_file():
            raise FileNotFoundError(f"MDIN file not found: {mdin_path}")

        contents = mdin_path.read_text()

        found_keys = set(re.findall(r"igb|\$restraint|\$extdiel|nmropt", contents))
        missing_keys = required_keys - found_keys

        if missing_keys:
            raise ValueError(
                f"Missing required MDIN arguments: {sorted(missing_keys)}.\n"
                f"Please add them to your file: {mdin_path}"
            )
    
    def _write_restraint_template(
        self,
        template_paths: Union[str, tuple[str, str]],
        force_values: dict[str, float],
        output_list: list
    ):
        """
        Writes a processed restraint file with substituted force placeholders.

        Parameters
        ----------
        template_paths : str or tuple of str
            Path(s) to the input restraint template file(s). For complex restraints, provide a tuple.
        force_values : dict
            Mapping from placeholder (e.g. "frest", "drest") to numeric value.
        output_list : list
            List to which the output file path will be appended.
        """
        if isinstance(template_paths, str):
            template_paths = (template_paths,)  # promote to tuple

        all_lines = []

        for path in template_paths:
            with Path(path).open("r") as f:
                lines = f.readlines()

            for line in lines:
                for key, value in force_values.items():
                    if key in line:
                        line = line.replace(key, str(value))
                if "&end" in line:
                    line = line.replace("&end", "")
                all_lines.append(line)

        filename_stem = Path(template_paths[-1]).stem
        suffix = "_".join(str(v) for v in force_values.values())
        output_path = Path(self.tempdir.name) / f"{filename_stem}_{suffix}.RST"
        output_path.write_text("".join(all_lines))

        output_list.append(str(output_path))


    def write_ligand_restraint(self, conformational_force: float):
        """Generate a ligand restraint file using the conformational force value."""
        if self.guest_restraint_template:
            self._write_restraint_template(
                template_paths=self.guest_restraint_template,
                force_values={"frest": conformational_force},
                output_list=self.guest_restraint_files
            )


    def write_receptor_restraints(self, conformational_force: float):
        """Generate a receptor restraint file using the conformational force value."""
        if self.receptor_restraint_template:
            self._write_restraint_template(
                template_paths=self.receptor_restraint_template,
                force_values={"frest": conformational_force},
                output_list=self.receptor_restraint_files
            )


    def write_complex_restraints(self, conformational_force: float, orientational_force: float):
        """Generate a complex restraint file using both conformational and orientational force values."""
        if self.complex_conformational_template and self.complex_orientational_template:
            self._write_restraint_template(
                template_paths=(self.complex_conformational_template, self.complex_orientational_template),
                force_values={
                    "frest": conformational_force,
                    "drest": conformational_force,
                    "arest": orientational_force,
                    "trest": orientational_force
                },
                output_list=self.complex_restraint_files
            )

    def toil_import_user_restraints(self, toil: "Toil") -> None:
        """
        Imports user-defined ligand, receptor, and complex restraint files into the Toil job store.

        Updates each entry in the corresponding file list with the returned Toil FileStore ID.
        """
        for i, (guest_rest, receptor_rest, complex_rest) in enumerate(
            zip(self.guest_restraint_files, self.receptor_restraint_files, self.complex_restraint_files)
        ):
            self.guest_restraint_files[i] = toil.import_file(f"file://{Path(guest_rest).resolve()}")
            self.receptor_restraint_files[i] = toil.import_file(f"file://{Path(receptor_rest).resolve()}")
            self.complex_restraint_files[i] = toil.import_file(f"file://{Path(complex_rest).resolve()}")


    def toil_import_user_mdin(self, toil: "Toil") -> None:
        """
        Imports the user-provided MDIN file for intermediate states into the Toil job store.

        Updates the internal `mdin_intermediate_file` path with the corresponding FileStore ID.
        """
        self.mdin_intermediate_file = toil.import_file(
            f"file://{Path(self.mdin_intermediate_file).resolve()}"
        )

@dataclass
class Config:
    """
    Encapsulates user-specified configuration parameters for the simulation workflow.

    Attributes
    ----------
    workflow : Workflow
        Workflow toggles for all MD jobs and analysis steps.
    system_settings : SystemSettings
        Specifies compute environment (e.g., GPU, SLURM, memory).
    endstate_files : ParameterFiles
        File paths for complex, receptor, ligand inputs and outputs.
    num_cores_per_system : NumberOfCoresPerSystem
        Core allocation for each system type.
    amber_masks : AmberMasks
        Atom masks used to define key regions (e.g., ligand, receptor).
    endstate_method : EndStateMethod
        Specifies whether to use REMD, basic MD, or a custom endstate.
    intermediate_args : IntermediateStateArgs
        Parameters for the intermediate phase of the thermodynamic cycle.
    inputs : dict
        General inputs parsed from the configuration file.
    restraints : dict
        Restraint configuration (e.g., flat-bottom, Boresch).
    ignore_receptor : bool
        Whether to skip receptor preparation (auto-set if receptor coordinates are provided).
    """

    workflow: Workflow
    system_settings: SystemSettings
    endstate_files: ParameterFiles
    num_cores_per_system: NumberOfCoresPerSystem
    amber_masks: AmberMasks
    endstate_method: EndStateMethod
    intermediate_args: IntermediateStateArgs
    inputs: Dict
    restraints: Dict
    ignore_receptor: bool = False

    def __post_init__(self):
        self._config_sanity_check()

        # Decide on endstate procedure based on method type
        if self.endstate_method.endstate_method_type != 0:
            self.get_receptor_ligand_topologies()
        else:
            self.endstate_files.get_inital_coordinate()
            self.workflow.run_endstate_method = False

        # Disable GB dielectric scaling if no windows were specified
        if not self.intermediate_args.gb_extdiel_windows:
            self.workflow.gb_extdiel_windows = False

        # If receptor coordinates are provided, mark receptor as already prepared
        if self.endstate_files.receptor_coordinate_filename is not None:
            self.ignore_receptor = True

    def _config_sanity_check(self):
        """
        Perform validation checks on the overall configuration.

        This includes:
        - Validating AMBER masks
        - Verifying the endstate method
        - Checking for required flat-bottom parameters
        - Validating REMD temperature setup
        - Ensuring MDIN template contains required keywords
        """
        self._valid_amber_masks()
        self._check_endstate_method()
        self._check_missing_flatbottom_parameters()
        self._remd_target_temperature()
        self.intermediate_args._sanity_check_mdin()

    def _check_endstate_method(self):
        """
        Verifies that necessary ligand files are present when the user disables endstate simulations.
        """
        if self.endstate_method.endstate_method_type == 0:
            if self.endstate_files.ligand_parameter_filename is None:
                raise ValueError(
                    "User specified to skip endstate simulations (method_type=0), "
                    "but did not provide ligand parameter and coordinate files. "
                    "Please ensure `ligand_parameter_filename` is set in the config."
                )

    def _valid_amber_masks(self):
        """
        Validates that the provided AMBER masks correctly partition the complex into ligand and receptor atoms.
        """
        # Count atoms after stripping masks
        ligand_natoms = pt.strip(self.complex_pytraj_trajectory, self.amber_masks.receptor_mask).n_atoms
        receptor_natoms = pt.strip(self.complex_pytraj_trajectory, self.amber_masks.ligand_mask).n_atoms
        total_atoms = self.complex_pytraj_trajectory.n_atoms

        # Load parameter data to help with user-friendly reporting
        parm = pmd.amber.AmberFormat(self.endstate_files.complex_parameter_filename)
        residue_labels = parm.parm_data["RESIDUE_LABEL"]

        if total_atoms != ligand_natoms + receptor_natoms:
            raise RuntimeError(
                f"The sum of ligand and receptor atoms does not equal the total number of atoms in the complex.\n"
                f"  Ligand atoms     : {ligand_natoms}\n"
                f"  Receptor atoms   : {receptor_natoms}\n"
                f"  Total (expected) : {total_atoms}\n\n"
                f"Check your AMBER masks:\n"
                f"  Ligand mask  : '{self.amber_masks.ligand_mask}'\n"
                f"  Receptor mask: '{self.amber_masks.receptor_mask}'\n\n"
                f"Residue labels from parameter file:\n"
                f"  {residue_labels}"
            )

    def _remd_target_temperature(self):
        """
        Validates that the target temperature specified for analysis exists in the REMD temperature ladder.
        """
        if self.endstate_method.endstate_method_type == "remd":
            target_temp = self.intermediate_args.temperature
            available_temps = self.endstate_method.remd_args.temperatures

            if target_temp not in available_temps:
                raise RuntimeError(
                    f"The specified target temperature {target_temp} K is not present in the REMD temperature list.\n"
                    f"Available REMD temperatures: {available_temps}\n"
                    f"Please ensure the analysis temperature is included in the REMD schedule, as it will be used "
                    f"to extract the replica trajectory closest to this value."
                )

    @classmethod
    def from_config(
        cls: Type["Config"],
        user_config: dict,
        ignore_unique_naming: bool = False
    ) -> "Config":
        """
        Constructs a Config object from a user-provided configuration dictionary.

        Parameters
        ----------
        user_config : dict
            Dictionary containing all simulation configuration blocks.
        ignore_unique_naming : bool, optional
            Whether to skip enforcing unique restraint filenames. Defaults to False.

        Returns
        -------
        Config
            Fully initialized Config instance.
        """
        return cls(
            workflow=Workflow.from_config(user_config),
            system_settings=SystemSettings.from_config(user_config["hardware_parameters"]),
            endstate_files=ParameterFiles.from_config(user_config["endstate_parameter_files"]),
            num_cores_per_system=NumberOfCoresPerSystem.from_config(user_config["number_of_cores_per_system"]),
            amber_masks=AmberMasks.from_config(user_config["AMBER_masks"]),
            endstate_method=EndStateMethod.from_config(user_config["workflow"]),
            intermediate_args=IntermediateStateArgs.from_config(
                user_config["workflow"]["intermediate_states_arguments"]
            ),
            inputs={},       # Placeholder; can be populated post-init
            restraints={},   # Placeholder; can be populated post-init
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
        Extracts receptor and ligand topologies from the complex structure using AMBER masks.

        Writes parameter (.parm7) and coordinate (.ncrst) files if not already provided,
        and updates `endstate_files` with their absolute paths.
        """
        self.tempdir = tempfile.TemporaryDirectory()
        tempdir_path = Path(self.tempdir.name)

        # Handle receptor
        if self.endstate_files.receptor_parameter_filename is None:
            receptor_traj = self.complex_pytraj_trajectory[self.amber_masks.receptor_mask]
            receptor_basename = "receptor_system" #self.amber_masks.receptor_mask.strip(":")
            receptor_prefix = tempdir_path / receptor_basename

            pt.write_parm(str(receptor_prefix.with_suffix(".parm7")), receptor_traj.top)
            pt.write_traj(str(receptor_prefix.with_suffix(".ncrst")), receptor_traj)

            self.endstate_files.receptor_parameter_filename = str(receptor_prefix.with_suffix(".parm7").resolve())
            self.endstate_files.receptor_coordinate_filename = str(receptor_prefix.with_suffix(".ncrst.1").resolve())

        # Handle ligand
        if self.endstate_files.ligand_parameter_filename is None:
            ligand_traj = self.complex_pytraj_trajectory[self.amber_masks.ligand_mask]
            ligand_basename = "ligand_system" #self.amber_masks.ligand_mask.strip(":")
            ligand_prefix = tempdir_path / ligand_basename

            pt.write_parm(str(ligand_prefix.with_suffix(".parm7")), ligand_traj.top)
            pt.write_traj(str(ligand_prefix.with_suffix(".ncrst")), ligand_traj)

            self.endstate_files.ligand_parameter_filename = str(ligand_prefix.with_suffix(".parm7").resolve())
            self.endstate_files.ligand_coordinate_filename = str(ligand_prefix.with_suffix(".ncrst.1").resolve())

        # Ensure filenames are unique if needed
        self.endstate_files._create_unique_fileID()


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
    with open(
        "script_examples/config_files/basic_md_config.yaml"
    ) as fH:
        yaml_config = yaml.safe_load(fH)

    with Toil(options) as toil:
        config = Config.from_config(yaml_config)
        # # print(config)
        # # config.endstate_files.get_inital_coordinate()
        # if config.endstate_method.endstate_method_type != 0:
        #     config.get_receptor_ligand_topologies()
        # else:
        #     config.endstate_files.get_inital_coordinate()
        #     config.intermediate_args.toil_import_user_restriants(toil=toil)

        # config.endstate_files.toil_import_parmeters(toil=toil)
        # config.endstate_method.remd_args.toil_import_replica_mdin(toil=toil)
        config.intermediate_args.toil_import_user_mdin(toil=toil)
        print(config.intermediate_args.mdin_intermediate_file)
        print(config.intermediate_args)
        print(config.endstate_method.remd_args)

        # config.endstate_method.remd_args.toil_import_replica_mdins(toil=toil)
        # boresch_p = list(config.boresch_parameters.__dict__.values())

        # toil.start(Job.wrapJobFn(workflow, config))
        # print(config.intermediate_args)
