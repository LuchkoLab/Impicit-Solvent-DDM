import os
import random
import re
import shutil
import string
import tempfile
from dataclasses import dataclass, field
from enum import unique
from typing import List, Optional, Type, Union

import numpy as np
import parmed as pmd
import pytraj as pt
import yaml
from pydantic import NoneIsAllowedError
from toil.common import FileID, Toil
from toil.job import Job, JobFunctionWrappingJob


@dataclass 
class Workflow:
    setup_workflow: bool = True
    post_treatment: bool = True 
    run_endstate_method: bool = True
    end_state_postprocess: bool = False
    add_ligand_conformational_restraints: bool = True
    remove_GB_solvent_ligand: bool = True
    remove_ligand_charges: bool = True
    add_receptor_conformational_restraints: bool = True
    remove_GB_solvent_receptor: bool = True
    ignore_receptor_endstate: bool = False 
    complex_ligand_exclusions: bool = True
    complex_turn_off_exclusions: bool = True
    complex_turn_on_ligand_charges: bool = True
    complex_remove_restraint: bool = True
    @classmethod
    def from_config(cls: Type["Workflow"], obj:dict):
        if "workflow_jobs" in obj.keys():
            return cls(**obj["workflow_jobs"])
        else:
            return cls()   
        
    
    @classmethod
    def update_workflow(cls: Type["Workflow"], system: str, run_endstate: bool):
        
        if system == 'ligand':
            return cls(
                setup_workflow = False, 
                post_treatment=False, 
                end_state_postprocess = True,
                run_endstate_method = True, 
                ignore_receptor_endstate = True,
                add_receptor_conformational_restraints=False,
                remove_GB_solvent_receptor=False, 
                complex_ligand_exclusions = False, 
                complex_turn_off_exclusions = False, 
                complex_turn_on_ligand_charges = False, 
                complex_remove_restraint = False
            )
        elif system =='receptor':
            return cls(
                setup_workflow = False,
                post_treatment=False,
                end_state_postprocess = True,
                run_endstate_method = False,
                add_ligand_conformational_restraints = False, 
                remove_GB_solvent_ligand = False, 
                remove_ligand_charges = False, 
                complex_ligand_exclusions = False,
                complex_turn_off_exclusions = False, 
                complex_turn_on_ligand_charges = False, 
                complex_remove_restraint = False, 
            )
        elif system == 'complex':
            return cls(
                setup_workflow = False,
                post_treatment=False,
                end_state_postprocess = True,
                run_endstate_method = False,
                add_ligand_conformational_restraints = False,
                remove_GB_solvent_ligand = False,
                remove_ligand_charges  = False,
                ignore_receptor_endstate = True,
                add_receptor_conformational_restraints=False,
                remove_GB_solvent_receptor=False, 
                )
        else:
            raise Exception(f'system not valid: {system}')
    
@dataclass
class SystemSettings:
    executable: str
    mpi_command: str
    working_directory: str = 'no set' 
    CUDA: bool = field(default=False)
    memory: Optional[Union[int, str]] = field(default="5G")
    disk: Optional[Union[int, str]] = field(default="5G")
    @classmethod
    def from_config(cls: Type["SystemSettings"], obj:dict):
        return cls(**obj)

@dataclass 
class ParameterFiles:
    complex_parameter_filename: Union[str, FileID]
    complex_coordinate_filename: Union[str, FileID]
    ligand_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    ligand_coordinate_filename:  Optional[Union[str, FileID]]  = field(default=None)
    receptor_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    receptor_coordinate_filename: Optional[Union[str, FileID]] = field(default=None)
    
    complex_initial_coordinate:  Optional[Union[str, FileID]]  = field(default=None)
    ligand_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)
    receptor_initial_coordinate: Optional[Union[str, FileID]] = field(default=None)
    
    def __post_init__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        #check complex is a valid structure 
        #ASK LUCHKO HOW TO CHECK FOR VALID STRUCTURES
        complex_traj = pt.iterload(self.complex_coordinate_filename, self.complex_parameter_filename)
        pt.check_structure(traj=complex_traj)
        
        if self.receptor_parameter_filename is not None:
            self._create_unique_receptor_id()
            

    @classmethod 
    def from_config(cls: Type["ParameterFiles"], obj:dict):
       return cls(**obj)
   
    def get_inital_coordinate(self):
        solu_complex = re.sub(r"\..*", "", os.path.basename(self.complex_coordinate_filename))
        solu_receptor = re.sub(r"\..*", "", os.path.basename(self.receptor_coordinate_filename))  # type: ignore
        solu_ligand = re.sub(r"\..*", "", os.path.basename(self.ligand_coordinate_filename))  # type: ignore
        print(os.path.exists(self.receptor_parameter_filename))
        path = "mdgb/structs"
        if not os.path.exists(path):
            os.makedirs(path)
        
        complex_traj = pt.iterload(self.complex_coordinate_filename, self.complex_parameter_filename)
        print(complex_traj)
        receptor_traj = pt.iterload(self.receptor_coordinate_filename, self.receptor_parameter_filename)
        print(receptor_traj)
        ligand_traj = pt.iterload(self.ligand_coordinate_filename, self.ligand_parameter_filename)
        
        pt.write_traj(f"{path}/{solu_complex}.ncrst", complex_traj, frame_indices=[0])
        pt.write_traj(f"{path}/{solu_receptor}_.ncrst", receptor_traj, frame_indices=[0])
        pt.write_traj(f"{path}/{solu_ligand}_.ncrst", ligand_traj, frame_indices=[0])
        
        self.complex_initial_coordinate = f"{path}/{solu_complex}.ncrst.1"
        self.receptor_initial_coordinate = f"{path}/{solu_receptor}_.ncrst.1"
        self.ligand_initial_coordinate = f"{path}/{solu_ligand}_.ncrst.1"
    
    
    
    def _create_unique_receptor_id(self):
        '''
        Splits the complex into the ligand and receptor individual files.
        '''
        unique_id = ''.join(random.choice(string.ascii_letters) for x in range(3))
        
        ligand_basename = re.sub(r"\..*", "", os.path.basename(self.ligand_parameter_filename))  # type: ignore
        receptor_basename = re.sub(r"\..*", "", os.path.basename(self.receptor_parameter_filename))  # type: ignore
        unique_receptor_ID = os.path.join(self.tempdir.name, f"{receptor_basename}-{ligand_basename}_{unique_id}.parm7")
        
        unique_ligand_ID = os.path.join(self.tempdir.name, f"{ligand_basename}_{unique_id}.parm7")
        
        shutil.copyfile(self.receptor_parameter_filename, unique_receptor_ID)  # type: ignore
        shutil.copyfile(self.ligand_parameter_filename, unique_ligand_ID) # type: ignore
        
        self.receptor_parameter_filename = unique_receptor_ID
        self.ligand_parameter_filename = unique_ligand_ID
        
        print("test",os.path.exists(self.receptor_parameter_filename))
    
    
    def toil_import_parmeters(self, toil):
        
        self.complex_parameter_filename = str(
            toil.import_file(
                "file://"
                + os.path.abspath(self.complex_parameter_filename)
            )
        )
        self.complex_coordinate_filename = str(
            toil.import_file(
                "file://"
                + os.path.abspath(self.complex_coordinate_filename)
            )
        )
        if self.ligand_coordinate_filename is not None:
            self.ligand_coordinate_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(self.ligand_coordinate_filename)
                )
            )
        if self.ligand_parameter_filename is not None:
            self.ligand_parameter_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(self.ligand_parameter_filename)
                )
            )
        
        if self.receptor_parameter_filename is not None:
            self.receptor_parameter_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        self.receptor_parameter_filename
                    )
                )
            )
        if self.receptor_coordinate_filename is not None:
            self.receptor_coordinate_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        self.receptor_coordinate_filename
                    )
                )
            )
        if self.complex_initial_coordinate is not None:
            self.complex_initial_coordinate = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        self.complex_initial_coordinate
                    )
                )
            )
        if self.ligand_initial_coordinate is not None:
            self.ligand_initial_coordinate = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        self.ligand_initial_coordinate
                    )
                ) 
            )
        if self.receptor_initial_coordinate is not None:
            self.receptor_initial_coordinate = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        self.receptor_initial_coordinate
                    )
                ) 
            ) 
           
        
    
@dataclass
class NumberOfCoresPerSystem:
    complex_ncores: int 
    ligand_ncores: int
    receptor_ncores: int 
    
    @classmethod
    def from_config(cls: Type["NumberOfCoresPerSystem"], obj:dict):
        return cls(
            complex_ncores=obj["complex_ncores"],
            ligand_ncores=obj["ligand_ncores"],
            receptor_ncores=obj["receptor_ncores"]
        )
@dataclass
class AmberMasks:
    receptor_mask: str
    ligand_mask: str 
    
    @classmethod
    def from_config(cls: Type["AmberMasks"], obj:dict):
        return cls(
            receptor_mask=obj["receptor_mask"],
            ligand_mask=obj["ligand_mask"]
        )

@dataclass
class REMD:
    ngroups: int = 0
    target_temperature: float = 0.0 
    equilibration_replica_mdins: List[Union[FileID, str]] = field(default_factory=list)
    remd_mdins: List[str]  = field(default_factory=list)
    nthreads_complex: int = 0
    nthreads_receptor: int = 0
    nthreads_ligand: int = 0
    # nthreads: int = 0
    
    def  __post_init__(self):
        
        if len(self.equilibration_replica_mdins) != len(self.remd_mdins):
            raise RuntimeError(f"The size of {self.equilibration_replica_mdins} and {self.remd_mdins} do not match: {len(self.equilibration_replica_mdins)} | {len(self.remd_mdins)}")
            
                    
    @classmethod
    def from_config(cls: Type["REMD"], obj:dict):
        return cls(
            ngroups = obj["endstate_arguments"]["ngroups"],
            target_temperature = obj["endstate_arguments"]["target_temperature"],
            equilibration_replica_mdins = obj["endstate_arguments"]["equilibration_replica_mdins"],
            remd_mdins = obj["endstate_arguments"]["remd_mdins"],
            nthreads_complex = obj["endstate_arguments"]["nthreads_complex"],
            nthreads_receptor=obj["endstate_arguments"]["nthreads_receptor"],
            nthreads_ligand=obj["endstate_arguments"]["nthreads_ligand"]
            )

    def toil_import_replica_mdins(self, toil:Toil):
          for index, (equil_mdin, remd_mdin) in enumerate(
            zip(
                self.equilibration_replica_mdins,
                self.remd_mdins,
            )
        ):
            self.equilibration_replica_mdins[index] = toil.import_file("file://" + os.path.abspath(equil_mdin))  # type: ignore
            
            self.remd_mdins[index] = toil.import_file("file://" + os.path.abspath(remd_mdin))  # type: ignore

@dataclass
class EndStateMethod: 
    endstate_method_type: str 
    remd_args: REMD
    flat_bottom_restraints: Optional[List[float]] = None 
    
    def __post_init__(self):
        endstate_method_options = ["remd", "md", 0]
        if self.endstate_method_type not in endstate_method_options:
            raise NameError(f"'{self.endstate_method_type}' is not a valid endstate method. Options: {endstate_method_options}")
    
    @classmethod
    def from_config(cls: Type["EndStateMethod"], obj:dict):
        if obj["endstate_method"] == 0:
            return cls(
                endstate_method_type=obj["endstate_method"],
                remd_args = REMD(),
                flat_bottom_restraints=obj["endstate_arguments"]["flat_bottom_restraints"]
            )
        elif obj["endstate_method"].lower() == 'remd':
            return cls(
                endstate_method_type=str(obj["endstate_method"]).lower(),
                remd_args=REMD.from_config(obj=obj),
                flat_bottom_restraints=obj["endstate_arguments"]["flat_bottom_restraints"]
                )
        else:
            return cls(
                endstate_method_type=obj["endstate_method"],
                remd_args = REMD()
            )
@dataclass
class IntermidateStatesArgs: 
    exponent_conformational_forces: List[float]
    exponent_orientational_forces: List[float]
    restraint_type: int 
    igb_solvent: int 
    mdin_intermidate_config: str 
    temperature: float 
    
    guest_restraint_template: Optional[str] = None
    receptor_restraint_template: Optional[str] = None
    complex_conformational_template: Optional[str] = None
    complex_orientational_template: Optional[str] = None 
    
    guest_restraint_files: List[Union[str,FileID]] = field(default_factory=list)
    receptor_restraint_files: List[Union[str,FileID]] = field(default_factory=list)
    complex_restraint_files: List[Union[str,FileID]] = field(default_factory=list)
      
    conformational_restraints_forces: np.ndarray = field(init=False)
    orientational_restriant_forces: np.ndarray = field(init=False)
    max_conformational_restraint: float = field(init=False)
    max_orientational_restraint: float = field(init=False)
   
    
    def __post_init__(self):
        
        self.conformational_restraints_forces = np.exp2(self.exponent_conformational_forces)
        self.orientational_restriant_forces = np.exp2(self.exponent_orientational_forces)
        
        self.max_conformational_restraint = max(self.conformational_restraints_forces)
        self.max_orientational_restraint = max(self.orientational_restriant_forces)
        
        self.mdin_intermidate_config = os.path.abspath(self.mdin_intermidate_config)

        with open(self.mdin_intermidate_config) as mdin_args:
            self.mdin_intermidate_config = yaml.safe_load(mdin_args)

            
        if self.guest_restraint_template or self.receptor_restraint_template or self.complex_orientational_template:
            
            
            #self.tempdir = "mdgb/restraints"
            self.tempdir = tempfile.TemporaryDirectory()
            # if not os.path.exists('mdgb/restraints'):
            #     os.makedirs('mdgb/restraints')
            
            for con_force, orient_force in zip(self.conformational_restraints_forces, self.orientational_restriant_forces):
                self.write_ligand_restraint(conformational_force=con_force)
                self.write_receptor_restraints(conformational_force=con_force)
                self.write_complex_restraints(conformational_force=con_force, orientational_force=orient_force)
                
            
            
    @classmethod
    def from_config(cls: Type["IntermidateStatesArgs"], obj:dict):
        return cls(**obj)   

    def write_ligand_restraint(self, conformational_force):
        filename = re.sub(r"\..*", "", os.path.basename(self.guest_restraint_template))  # type: ignore
        
        with open(self.guest_restraint_template) as f:  # type: ignore
            ligand_restraints = f.readlines()

        string_template = ""
        for line in ligand_restraints:
            if 'frest' in line:
                line = line.replace('frest', str(conformational_force))
                
            string_template += line 
            
        with open(f"{self.tempdir.name}/{filename}_{conformational_force}.RST", "w") as output:
            output.write(string_template)

        
        self.guest_restraint_files.append(f"{self.tempdir.name}/{filename}_{conformational_force}.RST")      # type: ignore
    
    def write_receptor_restraints(self, conformational_force):
        filename = re.sub(r"\..*", "", os.path.basename(self.receptor_restraint_template))  # type: ignore
        
        with open(self.receptor_restraint_template) as f:  # type: ignore
            receptor_restraints = f.readlines()

        string_template = ""
        for line in receptor_restraints:
            if 'frest' in line:
                 line = line.replace('frest', str(conformational_force))
            string_template += line 
    
        
        with open(f"{self.tempdir.name}/{filename}_{conformational_force}.RST", "w") as output:
            output.write(string_template)
            
            
        self.receptor_restraint_files.append(f"{self.tempdir.name}/{filename}_{conformational_force}.RST")      # type: ignore
    
    def write_complex_restraints(self, conformational_force, orientational_force):
        
        filename = re.sub(r"\..*", "", os.path.basename(self.complex_orientational_template))  # type: ignore
        
        with open(self.complex_conformational_template) as f:  # type: ignore
            complex_conformational = f.readlines()
        
        with open(self.complex_orientational_template) as fH:  # type: ignore
            complex_orientational = fH.readlines()
        
        string_template = ""
        for line in complex_orientational:
            if 'drest' in line:
                line = line.replace('drest', str(conformational_force))
            if 'arest' in line:
                line = line.replace('arest', str(orientational_force))
            if 'trest' in line:
                line = line.replace('trest', str(orientational_force))
            if '&end' in line:
                line = line.replace('&end', "")
            
            string_template += line 
            
        for line in complex_conformational:
            if 'frest' in line:
                line = line.replace('frest', str(conformational_force))
            string_template += line 
            
        with open(f"{self.tempdir.name}/{filename}_{conformational_force}_{orientational_force}.RST", "w") as output:
            output.write(string_template) 
            
        
        self.complex_restraint_files.append(f"{self.tempdir.name}/{filename}_{conformational_force}_{orientational_force}.RST")   # type: ignore
       
    def toil_import_user_restriants(self, toil:Toil):
        """
        import restraint files into Toil job store 
        """
        for i, (guest_rest, receptor_rest, complex_rest) in enumerate(zip(self.guest_restraint_files, self.receptor_restraint_files, self.complex_restraint_files)):  # type: ignore
                    self.guest_restraint_files[i] = toil.import_file("file://" + os.path.abspath(guest_rest))  # type: ignore
                    self.receptor_restraint_files[i] = toil.import_file(("file://" + os.path.abspath(receptor_rest)))  # type: ignore
                    self.complex_restraint_files[i] = toil.import_file(("file://" + os.path.abspath(complex_rest)))  # type: ignore
        
        #self.tempdir.cleanup()
    
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
    endstate_method:  EndStateMethod
    intermidate_args: IntermidateStatesArgs
    inputs: dict 
    restraints: dict 
    ignore_receptor: bool = False 
    
    def __post_init__(self):
        self._config_sanitity_check()
        
        #if endstate_method_type =0 don't run any endstate calculations 
        if self.endstate_method.endstate_method_type == 0:
            self.workflow.run_endstate_method = False 
            
            if self.endstate_files.ligand_parameter_filename == None:
                raise ValueError(f"user specified to not run an endstate simulation but did not provided ligand_parameter_filename/coordinate endstate files")
        
       
    def _config_sanitity_check(self):
        #check if the amber mask are valid 
        
        traj = pt.iterload(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
        ligand_natoms = pt.strip(traj, self.amber_masks.receptor_mask).n_atoms 
        receptor_natoms = pt.strip(traj, self.amber_masks.ligand_mask).n_atoms 
        parm = pmd.amber.AmberFormat(self.endstate_files.complex_parameter_filename)
        #check if sum of ligand & receptor atoms = complex total num of atoms 
        if traj.n_atoms != ligand_natoms + receptor_natoms:
            raise RuntimeError(f'''The sum of ligand/guest and receptor/host atoms != number of total complex atoms
                                number of ligand atoms: {ligand_natoms} + number of receptor atoms {receptor_natoms} != complex total atoms: {traj.n_atoms}
                                Please check if AMBER masks are correct ligand_mask: "{self.amber_masks.ligand_mask}" receptor_mask: "{self.amber_masks.receptor_mask}"
                                {self.endstate_files.complex_parameter_filename} residue lables are: {parm.parm_data['RESIDUE_LABEL']}''')
        
    @classmethod 
    def from_config(cls: Type["Config"], user_config:dict):
        return cls(
            workflow=Workflow.from_config(user_config),
            system_settings=SystemSettings.from_config(user_config["system_parameters"]),
            endstate_files = ParameterFiles.from_config(user_config["endstate_parameter_files"]),
            num_cores_per_system = NumberOfCoresPerSystem.from_config(user_config["number_of_cores_per_system"]),
            amber_masks=AmberMasks.from_config(user_config["AMBER_masks"]),
            endstate_method=EndStateMethod.from_config(user_config["workflow"]),     
            intermidate_args = IntermidateStatesArgs.from_config(user_config["workflow"]["intermidate_states_arguments"]),
            inputs= {},
            restraints={}
        )  
    
    
            
        
        
    @property 
    def complex_pytraj_trajectory(self)->pt.Trajectory:
        traj = pt.iterload(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
        return traj 
    @property 
    def ligand_pytraj_trajectory(self)->pt.Trajectory:
        return self.complex_pytraj_trajectory[self.amber_masks.ligand_mask]
    @property 
    def receptor_pytraj_trajectory(self)->pt.Trajectory:
        return self.complex_pytraj_trajectory[self.amber_masks.receptor_mask]
    
    def get_receptor_ligand_topologies(self):
        '''
        Splits the complex into the ligand and receptor individual files. 
        
        These parameters files are inputs for ENDSTATE simulation 
        '''
        self.tempdir = tempfile.TemporaryDirectory()
       
        
        #don't use strip!!! use masks ligand[mask] instead!!!
        complex_traj = pt.iterload(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
        receptor_traj = complex_traj[self.amber_masks.receptor_mask]

        #get ligand trajectory coordinate from provided complex.parm7
        ligand_traj= complex_traj[self.amber_masks.ligand_mask]
        ligand_name = self.amber_masks.ligand_mask.strip(":")
        
        ligand_filename = os.path.join(self.tempdir.name, ligand_name)
        receptor_filename = os.path.join(self.tempdir.name, f"{self.amber_masks.receptor_mask.strip(':')}")
        
        print("lignad_filename", ligand_filename)
        pt.write_parm(f"{ligand_filename}.parm7", ligand_traj.top)
        pt.write_traj(f"{ligand_filename}.ncrst", ligand_traj)
        
        self.endstate_files.ligand_parameter_filename = os.path.abspath(f"{ligand_filename}.parm7")
        self.endstate_files.ligand_coordinate_filename = os.path.abspath(f"{ligand_filename}.ncrst.1")
        
        pt.write_parm(f"{receptor_filename}.parm7",receptor_traj.top)
        pt.write_traj(f"{receptor_filename}.ncrst",receptor_traj)
        self.endstate_files.receptor_parameter_filename = os.path.abspath(f"{receptor_filename}.parm7")
        self.endstate_files.receptor_coordinate_filename = os.path.abspath(f"{receptor_filename}.ncrst.1")

        self.endstate_files._create_unique_receptor_id()
       
def workflow(job, config:Config):
    
    tempdir = job.fileStore.getLocalTempDir()
    
    job.fileStore.readGlobalFile(
                    config.endstate_files.complex_coordinate_filename, 
                    userPath=os.path.join(tempdir, os.path.basename(config.endstate_files.complex_coordinate_filename)))
    return 1
if __name__ == "__main__":
  
    import yaml
  
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "OFF"
    options.clean = "always"
    with open("/nas0/ayoub/CB7_restraint_workflow/CB7_config_files/mdgb_01.yaml") as fH:
        yaml_config = yaml.safe_load(fH)

    with Toil(options) as toil:
       
        config = Config.from_config(yaml_config)    
       #print(config)
        #config.endstate_files.get_inital_coordinate()
        if config.endstate_method.endstate_method_type != 0:
            config.get_receptor_ligand_topologies()
        else:
            config.endstate_files.get_inital_coordinate()
            config.intermidate_args.toil_import_user_restriants(toil=toil)
    
        config.endstate_files.toil_import_parmeters(toil=toil)
       
        print(config)
        # config.endstate_method.remd_args.toil_import_replica_mdins(toil=toil)
        # boresch_p = list(config.boresch_parameters.__dict__.values())
        
        toil.start(Job.wrapJobFn(workflow, config))
        # print(config.intermidate_args)
