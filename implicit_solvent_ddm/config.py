import os
from copy import deepcopy
from dataclasses import dataclass, field
from optparse import Option
from tempfile import TemporaryFile
from typing import List, Optional, Type, Union

import numpy as np
import parmed as pmd
import pytraj as pt
import yaml


@dataclass 
class Workflow:
    setup_workflow: bool = True
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
                end_state_postprocess = run_endstate,
                run_endstate_method = False, 
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
                end_state_postprocess = run_endstate,
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
                end_state_postprocess = run_endstate,
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
    memory: Optional[Union[int, str]] = field(default="10G")
    disk: Optional[Union[int, str]] = field(default="10G")
    @classmethod
    def from_config(cls: Type["SystemSettings"], obj:dict):
        return cls(**obj)

@dataclass 
class ParameterFiles:
    complex_parameter_filename: str
    complex_coordinate_filename: str
    ligand_parameter_filename: Optional[str] = field(default=None)
    ligand_coordinate_filename:  Optional[str]  = field(default=None)
    receptor_parameter_filename: Optional[str] = field(default=None)
    receptor_coordinate_filename: Optional[str] = field(default=None)
    
    
    def __post_init__(self):
        self.complex_parameter_filename = os.path.abspath(self.complex_parameter_filename)
        self.complex_coordinate_filename = os.path.abspath(self.complex_coordinate_filename)
        
        #check complex is a valid structure 
        #ASK LUCHKO HOW TO CHECK FOR VALID STRUCTURES
        complex_traj = pt.load(self.complex_coordinate_filename, self.complex_parameter_filename)
        pt.check_structure(traj=complex_traj)
        
        

    @classmethod 
    def from_config(cls: Type["ParameterFiles"], obj:dict):
       return cls(**obj)
    
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
    equilibration_replica_mdins: List[str] = field(default_factory=list)
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
                remd_args = REMD()
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
        
    @classmethod
    def from_config(cls: Type["IntermidateStatesArgs"], obj:dict):
        return cls(**obj)   


    
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
    output: dict 
    ignore_receptor: bool = False 
    
    def __post_init__(self):
        self._config_sanitity_check()
        
        #if endstate_method_type =0 don't run any endstate calculations 
        if self.endstate_method.endstate_method_type == 0:
            self.workflow.run_endstate_method = False 
            
            if self.endstate_files.ligand_parameter_filename == None:
                raise ValueError(f"user specified no endstate simulation but did not provided ligand_parameter_filename/coordinate file")
           
    def _config_sanitity_check(self):
        #check if the amber mask are valid 
        
        traj = pt.load(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
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
            output={}
        )  
    @property 
    def complex_pytraj_trajectory(self)->pt.Trajectory:
        traj = pt.load(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
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
        '''
        receptor_ligand_path = []
        #/mdgb/structs/receptor /ligand
        receptor_ligand_path.append(os.path.join(self.system_settings.working_directory, "mdgb/structs/ligand"))
        receptor_ligand_path.append(os.path.join(self.system_settings.working_directory,"mdgb/structs/receptor"))
        
        #don't use strip!!! use masks ligand[mask] instead!!!
        complex_traj = pt.load(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
        receptor = pt.strip(complex_traj, self.amber_masks.ligand_mask)
        ligand = pt.strip(complex_traj, self.amber_masks.receptor_mask)
        
        ligand_name = os.path.join(receptor_ligand_path[0], self.amber_masks.ligand_mask.strip(":"))
        receptor_name = os.path.join(receptor_ligand_path[1], self.amber_masks.receptor_mask.strip(":"))
        
        file_number = 0
        while os.path.exists(f"{ligand_name}_{file_number:03}.parm7"):
            file_number +=1    
        pt.write_parm(f"{ligand_name}_{file_number:03}.parm7", ligand.top, overwrite=True)
        pt.write_traj(f"{ligand_name}_{file_number:03}.ncrst", ligand, overwrite=True)
        
        self.endstate_files.ligand_parameter_filename = os.path.abspath(f"{ligand_name}_{file_number:03}.parm7")
        self.endstate_files.ligand_coordinate_filename = os.path.abspath(f"{ligand_name}_{file_number:03}.ncrst.1")
        
        file_number = 0
        while os.path.exists(f"{receptor_name}_{file_number:03}.parm7"):
            file_number +=1
        # if os.path.exists(f"{receptor_name}_{0:03}.parm7"):
        if  self.ignore_receptor:
            self.endstate_files.receptor_parameter_filename = f"{receptor_name}_{file_number:03}.parm7"
        else:
            pt.write_parm(f"{receptor_name}_{file_number:03}.parm7",receptor.top)
            pt.write_traj(f"{receptor_name}_{file_number:03}.ncrst",receptor)
            self.endstate_files.receptor_parameter_filename = os.path.abspath(f"{receptor_name}_{file_number:03}.parm7")
            self.endstate_files.receptor_coordinate_filename = os.path.abspath(f"{receptor_name}_{file_number:03}.ncrst.1")

def workflow(job, config:Config):
    
    job.log(f"config.ligand_pytraj_trajectory : {config.ligand_pytraj_trajectory}")

if __name__ == "__main__":
    import yaml
    with open("/nas0/ayoub/Impicit-Solvent-DDM/new_workflow.yaml") as fH:
        config = yaml.safe_load(fH)
    config_object = Config.from_config(config)
    print(config_object.system_settings.memory)
    print(config_object.system_settings.disk)
   
    print(config_object.endstate_method.remd_args)
    print(config_object.intermidate_args.mdin_intermidate_config)
    # print(new_workflow)
    # print(config_object.workflow)
    # import yaml
    # options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    # options.logLevel = "INFO"
    # options.clean = "always"
    
    # with open('new_workflow.yaml') as yaml_file:
    #     config = yaml.safe_load(yaml_file)
    # config_object = Config.from_config(config)
    # config_object.ignore_receptor = True
    # print(config_object.ligand_pytraj_trajectory)
    # with Toil(options) as toil:

    #     print(toil.start(Job.wrapJobFn(workflow,  config_object)))   
    # # print(config_object.endstate_files)
    # # print("*" * 20)
    # #config_object.get_receptor_ligand_topologies()
    
   
    
  