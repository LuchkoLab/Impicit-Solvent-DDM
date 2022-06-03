import os
from dataclasses import dataclass, field
from typing import List, Optional, Type

import pytraj as pt
from pydantic import NoneIsAllowedError


@dataclass 
class Workflow:
    run_endstate_method: bool = True
    add_ligand_conformational_restraints: bool = True
    remove_GB_solvent_ligand: bool = True
    remove_ligand_charges: bool = True
    add_receptor_lambda_windows: bool = True
    remove_GB_solvent_receptor: bool = True
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
    def update_worflow(cls:Type["Workflow"], system:str):
        
        if system == 'ligand':
            return cls(
                run_endstate_method = False, 
                remove_GB_solvent_ligand= False, 
            )
    
@dataclass
class SystemSettings:
    executable: str
    mpi_command: str
    working_directory: str = 'no set' 
    CUDA: bool = field(default=False)
    
    @classmethod
    def from_config(cls: Type["SystemSettings"], obj:dict):
        return cls(**obj)

@dataclass 
class ParameterFiles:
    complex_parameter_filename: str
    complex_coordinate_filename: str
    ligand_parameter_filename: str = 'ligand_parm'
    ligand_coordinate_filename:  str = 'ligand_coordinate' 
    receptor_parameter_filename: str = 'receptor_parm' 
    receptor_coordinate_filename: str = 'receptor_coordinate' 
   
       
    def __post_init__(self):
        self.complex_parameter_filename = os.path.abspath(self.complex_parameter_filename)
        self.complex_coordinate_filename = os.path.abspath(self.complex_coordinate_filename)

    @classmethod 
    def from_config(cls: type["ParameterFiles"], obj:dict):
       return cls(**obj)
   
        
@dataclass
class NumberOfCoresPerSystem:
    complex_ncores: int 
    ligand_ncores: int
    receptor_ncores: int 
    
    @classmethod
    def from_config(cls:type["NumberOfCoresPerSystem"], obj:dict):
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
    def from_config(cls:type["AmberMasks"], obj:dict):
        return cls(
            receptor_mask=obj["receptor_mask"],
            ligand_mask=obj["ligand_mask"]
        )

@dataclass
class REMD:
    ngroups: int = 0
    nthreads: int = 0
    target_temperature: float = 0.0 
    equilibration_replica_mdins: List[str] = field(default_factory=list)
    remd_mdins: List[str]  = field(default_factory=list)
    
    
    @classmethod
    def from_config(cls: Type["REMD"], obj:dict):
        return cls(
            nthreads = obj["endstate_arguments"]["nthreads"],
            ngroups = obj["endstate_arguments"]["ngroups"],
            target_temperature = obj["endstate_arguments"]["target_temperature"],
            equilibration_replica_mdins = obj["endstate_arguments"]["equilibration_replica_mdins"],
            remd_mdins = obj["endstate_arguments"]["remd_mdins"]
            )

@dataclass
class EndStateMethod: 
    endstate_method_type: str 
    flat_bottom_restraints: List[float]
    remd_args: REMD
     
    def __post_init__(self):
        endstate_method_options = ["remd", "md", 0]
        if self.endstate_method_type not in endstate_method_options:
            raise NameError(f"'{self.endstate_method_type}' is not a valid endstate method. Options: {endstate_method_options}")
    
    @classmethod
    def from_config(cls:type["EndStateMethod"], obj:dict):
        if obj["endstate_method"] == 'REMD':
            return cls(
                endstate_method_type=str(obj["endstate_method"]).lower(),
                remd_args=REMD.from_config(obj=obj),
                flat_bottom_restraints=obj["endstate_arguments"]["flat_bottom_restraints"]
                )
        else:
            return cls(
                endstate_method_type=obj["endstate_method"],
                flat_bottom_restraints=obj["endstate_arguments"]["flat_bottom_restraints"],
                remd_args = REMD()
            )
@dataclass
class IntermidateStatesArgs: 
    conformational_restraints_forces: List[float]
    orientational_restriant_forces: List[float]
    restraint_type: int 
    igb_solvent: int 
    max_conformational_restraint: float = field(init=False)
    max_orientational_restraint: float = field(init=False)
    mdin_intermidate_config: str 
    
    def __post_init__(self):

        self.max_conformational_restraint = max(self.conformational_restraints_forces)
        self.max_orientational_restraint = max(self.orientational_restriant_forces)
        self.mdin_intermidate_config = os.path.abspath(self.mdin_intermidate_config)
    
    @classmethod
    def from_config(cls: type["IntermidateStatesArgs"], obj:dict):
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
    ignore_receptor: bool = False 
    
    def __post_init__(self):
        self._config_sanitity_check()

    def _config_sanitity_check(self):
        pass
    
    @classmethod 
    def from_config(cls:type["Config"], user_config:dict):
        return cls(
            workflow=Workflow.from_config(user_config),
            system_settings=SystemSettings.from_config(user_config["system_parameters"]),
            endstate_files = ParameterFiles.from_config(user_config["endstate_parameter_files"]),
            num_cores_per_system = NumberOfCoresPerSystem.from_config(user_config["number_of_cores_per_system"]),
            amber_masks=AmberMasks.from_config(user_config["AMBER_masks"]),
            endstate_method=EndStateMethod.from_config(user_config["workflow"]),     
            intermidate_args = IntermidateStatesArgs.from_config(user_config["workflow"]["intermidate_states_arguments"]),
            inputs= {}
        )  

    def get_receptor_ligand_topologies(self):
        '''
        Splits the complex into the ligand and receptor individual files.
        '''
        receptor_ligand_path = []
        #/mdgb/structs/receptor /ligand
        receptor_ligand_path.append(os.path.join(self.system_settings.working_directory, "mdgb/structs/ligand"))
        receptor_ligand_path.append(os.path.join(self.system_settings.working_directory,"mdgb/structs/receptor"))
        
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
        
        
        if os.path.exists(f"{receptor_name}_{0:03}.parm7"):
            if not self.ignore_receptor:
                self.ignore_receptor = True
            self.endstate_files.receptor_parameter_filename = f"{receptor_name}_{0:03}.parm7"
        else:
            pt.write_parm(f"{receptor_name}_{0:03}.parm7",receptor.top)
            pt.write_traj(f"{receptor_name}_{0:03}.ncrst",receptor)
            self.endstate_files.receptor_parameter_filename = os.path.abspath(f"{receptor_name}_{0:03}.parm7")
            self.endstate_files.receptor_coordinate_filename = os.path.abspath(f"{receptor_name}_{0:03}.ncrst.1")
        
if __name__ == "__main__":
    import yaml
    with open('new_workflow.yaml') as yaml_file:
        config = yaml.safe_load(yaml_file)
    config_object = Config.from_config(config)
    config_object.ignore_receptor = True
    # print(config_object.endstate_files)
    # print("*" * 20)
    config_object.get_receptor_ligand_topologies()
    print(config_object.endstate_method.endstate_method_type)

   
    
  