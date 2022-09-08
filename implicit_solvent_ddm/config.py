import os
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Complex
from optparse import Option
from tempfile import TemporaryFile
from typing import List, Optional, Type, Union

import numpy as np
import parmed as pmd
import pytraj as pt
import yaml
from toil.common import FileID, Toil
from toil.job import Job


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
    complex_parameter_filename: Union[str, FileID]
    complex_coordinate_filename: Union[str, FileID]
    ligand_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    ligand_coordinate_filename:  Optional[Union[str, FileID]]  = field(default=None)
    receptor_parameter_filename: Optional[Union[str, FileID]] = field(default=None)
    receptor_coordinate_filename: Optional[Union[str, FileID]] = field(default=None)
    
    def __post_init__(self):
       
        #check complex is a valid structure 
        #ASK LUCHKO HOW TO CHECK FOR VALID STRUCTURES
        complex_traj = pt.iterload(self.complex_coordinate_filename, self.complex_parameter_filename)
        pt.check_structure(traj=complex_traj)
        

    @classmethod 
    def from_config(cls: Type["ParameterFiles"], obj:dict):
       return cls(**obj)
   
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
    
    guest_restraint_files: Optional[list[Union[str, FileID]]] = field(default=None)  
    receptor_restraint_files: Optional[list[Union[str, FileID]]] = field(default=None)  
    complex_restraint_files: Optional[list[Union[str, FileID]]] = field(default=None)
      
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

        if self.guest_restraint_files != None or self.receptor_restraint_files != None or self.complex_restraint_files != None:
            
            if all(isinstance(i, list) for i in [self.guest_restraint_files, self.receptor_restraint_files, self.complex_restraint_files]):
                num_guest = len(self.guest_restraint_files) # type: ignore
                num_receptor = len(self.receptor_restraint_files) # type: ignore
                num_complex = len(self.complex_restraint_files)  # type: ignore
                if num_guest != num_receptor or num_guest != num_complex:   
                    raise RuntimeError(f'''The number of restraint files do not equal each other
                                    guest_restraint_files: {num_guest}
                                    receptor_restraint_files: {num_receptor}
                                    complex_restraint_files: {num_complex}''')
            else:
                raise TypeError("guest, receptor or complex did not specify restraint files")
    
    @classmethod
    def from_config(cls: Type["IntermidateStatesArgs"], obj:dict):
        return cls(**obj)   

    def toil_import_user_restriants(self, toil:Toil):
        """
        import restraint files into Toil job store 
        """
        for i, (guest_rest, receptor_rest, complex_rest) in enumerate(zip(self.guest_restraint_files, self.receptor_restraint_files, self.complex_restraint_files)):  # type: ignore
                    self.guest_restraint_files[i] = toil.import_file("file://" + os.path.abspath(guest_rest))  # type: ignore
                    self.receptor_restraint_files[i] = toil.import_file(("file://" + os.path.abspath(receptor_rest)))  # type: ignore
                    self.complex_restraint_files[i] = toil.import_file(("file://" + os.path.abspath(complex_rest)))  # type: ignore
@dataclass 
class BoreschParameters:
    dist_restraint_r: float = None 
    angle2_rest_val: float  = None 
    dist_rest_Kr: float  = None 
    max_conformational_force: float = None 
    max_orientational_force: float  = None 
    angle1_rest_Ktheta1:float = field(init=False)
    angle2_rest_Ktheta2:float = field(init=False)
    torsion1_rest_Kphi1:float = field(init=False)
    torsion2_rest_Kphi2:float = field(init=False)
    torsion3_rest_Kphi3:float = field(init=False)
    
    def __post_init__(self):
        
        self.angle1_rest_Ktheta1 = self.angle2_rest_Ktheta2 = self.max_conformational_force        
        self.torsion1_rest_Kphi1 = self.torsion2_rest_Kphi2 = self.torsion3_rest_Kphi3 = self.max_orientational_force
    
    
    @classmethod
    def from_config(cls: Type["BoreschParameters"], obj:dict):
        if "boresch_parametes" in obj.keys():
            return cls(**obj["boresch_parametes"])
        else:
            return cls()
    
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
    boresch_parameters: BoreschParameters
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
        
      
        if self.intermidate_args.guest_restraint_files is not None:
            boresch_parameters = list(self.boresch_parameters.__dict__.values())
        
            boresch_p_type = all(i==None for i in boresch_parameters)
            if boresch_p_type:
                raise RuntimeError(f''' User restraints did not specify boresch parameters
                                   If you providing your own restraints please 
                                   specifiy all necessary boresch parameters within the config file
                                   to compute analytical dG''')
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
            boresch_parameters = BoreschParameters.from_config(user_config),
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
        '''
        receptor_ligand_path = []
        #/mdgb/structs/receptor /ligand
        receptor_ligand_path.append(os.path.join(self.system_settings.working_directory, "mdgb/structs/ligand"))
        receptor_ligand_path.append(os.path.join(self.system_settings.working_directory,"mdgb/structs/receptor"))
        
        #don't use strip!!! use masks ligand[mask] instead!!!
        complex_traj = pt.iterload(self.endstate_files.complex_coordinate_filename, self.endstate_files.complex_parameter_filename)
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
    
    job.log(f"config.ligand_pytraj_trajectory : {config}")

if __name__ == "__main__":
  
    import yaml
  
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "OFF"
    options.clean = "always"
    with open("new_workflow.yaml") as fH:
        yaml_config = yaml.safe_load(fH)

    with Toil(options) as toil:
       
        config = Config.from_config(yaml_config)    
        example = toil.import_file("file://" + os.path.abspath("implicit_solvent_ddm/tests/structs/cb7-mol01.parm7"))
        print(example)
        print(config.endstate_method.remd_args)
        
        config.endstate_method.remd_args.toil_import_replica_mdins(toil=toil)
        print(config.endstate_method.remd_args)
        boresch_p = list(config.boresch_parameters.__dict__.values())
        print(boresch_p)
        print(all(i==None for i in boresch_p))
        #toil.start(Job.wrapJobFn(workflow, config))
  