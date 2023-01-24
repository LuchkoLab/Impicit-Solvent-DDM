
import itertools
import logging
import os
from email import message
from importlib.metadata import files
from pathlib import Path
from sre_constants import ANY
from typing import Any, Dict, List, Optional, TypedDict, Union

import pandas as pd
import yaml
from matplotlib.backend_bases import key_press_handler
from toil.batchSystems import abstractBatchSystem
from toil.common import Toil
from toil.job import FileID, Job, JobFunctionWrappingJob, PromisedRequirement

from implicit_solvent_ddm.postTreatment import create_mdout_dataframe
from implicit_solvent_ddm.restraints import RestraintMaker
from implicit_solvent_ddm.simulations import Simulation


class IntermidateRunner(Job):
    #simulations: dict[Simulation, int]
    def __init__(self, 
                 simulations: list[Simulation], restraints:RestraintMaker, 
                 post_process_no_solv_mdin: FileID, post_process_mdin: FileID,
                 post_process_distruct: str, memory: Optional[Union[int, str]] = None, cores: Optional[Union[int, float, str]] = None, disk: Optional[Union[int, str]] = None, preemptable: Optional[Union[bool, int, str]] = None, unitName: Optional[str] = "", checkpoint: Optional[bool] = False, displayName: Optional[str] = "", descriptionClass: Optional[str] = None) -> None:
       
        super().__init__(memory, cores, disk, preemptable, unitName, checkpoint, displayName, descriptionClass)
      
        self.simulations = simulations
        self.restraints = restraints
        self.no_solvent_mdin = post_process_no_solv_mdin
        self.mdin = post_process_mdin
        self.post_output = []
        self.ligand_output = []
        self.receptor_output = []
        self.complex_output = [] 
        self.post_process_distruct = post_process_distruct
  
    
    def run(self, fileStore):
        
        def run_post_process(job:Job, ran_simulation:Simulation):
            
            for post_simulation in self.simulations:
                
                #check if the system are the same. ligand with ligand | receptor with receptor ect 
                if ran_simulation.system_type == post_simulation.system_type:
                    
                    directory_args = post_simulation.directory_args.copy()
        
                    directory_args.update(self.update_postprocess_dirstruct(ran_simulation.directory_args, job.rv(1)))  # type: ignore
                    fileStore.logToMaster(f"RUNNER directory args {directory_args}\n")
                    
                    mdin = self.mdin
                    if post_simulation.directory_args["igb_value"] == 6:
                       mdin = self.no_solvent_mdin
                        
                    #run simulation if its not endstate with endstate 
                    if post_simulation.inptraj != ran_simulation.inptraj or post_simulation.inptraj == None: 
                        post_dirstruct = self.get_system_dirs(post_simulation.system_type)
                        fileStore.logToMaster(f"get_system_dirs {post_simulation.system_type}")
                        fileStore.logToMaster(f"current dirstruct {post_dirstruct}")
                        fileStore.logToMaster(f"inptraj ERROR {job.rv(1)}")
                        post_process_job = Simulation(executable=post_simulation.executable, 
                                                    mpi_command=post_simulation.mpi_command, 
                                                    num_cores=post_simulation.num_cores, 
                                                    prmtop=post_simulation.prmtop, incrd=post_simulation.incrd,
                                                    input_file=mdin, restraint_file=post_simulation.restraint_file, 
                                                    working_directory=post_simulation.working_directory,
                                                    directory_args=directory_args, dirstruct=post_dirstruct, 
                                                    inptraj=job.rv(1), restraint_key=post_simulation.restraint_key)
                        job.addChild(post_process_job)
                        data_frame = post_process_job.addFollowOnJobFn(create_mdout_dataframe, post_process_job.directory_args, post_process_job.dirstruct, post_process_job.output_dir)
                    
                    else:
                        data_frame = job.addFollowOnJobFn(create_mdout_dataframe, directory_args, post_simulation.dirstruct, post_simulation.output_dir)
                
                    self.post_output.append(data_frame.rv())
                    #self.system_specific_add(post_simulation.system_type, data_frame=data_frame.rv())
        
        # iterate and submit all intermidate simulations. Then followup with post-process        
        for simulation in self.simulations:
            run_post_process(job=self.addChild(simulation), ran_simulation = simulation)
            
        return self.post_output
    
    @staticmethod
    def update_postprocess_dirstruct(run_time_args:dict, inptaj)->dict[str, Union[str, object]]:
        
        if "orientational_restraints" in run_time_args.keys():
            return {
                "traj_state_label": run_time_args["state_label"],
                "trajectory_restraint_conrest": run_time_args["conformational_restraint"],
                "trajectory_restraint_orenrest": run_time_args["orientational_restraints"],
                "traj_igb": run_time_args["igb"],
                "traj_charge": run_time_args["charge"], 
                "filename": f"{run_time_args['filename']}_postprocess",
                "runtype": f"Running post process with trajectory: {inptaj}",       
            }
        else:
            return {
                "traj_state_label": run_time_args["state_label"],
                "trajectory_restraint_conrest": run_time_args["conformational_restraint"],
                "traj_igb": run_time_args["igb"],
                "traj_charge": run_time_args["charge"],
                "filename": f"{run_time_args['filename']}_postprocess",
                "runtype": f"Running post process with trajectory: {inptaj}",
                
            }
    @staticmethod
    def get_system_dirs(system_type):
        if system_type == "ligand" or system_type == "receptor":
            return "post_process_apo"
        
        return "post_process_halo"
    
    
    def system_specific_add(self, system_type, data_frame):
        
        if system_type == 'ligand': 
            self.ligand_output.append(data_frame)
                    
        elif system_type == 'receptor':
            self.receptor_output.append(data_frame)
        
        else: 
            self.complex_output.append(data_frame)

