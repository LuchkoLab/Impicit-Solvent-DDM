

'''
class that will parse in pandas dataframe for mbar analysis 
'''
import os
from cProfile import run
from itertools import chain
from typing import List

import pandas as pd

from implicit_solvent_ddm.get_dirstruct import Dirstruct
from implicit_solvent_ddm.mdout import min_to_dataframe
from implicit_solvent_ddm.simulations import Simulation

working_directory = os.getcwd()


    
class PostTreatment:
    AVAGADRO  = 6.0221367e23
    BOLTZMAN = 1.380658e-23
    TEMP = 298
    JOULES_PER_KCAL= 4184
    kt_conversion = 1/(((BOLTZMAN*(AVAGADRO))/JOULES_PER_KCAL)*TEMP)
    
    def __init__(self, data_list: list[list[pd.DataFrame]], restraint_filename: str) -> None:
        
        self.restraint_filename = restraint_filename
        self.data_list = data_list
        self._load_restraints()
        self._load_dfs()
        self._create_MBAR_format()
    
    def _load_restraints(self):
        pass 
       
    def _load_dfs(self):
        flatten_dfs = list(chain(*self.data_list))
        self.df =  pd.concat(flatten_dfs, axis=0, ignore_index=True)
    
    
    def _create_MBAR_format(self):
        
        
        self.df = self.df.set_index(["solute", "traj_con_rest", "parm_con_restraint", "Frames", "parm_state", "traj_state"], drop=True)
        
        self.df = self.df[["ENERGY"]]
        self.df = self.df.unstack(["parm_state","parm_con_restraint"])  # type: ignore
        self.df = self.df.reset_index(["Frames","traj_state","solute"],drop=True)
        states = [_ for _ in zip(*self.df.columns)][1]
        restraints = [_ for _ in zip(*self.df.columns)][2]
        column_names = []
        
        
        self.df.columns = column_names
        
    def calculate_mbar(self):
        pass 
         
   
    
def create_mdout_dataframe(job, calculated_simulation: Simulation) -> pd.DataFrame:

    sim = Dirstruct("mdgb", calculated_simulation.directory_args, dirstruct=calculated_simulation.dirstruct)

    output_dir= os.path.join(working_directory,  sim.dirStruct.fromArgs(**sim.parameters))

    mdout = f"{output_dir}/mdout"
    
    run_args = sim.dirStruct.fromPath2Dict(mdout)
    data = min_to_dataframe(mdout)
    
    #data["traj_state_label"] = run_args["traj_state_label"]
    #data["state_label"] = run_args["state_label"]
    
    data["solute"] = run_args["topology"]
    data["parm_state"] = run_args["state_label"]
    data["traj_state"] = run_args["traj_state_label"]
    data['Frames'] = data.index
    
    data["parm_restraints"] = run_args["conformational_restraint"]
    data["traj_restraints"] = run_args["trajectory_restraint_conrest"]
    #complex datastructure 
    if "trajectory_restraint_orenrest" in run_args.keys():
            data['parm_restraints'] = f"{run_args['conformational_restraint']}_{run_args['orientational_restraints']}"
            data['traj_restraints'] = f"{run_args['trajectory_restraint_conrest']}_{run_args['trajectory_restraint_orenrest']}" 
            
            # data["traj_restraint"] = f"{run_args['trajectory_restraint_conrest']}_{run_args['trajectory_restraint_orenrest']}" 
            # data["parm_restraint"] = f"{run_args['conformational_restraint']}_{run_args['orientational_restraints']}"
    
    # else:
    #     data['parm_restraints'] = run_args[""]
    #     parm_state = run_args["conformational_restraint"]
    #     traj_state = run_args["trajectory_restraint_conrest"]
            
    # #lambda state windows   
    # job.log(f"THE STATE LABEL {run_args['state_label']}")
    # if run_args["state_label"] == "lambda_window":
    #     data["parm_state"] = parm_state
    
    # if run_args["traj_state_label"] == "lambda_window":
    #     data["traj_state"] = traj_state
   
  
    data.to_hdf(f"{calculated_simulation.output_dir}/simulation_mdout.h5", key="df")
    
    return data
    


