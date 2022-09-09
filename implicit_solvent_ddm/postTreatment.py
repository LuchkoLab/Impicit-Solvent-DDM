

'''
class that will parse in pandas dataframe for mbar analysis 
'''
import os
import re
from ast import parse
from cProfile import run
from itertools import chain
from numbers import Complex
from re import L
from typing import List

import pandas as pd
from toil.job import Job

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm.get_dirstruct import Dirstruct
from implicit_solvent_ddm.mdout import min_to_dataframe
from implicit_solvent_ddm.simulations import Simulation

WORKDIR = os.getcwd()

AVAGADRO  = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL= 4184


class PostTreatment(Job):
    
    def __init__(self, simulation_data: List[List[pd.DataFrame]], temp:float, system:str, max_conformation_force:float, max_orientational_force = None) -> None:
        super().__init__()
        self.simulations_data = simulation_data
        self.temp = temp
        self.system = system 
        self.max_con_force = str(max_conformation_force)
        self.max_orien_force = str(max_orientational_force)
        self._kt_conversion()
    
    def _kt_conversion(self):
        self.kt_conversion = 1/(((BOLTZMAN*(AVAGADRO))/JOULES_PER_KCAL)*self.temp)
    
    
    def _load_dfs(self):
        
        
        flatten_dfs = list(chain(*self.simulations_data))
        self.df =  pd.concat(flatten_dfs, axis=0, ignore_index=True)
        
        self.name = self.df["solute"].iloc[0]
    
    def _create_MBAR_format(self):
        
        self.df = self.df.set_index(["solute", "parm_state", "parm_restraints", "traj_state", "traj_restraints", "Frames"], drop=True)     
        self.df = self.df[["ENERGY"]]
        self.df = self.df.unstack(["parm_state","parm_restraints"])  # type: ignore
        self.df = self.df.reset_index(["Frames","solute"],drop=True)
        states = [_ for _ in zip(*self.df.columns)][1]
        restraints = [_ for _ in zip(*self.df.columns)][2]
        column_names = [(state, restraint) for state, restraint in zip(states, restraints)]
        
        self.df.columns = column_names  # type: ignore
    
    def compute_binding_deltaG(self, system1: float, system2: float, borech_dG=0.0):
                
        return self.deltaG + system1 + system2 + borech_dG
      
    def run(self, fileStore):
        
        self._load_dfs()
        self._create_MBAR_format()
        fileStore.logToMaster(f"self.df {self.df}")
        equil_info = pdmbar.detectEquilibration(self.df)
        
        df_subsampled = pdmbar.subsampleCorrelatedData(self.df, equil_info=equil_info)
        
        fe, error, mbar =  (pdmbar.mbar(df_subsampled))
        
        fe = fe/self.kt_conversion
        
        error = error/self.kt_conversion
        
        if self.system == 'ligand':
            self.deltaG = fe.loc[('endstate',  '0.0'), [('no_charges', self.max_con_force)]].values[0]  # type: ignore
        
        elif self.system == 'receptor':
            self.deltaG = fe.loc[('endstate',  '0.0'), [('no_gb', self.max_con_force)]].values[0] # type: ignore
        
        #system is complex 
        else: 
            self.deltaG = fe.loc[('no_interactions', f"{self.max_con_force}_{self.max_orien_force}"), [('endstate', '0.0_0.0')]].values[0] # type: ignore
        
        self.fe = fe 
        self.error = error

        return self 

def consolidate_output(job, ligand_system: PostTreatment, receptor_system: PostTreatment, complex_system: PostTreatment, boresch_df:pd.DataFrame):
    
    output_path = os.path.join(f"{WORKDIR}",".cache")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #parse out free energies 
    complex_system.fe.to_hdf(f"{output_path}/{complex_system.name}_fe.h5", key="df", mode='w')
    receptor_system.fe.to_hdf(f"{output_path}/receptor_{complex_system.name}_fe.h5", key="df", mode='w')
    ligand_system.fe.to_hdf(f"{output_path}/ligand_{complex_system.name}_fe.h5", key="df", mode='w')
    
    #parse out errors of mbar
    complex_system.error.to_hdf(f"{output_path}/{complex_system.name}_error.h5", key="df", mode='w')
    receptor_system.error.to_hdf(f"{output_path}/receptor_{complex_system.name}_error.h5", key="df", mode='w')
    ligand_system.error.to_hdf(f"{output_path}/ligand_{complex_system.name}_error.h5", key="df", mode='w')
    
    
    borech_dG = boresch_df["DeltaG"].values[0] 
    #compute total deltaG 
    deltaG_tot = complex_system.compute_binding_deltaG(system1=ligand_system.deltaG, system2=receptor_system.deltaG)

    deltaG_df = pd.DataFrame()
    
    deltaG_df[f"{ligand_system.name}_endstate->no_charges"] = [ligand_system.deltaG]
    deltaG_df[f"{receptor_system.name}_endstate->no_gb"] = [receptor_system.deltaG]
    deltaG_df["boresch_restraints"] = [borech_dG]
    deltaG_df[f"{complex_system.name}_no-interactions->endstate"] = [complex_system.deltaG]
    deltaG_df["deltaG"] = [deltaG_tot]
    
    deltaG_df.to_hdf(f"{output_path}/deltaG_{complex_system.name}.h5", key="df", mode='w')

              
def create_mdout_dataframe(job, directory_args: dict, dirstruct: str, output_dir: str) -> pd.DataFrame:

    sim = Dirstruct("mdgb", directory_args, dirstruct=dirstruct)

    output_dir= os.path.join(WORKDIR,  sim.dirStruct.fromArgs(**sim.parameters))

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
   
   
  
    data.to_hdf(f"{output_dir}/simulation_mdout.h5", key="df")
    
    return data
    











class PostProcess:
    def __init__(self, df: pd.DataFrame, system: str, temp:float, max_conformation_force:float, max_orientational_force = None) -> None:
        self.df = df
        self.temp = temp
        self.system = system 
        self.max_con_force = str(max_conformation_force)
        self.max_orien_force = str(max_orientational_force)
        self.fe = None
        self.error = None 
        self._kt_conversion()
        self._create_MBAR_format()
        
    def _kt_conversion(self):
        self.kt_conversion = 1/(((BOLTZMAN*(AVAGADRO))/JOULES_PER_KCAL)*self.temp)
    
    def _create_MBAR_format(self):
        
        
        self.df = self.df.set_index(["solute", "parm_state", "parm_restraints", "traj_state", "traj_restraints", "Frames"], drop=True)     
        self.df = self.df[["ENERGY"]]
        self.df = self.df.unstack(["parm_state","parm_restraints"])  # type: ignore
        self.df = self.df.reset_index(["Frames","solute"],drop=True)
        states = [_ for _ in zip(*self.df.columns)][1]
        restraints = [_ for _ in zip(*self.df.columns)][2]
        column_names = [(state, restraint) for state, restraint in zip(states, restraints)]
        
        self.df.columns = column_names  # type: ignore
    
        
    def run(self):
        
        equil_info = pdmbar.detectEquilibration(self.df)
        
        df_subsampled = pdmbar.subsampleCorrelatedData(self.df, equil_info=equil_info)
        
        fe, error, mbar =  (pdmbar.mbar(df_subsampled))
        
        fe = fe/self.kt_conversion
        
        error = error/self.kt_conversion
        
        if self.system == 'ligand':
            self.deltaG = fe.loc[('endstate',  '0.0'), [('no_charges', self.max_con_force)]].values[0]  # type: ignore
        
        elif self.system == 'receptor':
            self.deltaG = fe.loc[('endstate',  '0.0'), [('no_gb', self.max_con_force)]].values[0] # type: ignore
        
        #system is complex 
        else: 
            self.deltaG = fe.loc[('no_interactions', f"{self.max_con_force}_{self.max_orien_force}"), [('endstate', '0.0_0.0')]].values[0] # type: ignore
        
        self.fe = fe 
        self.error = error         
    
    def compute_binding_deltaG(self, system1: float, system2: float, borech_dG=0.0):
                
        return self.deltaG + system1 + system2 + borech_dG
def main(complex_file, receptor_file, ligand_file, boresch_file):

    import re

    complex_df = pd.read_hdf(complex_file)
    receptor_df = pd.read_hdf(receptor_file)
    ligand_df = pd.read_hdf(ligand_file) 
    boresch_df = pd.read_hdf(boresch_file)
    complex_obj = PostProcess(complex_df, system="complex", temp=298, max_conformation_force=4.0, max_orientational_force=8.0)
    receptor_obj = PostProcess(receptor_df, system="receptor", temp=298, max_conformation_force=4.0)
    ligand_obj = PostProcess(ligand_df, system="ligand", temp=298, max_conformation_force=4.0)
    complex_obj.run()
    receptor_obj.run()
    ligand_obj.run()
    
    complex_name = re.sub(r"\..*", "", os.path.basename(complex_file))
    ligand_name = re.sub(r"\..*", "", os.path.basename(ligand_file))
    receptor_name = re.sub(r"\..*", "", os.path.basename(receptor_file))
    
    #parse out mbar_formate dataframe 
    complex_obj.df.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/complex/{complex_name}_mbar_format.h5", key="df", mode='w')
    receptor_obj.df.to_hdf(f"output_path/receptor_{complex_name}_mbar_format.h5", key="df", mode='w')
    ligand_obj.df.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/ligand/ligand_{complex_name}_mbar_format.h5", key="df", mode='w')
    
    #parse out free energies 
    complex_obj.fe.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/complex/{complex_name}_fe.h5", key="df", mode='w')
    receptor_obj.fe.to_hdf(f"output_path/receptor_{complex_name}_fe.h5", key="df", mode='w')
    ligand_obj.fe.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/ligand/ligand_{complex_name}_fe.h5", key="df", mode='w')
    
    #parse out errors of mbar
    complex_obj.error.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/complex/{complex_name}_error.h5", key="df", mode='w')
    receptor_obj.error.to_hdf(f"output_path/receptor_{complex_name}_error.h5", key="df", mode='w')
    ligand_obj.error.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/ligand/ligand_{complex_name}_error.h5", key="df", mode='w')
    
    
    borech_dG = boresch_df["DeltaG"].values[0] 
    #compute total deltaG 
    deltaG_tot = complex_obj.compute_binding_deltaG(system1=ligand_obj.deltaG, system2=receptor_obj.deltaG)

    deltaG_df = pd.DataFrame()
    
    deltaG_df[f"{ligand_name}_endstate->no_charges"] = [ligand_obj.deltaG]
    deltaG_df[f"{receptor_name}_endstate->no_gb"] = [receptor_obj.deltaG]
    deltaG_df["boresch_restraints"] = [borech_dG]
    deltaG_df[f"{complex_name}_no-interactions->endstate"] = [complex_obj.deltaG]
    deltaG_df["deltaG"] = [deltaG_tot]
    
    deltaG_df.to_hdf(f"/nas0/ayoub/Impicit-Solvent-DDM/barton_cache/deltaG/deltaG_{complex_name}.h5", key="df", mode='w')
    
if __name__ == "__main__":
    import argparse

    #create an ArgumentParser object
    parser = argparse.ArgumentParser(description = 'Compute the detlaG of each system')
    #declare arguments
    parser.add_argument('--complex', type = str, help='mdout.h5 file', required=True)
    parser.add_argument('--receptor', type = str, help='mdout.h5 file', required=True)
    parser.add_argument('--ligand', type = str, help='mdout.h5 file', required=True)
    parser.add_argument('--boresch', type = str, help='mdout.h5 file', required=True)
    
    args = parser.parse_args()
    main(args.complex, args.receptor, args.ligand, args.boresch)
