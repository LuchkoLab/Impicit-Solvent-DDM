"""
class that will parse in pandas dataframe for mbar analysis 
"""
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
from implicit_solvent_ddm.restraints import RestraintMaker
from implicit_solvent_ddm.simulations import Simulation

WORKDIR = os.getcwd()

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184


class ConsolidateData(Job):
    
    def __init__(self, complex_adative_run, 
                 ligand_adaptive_run, receptor_adaptive_run,
                 temperature:float, max_conformation_force,
                 max_orientational_force,
                 boresch_df: RestraintMaker,
                 working_path, complex_filename, 
                 ligand_filename, receptor_filename, 
                 ):
        Job.__init__(self,  memory="2G", cores=2, disk="3G")
        self.temp = temperature
        self.complex_adative_run = complex_adative_run
        self.receptor_adaptive_run = receptor_adaptive_run 
        self.ligand_adaptive_run = ligand_adaptive_run
        
        self.max_con_force = str(max_conformation_force)
        self.max_orien_force = str(max_orientational_force)
        self.boresch = boresch_df
        self.complex_name = re.sub(r"\..*", "", os.path.basename(complex_filename))
        self.ligand_name = re.sub(r"\..*", "", os.path.basename(ligand_filename))
        self.receptor_name = re.sub(r"\..*", "", os.path.basename(receptor_filename))
        self.working_path = working_path
        #self._set_kcal_per_mol()
        
    # def _set_kcal_per_mol(self):
        
    #     self.complex_fe_kcal_per_mol = self.complex_fe * self.kcals_per_Kt
    #     self.complex_error_kcal_per_mol = self.complex_error * self.kcals_per_Kt
    #     self.ligand_fe_kcal_per_mol  = self.ligand_fe * self.kcals_per_Kt
    #     self.ligand_error_kcal_per_mol  = self.ligand_error * self.kcals_per_Kt
    #     self.receptor_fe_kcal_per_mol  = self.receptor_fe * self.kcals_per_Kt
    #     self.receptor_error_kcal_per_mol  = self.receptor_error * self.kcals_per_Kt
    @property
    def complex_mbar_formatted_df(self)->pd.DataFrame:
        
        return self.complex_adative_run[1] * self.kcals_per_Kt
    
    @property 
    def complex_fe(self)->pd.DataFrame:
        
        return self.complex_adative_run[0][0] * self.kcals_per_Kt
    
    @property
    def complex_error(self)->pd.DataFrame:
        
        return self.complex_adative_run[0][1] * self.kcals_per_Kt
    
    @property
    def ligand_mbar_formatted_df(self)->pd.DataFrame:
        
        return self.ligand_adaptive_run[1] * self.kcals_per_Kt
    
    @property
    def ligand_fe(self)->pd.DataFrame:
        
        return self.ligand_adaptive_run[0][0] * self.kcals_per_Kt
    
    @property
    def ligand_error(self)->pd.DataFrame:
    
        return self.ligand_adaptive_run[0][1] * self.kcals_per_Kt
    
    @property
    def receptor_mbar_formatted_df(self)->pd.DataFrame:
        
        return self.receptor_adaptive_run[1] * self.kcals_per_Kt
    
    
    @property
    def receptor_fe(self):
        
        return self.receptor_adaptive_run[0][0] * self.kcals_per_Kt
    
    @property
    def receptor_error(self)->pd.DataFrame:
        
        return self.receptor_adaptive_run[0][1] * self.kcals_per_Kt
    
    
    @property
    def kcals_per_Kt(self):
        
        return ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * self.temp
    

    @property
    def _get_ligand_deltaG(self):
        
        return self.ligand_fe.loc[("endstate", "78.5", "1.0", "0.0"), [("electrostatics", "0.0", "0.0", self.max_con_force)]].values[0]
    
    @property
    def _get_receptor_deltaG(self):
        
        return self.receptor_fe.loc[("endstate", "78.5", "1.0", "0.0"), [("no_gb", "0.0", "1.0", self.max_con_force)]].values[0]
    
    @property
    def _get_complex_deltaG(self):
        
        return self.complex_fe.loc[("no_interactions", "0.0", "0.0", f"{self.max_con_force}_{self.max_orien_force}"), [("endstate", "78.5", "1.0", "0.0_0.0")]].values[0]
    
    @property
    def _get_boresch_standard_state(self):
        
        return self.boresch.boresch_deltaG["DeltaG"].values[0]
   
    @property
    def compute_binding_deltaG(self)->float:

        return self._get_complex_deltaG + self._get_ligand_deltaG + self._get_receptor_deltaG + self._get_boresch_standard_state
    
    
    def run(self, fileStore):
        
        output_path = os.path.join(f"{self.working_path}", f".cache/{self.complex_name}")    
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # parse out formatted dataframe
        self.complex_mbar_formatted_df.to_hdf(
            f"{output_path}/{self.complex_name}_formatted.h5", key="df", mode="w"
        )
        self.receptor_mbar_formatted_df.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_formatted.h5", key="df", mode="w"
        )
        self.ligand_mbar_formatted_df.to_hdf(
            f"{output_path}/ligand_{self.ligand_name}_formatted.h5", key="df", mode="w"
        )
        
        # parse out free energies
        self.complex_fe.to_hdf(
        f"{output_path}/{self.complex_name}_fe.h5", key="df", mode="w"
        )
        self.receptor_fe.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_fe.h5", key="df", mode="w"
        )
        self.ligand_fe.to_hdf(
            f"{output_path}/ligand_{self.ligand_name}_fe.h5", key="df", mode="w"
        )
        
        
        # parse out errors of mbar
        self.complex_error.to_hdf(
            f"{output_path}/{self.complex_name}_error.h5", key="df", mode="w"
        )
        self.receptor_error.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_error.h5", key="df", mode="w"
        )
        self.ligand_error.to_hdf(
            f"{output_path}/ligand_{self.ligand_name}_error.h5", key="df", mode="w"
        )
        
        # parse out boresch restraints dataframe

        self.boresch.boresch_deltaG.to_hdf(
            f"{output_path}/boresch_{self.complex_name}.h5", key="df", mode="w"
        )
        deltaG_df = pd.DataFrame()
        
        
        fileStore.logToMaster(f"BORESCH standard state {self._get_boresch_standard_state}\n")
        
        fileStore.logToMaster(f"Ligand Delta G: {self._get_ligand_deltaG}")
        fileStore.logToMaster(f"Receptor Delta G: {self._get_receptor_deltaG}\n")
        
        fileStore.logToMaster(f"Complex unique index keys:\n {self.complex_fe.index.unique()}\n")
         
        fileStore.logToMaster(f"Complex Delta G: {self._get_complex_deltaG}\n")

        deltaG_df[f"{self.ligand_name}_endstate->no_charges"] = [self._get_ligand_deltaG]
        deltaG_df[f"{self.receptor_name}_endstate->no_gb"] = [self._get_receptor_deltaG]
        deltaG_df["boresch_restraints"] = [self._get_boresch_standard_state]
        deltaG_df[f"{self.complex_name}_no-interactions->endstate"] = [
           self._get_complex_deltaG
        ]
        deltaG_df["deltaG"] = [self.compute_binding_deltaG]

        deltaG_df.to_hdf(
            f"{output_path}/deltaG_{self.complex_name}.h5", key="df", mode="w"
        )
        
        
class PostTreatment(Job):
    def __init__(
        self,
        simulation_data: List[pd.DataFrame],
        temp: float,
        system: str,
        max_conformation_force: float,
        max_orientational_force=None,
    ) -> None:
        super().__init__()
        self.simulations_data = simulation_data
        self.temp = temp
        self.system = system
        self.max_con_force = str(max_conformation_force)
        self.max_orien_force = str(max_orientational_force)
        self._kcals_per_Kt()

    def _kcals_per_Kt(self):
        self.kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * self.temp

    def _load_dfs(self):

        self.df = pd.concat(self.simulations_data, axis=0, ignore_index=True)

        self.name = self.df["solute"].iloc[0]

    def _create_MBAR_format(self):

        self.df = self.df.set_index(
            [
                "solute",
                "parm_state",
                "extdiel",
                "charge",
                "parm_restraints",
                "traj_state",
                "traj_extdiel",
                "traj_charge",
                "traj_restraints",
                "Frames",
            ],
            drop=True,
        )
            
        self.df = self.df[["ENERGY"]]
        self.df = self.df.unstack(["parm_state", "extdiel",  "charge", "parm_restraints"])  # type: ignore
        self.df = self.df.reset_index(["Frames", "solute"], drop=True)
        states = [_ for _ in zip(*self.df.columns)][1]
        extdiels = [_ for _ in zip(*self.df.columns)][2]
        charges = [_ for _ in zip(*self.df.columns)][3]
        restraints = [_ for _ in zip(*self.df.columns)][4]

        column_names = [
            (state, extdiel, charge, restraint)
            for state, extdiel, charge, restraint in zip(states, extdiels, charges, restraints)
        ]

        self.df.columns = column_names  # type: ignore

        # divide by Kcal per Kt
        self.df = self.df / self.kcals_per_Kt

    def compute_binding_deltaG(self, system1: float, system2: float, boresch_dG: float):

        return self.deltaG + system1 + system2 + boresch_dG

    def run(self, fileStore):

        self._load_dfs()
        self._create_MBAR_format()
        fileStore.logToMaster(f"self.df {self.df}")

        equil_info = pdmbar.detectEquilibration(self.df)

        df_subsampled = pdmbar.subsampleCorrelatedData(self.df, equil_info=equil_info)

        fe, error, mbar = pdmbar.mbar(df_subsampled)

        fe = fe * self.kcals_per_Kt

        # then multiply by kt
        error = error * self.kcals_per_Kt

        if self.system == "ligand":
            self.deltaG = fe.loc[("endstate", "78.5", "1.0", "0.0"), [("electrostatics", "0.0", "0.0", self.max_con_force)]].values[0]  # type: ignore

        elif self.system == "receptor":
            self.deltaG = fe.loc[("endstate", "78.5", "1.0", "0.0"), [("no_gb", "0.0", "1.0", self.max_con_force)]].values[0]  # type: ignore

        # system is complex
        else:
            self.deltaG = fe.loc[("no_interactions", "0.0", "0.0", f"{self.max_con_force}_{self.max_orien_force}"), [("endstate", "78.5", "1.0", "0.0_0.0")]].values[0]  # type: ignore

        self.fe = fe
        self.error = error
        self.mbar = mbar
        return self


def consolidate_output(
    job,
    ligand_system: PostTreatment,
    receptor_system: PostTreatment,
    complex_system: PostTreatment,
    boresch_df: RestraintMaker,
):

    output_path = os.path.join(f"{WORKDIR}", f".cache/{complex_system.name}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # parse out formatted dataframe
    complex_system.df.to_hdf(
        f"{output_path}/{complex_system.name}_formatted.h5", key="df", mode="w"
    )
    receptor_system.df.to_hdf(
        f"{output_path}/receptor_{complex_system.name}_formatted.h5", key="df", mode="w"
    )
    ligand_system.df.to_hdf(
        f"{output_path}/ligand_{complex_system.name}_formatted.h5", key="df", mode="w"
    )

    # parse out free energies
    complex_system.fe.to_hdf(
        f"{output_path}/{complex_system.name}_fe.h5", key="df", mode="w"
    )
    receptor_system.fe.to_hdf(
        f"{output_path}/receptor_{complex_system.name}_fe.h5", key="df", mode="w"
    )
    ligand_system.fe.to_hdf(
        f"{output_path}/ligand_{complex_system.name}_fe.h5", key="df", mode="w"
    )

    # parse out errors of mbar
    complex_system.error.to_hdf(
        f"{output_path}/{complex_system.name}_error.h5", key="df", mode="w"
    )
    receptor_system.error.to_hdf(
        f"{output_path}/receptor_{complex_system.name}_error.h5", key="df", mode="w"
    )
    ligand_system.error.to_hdf(
        f"{output_path}/ligand_{complex_system.name}_error.h5", key="df", mode="w"
    )

    # parse out boresch restraints dataframe

    boresch_df.boresch_deltaG.to_hdf(
        f"{output_path}/boresch_{complex_system.name}.h5", key="df", mode="w"
    )

    boresch_dG = boresch_df.boresch_deltaG["DeltaG"].values[0]
    # compute total deltaG
    deltaG_tot = complex_system.compute_binding_deltaG(system1=ligand_system.deltaG, system2=receptor_system.deltaG, boresch_dG=boresch_dG)  # type: ignore

    deltaG_df = pd.DataFrame()

    deltaG_df[f"{ligand_system.name}_endstate->no_charges"] = [ligand_system.deltaG]
    deltaG_df[f"{receptor_system.name}_endstate->no_gb"] = [receptor_system.deltaG]
    deltaG_df["boresch_restraints"] = [boresch_dG]
    deltaG_df[f"{complex_system.name}_no-interactions->endstate"] = [
        complex_system.deltaG
    ]
    deltaG_df["deltaG"] = [deltaG_tot]

    deltaG_df.to_hdf(
        f"{output_path}/deltaG_{complex_system.name}.h5", key="df", mode="w"
    )


def create_mdout_dataframe(
    job, directory_args: dict, dirstruct: str, output_dir: str, compress:bool=True,
) -> pd.DataFrame:

    sim = Dirstruct("mdgb", directory_args, dirstruct=dirstruct)

    mdout = f"{output_dir}/mdout"

    job.log(f"List files in postprocess directory {os.listdir(output_dir)}\n")
    run_args = sim.dirStruct.fromPath2Dict(mdout)
    data = min_to_dataframe(mdout)

    # data["traj_state_label"] = run_args["traj_state_label"]
    # data["state_label"] = run_args["state_label"]

    data["solute"] = run_args["topology"]
    data["parm_state"] = run_args["state_label"]
    data["traj_state"] = run_args["traj_state_label"]
    data["Frames"] = data.index
    data["charge"] = run_args["charge"]
    data["traj_charge"] = run_args["traj_charge"]
    data["parm_restraints"] = run_args["conformational_restraint"]
    data["traj_restraints"] = run_args["trajectory_restraint_conrest"]
    data["extdiel"] = run_args["extdiel"]
    data["traj_extdiel"] = run_args["traj_extdiel"]
    # complex datastructure
    if "trajectory_restraint_orenrest" in run_args.keys():
        data[
            "parm_restraints"
        ] = f"{run_args['conformational_restraint']}_{run_args['orientational_restraints']}"
        data[
            "traj_restraints"
        ] = f"{run_args['trajectory_restraint_conrest']}_{run_args['trajectory_restraint_orenrest']}"
    
    if compress:
        data.to_parquet(f"{output_dir}/simulation_mdout.parquet.gzip", compression='gzip') 
        #data.to_parquet(f"{output_dir}/simulation_mdout.zip",  compression="gzip")
    
    if os.path.exists(mdout):
        os.remove(mdout)
    
    return data
