"""
class that will parse in pandas dataframe for mbar analysis 
"""
import os
import re


import pandas as pd
from toil.job import Job
from implicit_solvent_ddm.get_dirstruct import Dirstruct
from implicit_solvent_ddm.mdout import min_to_dataframe

from implicit_solvent_ddm.restraints import RestraintMaker
from typing import Union, Optional

WORKDIR = os.getcwd()

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184


class ConsolidateData(Job):
    def __init__(
        self,
        complex_adative_run,
        ligand_adaptive_run,
        receptor_adaptive_run,
        flat_botton_run,
        temperature: float,
        max_conformation_force,
        max_orientational_force,
        boresch_df: RestraintMaker,
        working_path,
        complex_filename,
        ligand_filename,
        receptor_filename,
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ):
        Job.__init__(
            self,
            memory="2G",
            cores=2,
            disk="3G",
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )
        self.temp = temperature
        self.complex_adative_run = complex_adative_run
        self.receptor_adaptive_run = receptor_adaptive_run
        self.ligand_adaptive_run = ligand_adaptive_run
        self.flat_botton_run = flat_botton_run
        self.max_con_force = str(max_conformation_force)
        self.max_orien_force = str(max_orientational_force)
        self.boresch = boresch_df
        self.complex_name = re.sub(r"\..*", "", os.path.basename(complex_filename))
        self.ligand_name = re.sub(r"\..*", "", os.path.basename(ligand_filename))
        self.receptor_name = re.sub(r"\..*", "", os.path.basename(receptor_filename))
        self.working_path = working_path

    @property
    def complex_mbar_formatted_df(self) -> pd.DataFrame:
        return self.complex_adative_run[1] * self.kcals_per_Kt

    @property
    def complex_fe(self) -> pd.DataFrame:
        return self.complex_adative_run[0][0] * self.kcals_per_Kt

    @property
    def complex_error(self) -> pd.DataFrame:
        return self.complex_adative_run[0][1] * self.kcals_per_Kt

    @property
    def ligand_mbar_formatted_df(self) -> pd.DataFrame:
        return self.ligand_adaptive_run[1] * self.kcals_per_Kt

    @property
    def ligand_fe(self) -> pd.DataFrame:
        return self.ligand_adaptive_run[0][0] * self.kcals_per_Kt

    @property
    def ligand_error(self) -> pd.DataFrame:
        return self.ligand_adaptive_run[0][1] * self.kcals_per_Kt

    @property
    def receptor_mbar_formatted_df(self) -> pd.DataFrame:
        return self.receptor_adaptive_run[1] * self.kcals_per_Kt

    @property
    def receptor_fe(self):
        return self.receptor_adaptive_run[0][0] * self.kcals_per_Kt

    @property
    def receptor_error(self) -> pd.DataFrame:
        return self.receptor_adaptive_run[0][1] * self.kcals_per_Kt

    @property
    def flat_bottom_fe(self):
        return self.flat_botton_run[0][0] * self.kcals_per_Kt

    @property
    def kcals_per_Kt(self):
        return ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * self.temp

    @property
    def _get_ligand_deltaG(self):
        return self.ligand_fe.loc[
            ("endstate", "78.5", "1.0", "0.0"),
            [("electrostatics", "0.0", "0.0", self.max_con_force)],
        ].values[0]

    @property
    def _get_receptor_deltaG(self):
        return self.receptor_fe.loc[
            ("endstate", "78.5", "1.0", "0.0"),
            [("no_gb", "0.0", "1.0", self.max_con_force)],
        ].values[0]

    @property
    def _get_complex_deltaG(self):
        return self.complex_fe.loc[
            (
                "no_interactions",
                "0.0",
                "0.0",
                f"{self.max_con_force}_{self.max_orien_force}",
            ),
            [("endstate", "78.5", "1.0", "0.0_0.0")],
        ].values[0]

    @property
    def _flat_bottom_contribution(self):
        return self.flat_bottom_fe.loc[
            (
                ("endstate", "78.5", "1.0", "0.0_0.0"),
                [("no_flat_bottom", "78.5", "1.0", "0.0_0.0")],
            )
        ].values[0]

    @property
    def _get_boresch_standard_state(self):
        return self.boresch.boresch_deltaG["DeltaG"].values[0]

    @property
    def compute_binding_deltaG(self) -> float:
        return (
            self._get_complex_deltaG
            + self._get_ligand_deltaG
            + self._get_receptor_deltaG
            + self._get_boresch_standard_state
            + self._flat_bottom_contribution
        )

    def run(self, fileStore):
        output_path = os.path.join(
            f"{self.working_path}", f".cache/{self.complex_name}"
        )

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # parse out formatted dataframe
        self.complex_mbar_formatted_df.to_hdf(
            f"{output_path}/{self.complex_name}_formatted.h5", key="df", mode="w"
        )
        self.receptor_mbar_formatted_df.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_formatted.h5",
            key="df",
            mode="w",
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

        fileStore.logToMaster(
            f"BORESCH standard state {self._get_boresch_standard_state}\n"
        )

        fileStore.logToMaster(f"Ligand Delta G: {self._get_ligand_deltaG}")
        fileStore.logToMaster(f"Receptor Delta G: {self._get_receptor_deltaG}\n")

        fileStore.logToMaster(
            f"Complex unique index keys:\n {self.complex_fe.index.unique()}\n"
        )

        fileStore.logToMaster(f"Complex Delta G: {self._get_complex_deltaG}\n")

        deltaG_df[f"{self.ligand_name}_endstate->no_charges"] = [
            self._get_ligand_deltaG
        ]
        deltaG_df[f"{self.receptor_name}_endstate->no_gb"] = [self._get_receptor_deltaG]
        deltaG_df["boresch_restraints"] = [self._get_boresch_standard_state]
        deltaG_df["flat_bottom_contribution"] = [self._flat_bottom_contribution]
        deltaG_df[f"{self.complex_name}_no-interactions->endstate"] = [
            self._get_complex_deltaG
        ]
        deltaG_df["deltaG"] = [self.compute_binding_deltaG]

        deltaG_df.to_hdf(
            f"{output_path}/deltaG_{self.complex_name}.h5", key="df", mode="w"
        )


def create_mdout_dataframe(
    job,
    directory_args: dict,
    dirstruct: str,
    output_dir: str,
    compress: bool = True,
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
        data.to_parquet(
            f"{output_dir}/simulation_mdout.parquet.gzip", compression="gzip"
        )
        # data.to_parquet(f"{output_dir}/simulation_mdout.zip",  compression="gzip")

    if os.path.exists(mdout):
        os.remove(mdout)

    return data
