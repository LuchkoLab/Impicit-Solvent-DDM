import os
import re
from email import message
from importlib.metadata import files
from pathlib import Path
from sre_constants import ANY
from typing import Optional, Type, TypedDict, Union

import pandas as pd
from matplotlib.backend_bases import key_press_handler
from toil.batchSystems import abstractBatchSystem
from toil.job import FileID, Job, JobFunctionWrappingJob, PromisedRequirement

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.postTreatment import create_mdout_dataframe
from implicit_solvent_ddm.restraints import RestraintMaker
from implicit_solvent_ddm.simulations import Simulation


class IntermidateRunner(Job):
    """
    Runner class that will initiate all MD simulations for the respected system (i.e. complex, ligand or receptor)
    IntermidateRunner is responsible for handling all the Simulations and collecting the output dataframes.
    """

    # simulations: dict[Simulation, int]
    def __init__(
        self,
        simulations: list[Simulation],
        restraints: RestraintMaker,
        post_process_no_solv_mdin: FileID,
        post_process_mdin: FileID,
        post_process_distruct: str,
        post_only: bool,
        config: Config,
        adaptive: bool = False,
        loaded_dataframe: list = [],
        post_output: Union[list, list[pd.DataFrame]] = [],
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ) -> None:
        super().__init__(
            memory,
            cores,
            disk,
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )

        self.simulations = simulations
        self.restraints = restraints
        self.no_solvent_mdin = post_process_no_solv_mdin
        self.mdin = post_process_mdin
        self.post_only = post_only
        self.config = config
        self.adaptive = adaptive
        self.post_output = post_output
        self.ligand_output = []
        self.receptor_output = []
        self.complex_output = []
        self._loaded_dataframe = loaded_dataframe
        self.post_process_distruct = post_process_distruct

    def run(self, fileStore):
        fileStore.logToMaster(f"post only is {self.post_only}")

        def run_post_process(job: Job, ran_simulation: Simulation):
            for post_simulation in self.simulations:
                directory_args = post_simulation.directory_args.copy()

                directory_args.update(self.update_postprocess_dirstruct(ran_simulation.directory_args))  # type: ignore
                # fileStore.logToMaster(f"RUNNER directory args {directory_args}\n")

                mdin = self.mdin

                if post_simulation.directory_args["igb_value"] == 6:
                    mdin = self.no_solvent_mdin

                # run simulation if its not endstate with endstate
                # if post_simulation.post:
                if (
                    post_simulation.inptraj != ran_simulation.inptraj
                    or post_simulation.inptraj == None
                ):
                    # endstate simulation has inptraj attribute
                    if ran_simulation.inptraj is not None:
                        input_traj = ran_simulation.inptraj

                    # whereas the intermidate states does not
                    else:
                        input_traj = job.rv(1)
                    # if job -> endstate
                    # if post is endstate wont access job.rv()
                    # but if post is not endstate then it will aceess job.rv

                    post_dirstruct = self.get_system_dirs(post_simulation.system_type)
                    fileStore.logToMaster(
                        f"get_system_dirs {post_simulation.system_type}"
                    )
                    fileStore.logToMaster(f"current dirstruct {post_dirstruct}")
                    post_process_job = Simulation(
                        executable="sander.MPI",
                        mpi_command=post_simulation.mpi_command,
                        num_cores=post_simulation.num_cores,
                        prmtop=post_simulation.prmtop,
                        incrd=post_simulation.incrd,
                        input_file=mdin,
                        restraint_file=post_simulation.restraint_file,
                        working_directory=post_simulation.working_directory,
                        directory_args=directory_args,
                        dirstruct=post_dirstruct,
                        inptraj=input_traj,
                        post_analysis=True,
                        restraint_key=post_simulation.restraint_key,
                    )
                    job.addChild(post_process_job)

                    data_frame = post_process_job.addFollowOnJobFn(
                        create_mdout_dataframe,
                        post_process_job.directory_args,
                        post_process_job.dirstruct,
                        post_process_job.output_dir,
                    )
                    self._loaded_dataframe.append(post_process_job.output_dir)
                else:
                    fileStore.logToMaster(f"parsing endstate post only")
                    data_frame = job.addFollowOnJobFn(
                        create_mdout_dataframe,
                        directory_args,
                        post_simulation.dirstruct,
                        post_simulation.output_dir,
                    )
                    self._loaded_dataframe.append(post_simulation.output_dir)

                self.post_output.append(data_frame.rv())
                fileStore.logToMaster(
                    f"First runs loaded data frames: {self._loaded_dataframe}"
                )

        # iterate and submit all intermidate simulations. Then followup with post-process
        for simulation in self.simulations:
            # if checking flat bottom constribution don't run

            # only post analysis

            fileStore.logToMaster(f"loaded dataframe: {self._loaded_dataframe}\n")
            fileStore.logToMaster(f"simulations args {simulation.directory_args}\n")
            if simulation.directory_args["state_label"] == "no_flat_bottom":
                continue

            if self._check_mdout(simulation=simulation) or simulation.inptraj != None:
                fileStore.logToMaster("simulation mdout may exisit?")
                fileStore.logToMaster(
                    f"simulation output directory {simulation.output_dir}"
                )

                if simulation.inptraj == None:
                    fileStore.logToMaster(
                        f"get mdtraj at directory:\n {simulation.output_dir}"
                    )
                    simulation.inptraj = [
                        fileStore.import_file(
                            "file://" + self._get_md_traj(simulation, fileStore),
                        )
                    ]
                self.only_post_analysis(
                    simulation, md_traj=simulation.inptraj, fileStore=fileStore
                )

            else:
                fileStore.logToMaster(f"RUNNING MD THEN POST")
                fileStore.logToMaster(
                    f"mdout does not exist path: {simulation.output_dir}"
                )
                run_post_process(
                    job=self.addChild(simulation), ran_simulation=simulation
                )

        return self

    def only_post_analysis(self, completed_sim: Simulation, md_traj, fileStore):
        """Assumes MD has already been completed and will only run post-analysis."""

        fileStore.logToMaster("RUNNING POST only\n")
        fileStore.logToMaster(f"loaded dataframe: {self._loaded_dataframe}")

        for post_simulation in self.simulations:
            directory_args = post_simulation.directory_args.copy()
            fileStore.logToMaster(f"directory args before update: {directory_args}\n")
            # fileStore.logToMaster(f"args {completed_sim.directory_args} & {md_traj}")
            directory_args.update(self.update_postprocess_dirstruct(completed_sim.directory_args))  # type: ignore
            fileStore.logToMaster(f"directory args after update: {directory_args}\n")
            mdin = self.mdin
            if post_simulation.directory_args["igb_value"] == 6:
                mdin = self.no_solvent_mdin

            # run simulation if its not endstate with endstate
            post_dirstruct = self.get_system_dirs(post_simulation.system_type)
            fileStore.logToMaster(f"post dirstruct {post_dirstruct}\n")

            post_process_job = Simulation(
                executable="sander.MPI",
                mpi_command=post_simulation.mpi_command,
                num_cores=post_simulation.num_cores,
                prmtop=post_simulation.prmtop,
                incrd=post_simulation.incrd,
                input_file=mdin,
                restraint_file=post_simulation.restraint_file,
                working_directory=post_simulation.working_directory,
                directory_args=directory_args,
                dirstruct=post_dirstruct,
                inptraj=md_traj,
                post_analysis=True,
                restraint_key=post_simulation.restraint_key,
            )

            if completed_sim.directory_args["runtype"] == "lambda_window":
                fileStore.logToMaster(f"COMPLETED MD simulation of lambda window")
                fileStore.logToMaster(
                    f"Using trajectory from {completed_sim.output_dir}\n"
                )

            mdout_parse = not "simulation_mdout.parquet.gzip" in os.listdir(
                post_process_job.output_dir
            )
            fileStore.logToMaster(f"mdout_parse: {mdout_parse}\n")
            fileStore.logToMaster(f"ADAPTIVE {self.adaptive}")

            if not "simulation_mdout.parquet.gzip" in os.listdir(
                post_process_job.output_dir
            ):
                fileStore.logToMaster(
                    f"simulations_mdout.parquet is not found in  {post_process_job.output_dir}\n"
                )

                fileStore.logToMaster(
                    f"RUNNING post analysis with inptraj trajecory: {md_traj}"
                )
                # fileStore.logToMaster(
                #     f"output postProces: {post_process_job.output_dir}"
                # )
                fileStore.logToMaster(
                    f"State potential energy {post_simulation.directory_args['state_label']}"
                )
                # fileStore.logToMaster(f"state args: {post_simulation.directory_args}")

                self.addChild(post_process_job)

                data_frame = post_process_job.addFollowOnJobFn(
                    create_mdout_dataframe,
                    post_process_job.directory_args,
                    post_process_job.dirstruct,
                    post_process_job.output_dir,
                )

                self.post_output.append(data_frame.rv())

            elif post_process_job.output_dir in self._loaded_dataframe:
                fileStore.logToMaster(f"ADAPTIVE lambda window set")
                fileStore.logToMaster(
                    f"Adapative restraints Is TRUE therefore {post_process_job.output_dir} is already loaded\n"
                )
                fileStore.logToMaster(
                    f"simulations output that were loaded \n {self._loaded_dataframe}"
                )
                continue

            else:
                if post_simulation.directory_args["state_label"] == "lambda_window":
                    fileStore.logToMaster(f"Retrieving ADAPTIVE lambda window set")
                    fileStore.logToMaster(
                        f"Adapative restraints Is TRUE therefore {post_process_job.output_dir} is being loaded\n"
                    )
                fileStore.logToMaster("Retrieving completed post-analysis data")
                self.post_output.append(
                    pd.read_parquet(
                        os.path.join(
                            post_process_job.output_dir, "simulation_mdout.parquet.gzip"
                        ),
                    )
                )
            self._loaded_dataframe.append(post_process_job.output_dir)

    def _add_complex_simulation(
        self,
        conformational,
        orientational,
        mdin,
        restraint_file,
        charge=1.0,
        charge_parm=None,
    ):
        con_force = float(round(conformational, 3))
        orient_force = float(round(orientational, 3))

        dirs_args = (
            {
                "topology": self.config.endstate_files.complex_parameter_filename,
                "state_label": "lambda_window",
                "extdiel": 78.5,
                "charge": charge,
                "igb": f"igb_{self.config.intermidate_args.igb_solvent}",
                "igb_value": self.config.intermidate_args.igb_solvent,
                "conformational_restraint": con_force,
                "orientational_restraints": orient_force,
                "filename": f"state_8_{con_force}_{orient_force}_prod",
                "runtype": f"Running restraint window. Conformational restraint: {con_force} and orientational restraint: {orient_force}",
                "topdir": self.config.system_settings.top_directory_path,
            },
        )

        parm_file = self.config.endstate_files.complex_parameter_filename
        if charge_parm is not None:
            parm_file = charge_parm.rv()
            dirs_args[0].update({"state_label": "electrostatics"})  # type: ignore

        else:
            restraint_file = restraint_file.rv()

        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.complex_ncores,
            prmtop=parm_file,
            incrd=self.config.inputs["endstate_complex_lastframe"],
            input_file=mdin,
            restraint_file=restraint_file,
            working_directory=self.config.system_settings.working_directory,
            system_type="complex",
            directory_args=dirs_args[0],
            dirstruct="dirstruct_halo",
        )

        self.simulations.append(new_job)

    def _add_ligand_simulation(
        self,
        conformational,
        mdin,
        restraint_file,
        charge=1.0,
        charge_parm=None,
    ):
        con_force = float(round(conformational, 3))

        dirs_args = (
            {
                "topology": self.config.endstate_files.ligand_parameter_filename,
                "state_label": "lambda_window",
                "conformational_restraint": con_force,
                "igb": f"igb_{self.config.intermidate_args.igb_solvent}",
                "extdiel": 78.5,
                "charge": charge,
                "igb_value": self.config.intermidate_args.igb_solvent,
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, Conformational restraint: {con_force}",
                "topdir": self.config.system_settings.top_directory_path,
            },
        )

        parm_file = self.config.endstate_files.ligand_parameter_filename
        if charge_parm is not None:
            parm_file = charge_parm.rv()
            dirs_args[0].update({"state_label": "electrostatics"})  # type: ignore
            dirs_args[0].update({"igb": "igb_6"})
            dirs_args[0].update({"filename": "state_4_prod"})
            dirs_args[0].update({"extdiel": 0.0})
            dirs_args[0].update(
                {
                    "runtype": f"Scailing ligand charges: {charge}",
                }
            )
        else:
            restraint_file = restraint_file.rv()
        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.ligand_ncores,
            prmtop=parm_file,
            incrd=self.config.inputs["ligand_endstate_frame"],
            input_file=mdin,
            restraint_file=restraint_file,
            working_directory=self.config.system_settings.working_directory,
            system_type="ligand",
            directory_args=dirs_args[0],
            dirstruct="dirstruct_apo",
        )

        self.simulations.append(new_job)

    def _add_receptor_simulation(self, conformational, mdin, restraint_file):
        con_force = float(round(conformational, 3))
        new_job = Simulation(
            executable=self.config.system_settings.executable,
            mpi_command=self.config.system_settings.mpi_command,
            num_cores=self.config.num_cores_per_system.receptor_ncores,
            prmtop=self.config.endstate_files.receptor_parameter_filename,
            incrd=self.config.inputs["receptor_endstate_frame"],
            input_file=mdin,
            restraint_file=restraint_file.rv(),
            working_directory=self.config.system_settings.working_directory,
            system_type="receptor",
            directory_args={
                "topology": self.config.endstate_files.receptor_parameter_filename,
                "state_label": "lambda_window",
                "extdiel": 78.5,
                "charge": 1.0,
                "igb": f"igb_{self.config.intermidate_args.igb_solvent}",
                "igb_value": self.config.intermidate_args.igb_solvent,
                "conformational_restraint": con_force,
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, Conformational restraint: {con_force}",
                "topdir": self.config.system_settings.top_directory_path,
            },
            dirstruct="dirstruct_apo",
        )

        self.simulations.append(new_job)

    @classmethod
    def new_runner(
        cls: Type["IntermidateRunner"],
        config: Config,
        obj: dict,
    ):
        return cls(
            simulations=obj["simulations"],
            restraints=obj["restraints"],
            config=config,
            post_process_distruct=obj["post_process_distruct"],
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            adaptive=True,
            post_only=True,
            post_output=obj["post_output"],
            loaded_dataframe=obj["_loaded_dataframe"],
        )

    @staticmethod
    def _get_md_traj(simulation: Simulation, fileStore):
        """Return an absolute path to completed AMBER (.nc) trajectory filename.

        Parameters
        ----------
        simulation: Simulation
            Simulation class object which contains all required MD input arguments.
        fileStore: job.fileStore
            Toil interface to read and write files.
        Returns
        -------
        Filepath to AMBER trajectory (.nc) file.
        """
        return os.path.join(
            simulation.output_dir,
            list(
                filter(
                    lambda file: re.match(r"^.*\.nc$", file),
                    os.listdir(simulation.output_dir),
                )
            )[0],
        )

    @staticmethod
    def _check_mdout(simulation: Simulation) -> bool:
        if "mdout" in os.listdir(simulation.output_dir):
            for line in reversed(
                open(os.path.join(simulation.output_dir, "mdout")).readlines()
            ):
                if "Final Performance Info" in line:
                    return True
        return False

    @staticmethod
    def update_postprocess_dirstruct(
        run_time_args: dict,
    ) -> dict[str, Union[str, object]]:
        if "orientational_restraints" in run_time_args.keys():
            return {
                "traj_state_label": run_time_args["state_label"],
                "trajectory_restraint_conrest": run_time_args[
                    "conformational_restraint"
                ],
                "trajectory_restraint_orenrest": run_time_args[
                    "orientational_restraints"
                ],
                "traj_extdiel": run_time_args["extdiel"],
                "traj_igb": run_time_args["igb"],
                "traj_charge": run_time_args["charge"],
                "filename": f"{run_time_args['filename']}_postprocess",
            }

        return {
            "traj_state_label": run_time_args["state_label"],
            "trajectory_restraint_conrest": run_time_args["conformational_restraint"],
            "traj_igb": run_time_args["igb"],
            "traj_extdiel": run_time_args["extdiel"],
            "traj_charge": run_time_args["charge"],
            "filename": f"{run_time_args['filename']}_postprocess",
        }

    @staticmethod
    def get_system_dirs(system_type):
        if system_type == "ligand" or system_type == "receptor":
            return "post_process_apo"

        return "post_process_halo"

    # /nas0/ayoub/sampl9_runs/sampl9_extend_windows_diel/WP6_G2_Hmass/lambda_window/1.0/78.5/-0.2857142857142864/3.7142857142857135/WP6_G2_Hmass_state_8_0.8203353560076375_13.1253656961222_prod_traj.nc
