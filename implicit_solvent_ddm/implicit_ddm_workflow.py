# from implicit_solvent_ddm.remd import run_remd
import itertools
import logging
import os
import os.path
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from toil.common import Toil
from toil.job import Job, JobFunctionWrappingJob

from implicit_solvent_ddm.alchemical import (get_intermidate_parameter_files,
                                             split_complex_system)
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.mdin import generate_replica_mdin, get_mdins
# from alchemical import get_intermidate_parameter_files, split_complex_system
# from config import Config
# from mdin import get_mdins
# from implicit_solvent_ddm.mdin import get_mdins
# from postTreatment import PostTreatment, create_mdout_dataframe
from implicit_solvent_ddm.postTreatment import (PostTreatment,
                                                consolidate_output,
                                                create_mdout_dataframe)
from implicit_solvent_ddm.restraints import (FlatBottom, RestraintMaker,
                                             write_empty_restraint)
from implicit_solvent_ddm.simulations import (ExtractTrajectories,
                                              REMDSimulation, Simulation)

# from restraints import (get_conformational_restraints,
#                         get_flat_bottom_restraints,
#                         get_orientational_restraints, write_empty_restraint,
#                         write_restraint_forces)
# from simulations import ExtractTrajectories, REMDSimulation, Simulation


# local imports

working_directory = os.getcwd()


def ddm_workflow(
    job: JobFunctionWrappingJob,
    config: Config,
    inptraj_id=None,
    solute="system",
    dirstuct_traj_args={},
    post_process=False,
    restraints="",
):
    """
    Double decoupling workflow

    Runs long simulations at the end states (receptor, ligand & complex).
    Creates orientational and conformational restraints for intermediate states (short simulation runs).

    Parameters
    ----------
    toil: class toil.common.Toil
        A contect manager that represents a Toil workflow
    df_config_inputs: pandas.DataFrame
        A data frame containing user's config parameters and imported Toil fileID's
    argSet: dict
        Dictionary containing user's config parameters
    work_dir: str
        User's initial working path directory

    Returns
    -------
    end_state_job: toil.job.JobFunctionWrappingJob
        contains the entire workflow in indiviudual jobs.
    """
    # temp_dir = job.fileStore.getLocalTempDir()

    calc_list = []
    ligand_df = []
    receptor_df = []
    complex_df = []

    # post process setup workflow
    if post_process:
        inptraj_id = inptraj_id
        workflow = config.workflow.update_workflow(
            solute, config.workflow.run_endstate_method
        )
        ligand_receptor_dirstruct = "post_process_apo"
        complex_dirstruct = "post_process_halo"
        mdin_no_solv = config.inputs["post_nosolv_mdin"]
        default_mdin = config.inputs["post_mdin"]
        # md_jobs = job.addChildJobFn(initilized_jobs)

    else:

        workflow = config.workflow
        ligand_receptor_dirstruct = "dirstruct_apo"
        complex_dirstruct = "dirstruct_halo"

        setup_inputs = job.wrapJobFn(
            get_intermidate_parameter_files,
            config.endstate_files.complex_parameter_filename,
            config.endstate_files.complex_coordinate_filename,
            config.amber_masks.ligand_mask,
            config.amber_masks.receptor_mask,
        )
        # Add inputs as first child to root job
        job.addChild(setup_inputs)

        # config.inputs["ligand_no_vdw_ID"] = setup_inputs.rv(0)
        # config.inputs["receptor_no_vdw_ID"] = setup_inputs.rv(2)
        config.inputs["ligand_no_charge_parm_ID"] = setup_inputs.rv(0)
        config.inputs["complex_ligand_no_charge_ID"] = setup_inputs.rv(1)
        config.inputs["complex_no_ligand_interaction_ID"] = setup_inputs.rv(2)

        # fill in intermidate mdin
        mdins = setup_inputs.addChildJobFn(
            get_mdins, config.intermidate_args.mdin_intermidate_config
        )

        # set intermidate mdin files
        config.inputs["default_mdin"] = mdins.rv(0)
        config.inputs["no_solvent_mdin"] = mdins.rv(1)
        config.inputs["post_mdin"] = mdins.rv(2)
        config.inputs["post_nosolv_mdin"] = mdins.rv(3)
        default_mdin = mdins.rv(0)
        mdin_no_solv = mdins.rv(1)
        # write empty restraint.RST
        empty_restraint = setup_inputs.addChildJobFn(write_empty_restraint)
        config.inputs["empty_restraint"] = empty_restraint.rv()

        # flat bottom restraints potential restraints
        flat_bottom_template = setup_inputs.addChild(FlatBottom(config=config))

        config.inputs["flat_bottom_restraint"] = flat_bottom_template.rv()

        # Begin running END State Simulations
        # config.workflow.run_endstate_method and post_process==False:

        if workflow.run_endstate_method:

            endstate_method = setup_inputs.addFollowOnJobFn(initilized_jobs)

            if config.endstate_method.endstate_method_type == "remd":
                # prmtop = job.fileStore.readGlobalFile(
                #     config.endstate_files.complex_parameter_filename,
                #     userPath=os.path.join(
                #         tempdir,
                #         os.path.basename(
                #             config.endstate_files.complex_parameter_filename
                #         ),
                #     ),
                # )
                equil_mdins = endstate_method.addChildJobFn(
                    generate_replica_mdin,
                    config.endstate_method.remd_args.equil_template_mdin,
                    config.endstate_method.remd_args.temperatures,
                    runtype="relax",
                )
                remd_mdins = equil_mdins.addChildJobFn(
                    generate_replica_mdin,
                    config.endstate_method.remd_args.remd_template_mdin,
                    config.endstate_method.remd_args.temperatures,
                    runtype="remd",
                )

                minimization_complex = remd_mdins.addChild(
                    Simulation(
                        executable=config.system_settings.executable,
                        mpi_command=config.system_settings.mpi_command,
                        num_cores=config.num_cores_per_system.complex_ncores,
                        prmtop=config.endstate_files.complex_parameter_filename,
                        incrd=config.endstate_files.complex_coordinate_filename,
                        input_file=config.inputs["min_mdin"],
                        restraint_file=config.inputs["flat_bottom_restraint"],
                        directory_args={
                            "runtype": "minimization",
                            "filename": "min",
                            "topology": config.endstate_files.complex_parameter_filename,
                            "topdir": config.system_settings.top_directory_path,
                        },
                        working_directory=config.system_settings.working_directory,  #
                    )
                )
                # config.endstate_method.remd_args.nthreads
                equilibrate_complex = minimization_complex.addFollowOn(
                    REMDSimulation(
                        executable=config.system_settings.executable,
                        mpi_command=config.system_settings.mpi_command,
                        num_cores=config.endstate_method.remd_args.nthreads_complex,
                        prmtop=config.endstate_files.complex_parameter_filename,
                        incrd=minimization_complex.rv(0),
                        input_file=equil_mdins.rv(),
                        restraint_file=config.inputs["flat_bottom_restraint"],
                        runtype="equil",
                        ngroups=config.endstate_method.remd_args.ngroups,
                        directory_args={
                            "runtype": "equilibration",
                            "topology": config.endstate_files.complex_parameter_filename,
                            "topdir": config.system_settings.top_directory_path,
                        },
                        working_directory=config.system_settings.working_directory,
                        memory=config.system_settings.memory,
                        disk=config.system_settings.disk,
                    )
                )

                remd_complex = equilibrate_complex.addFollowOn(
                    REMDSimulation(
                        executable=config.system_settings.executable,
                        mpi_command=config.system_settings.mpi_command,
                        num_cores=config.endstate_method.remd_args.nthreads_complex,
                        ngroups=config.endstate_method.remd_args.ngroups,
                        prmtop=config.endstate_files.complex_parameter_filename,
                        incrd=equilibrate_complex.rv(0),
                        input_file=remd_mdins.rv(),
                        working_directory=config.system_settings.working_directory,
                        restraint_file=config.inputs["flat_bottom_restraint"],
                        runtype="remd",
                        directory_args={
                            "runtype": "remd",
                            "topology": config.endstate_files.complex_parameter_filename,
                            "topdir": config.system_settings.top_directory_path,
                        },
                        memory=config.system_settings.memory,
                        disk=config.system_settings.disk,
                    )
                )

                # extact target temparture trajetory and last frame
                extract_complex = remd_complex.addFollowOn(
                    ExtractTrajectories(
                        config.endstate_files.complex_parameter_filename,
                        remd_complex.rv(1),
                        config.endstate_method.remd_args.target_temperature,
                    )
                )

                config.inputs["endstate_complex_traj"] = extract_complex.rv(0)
                config.inputs["endstate_complex_lastframe"] = extract_complex.rv(1)
                #             # run minimization at the end states for ligand system only
                minimization_ligand = Simulation(
                    executable=config.system_settings.executable,
                    mpi_command=config.system_settings.mpi_command,
                    num_cores=config.num_cores_per_system.ligand_ncores,
                    prmtop=config.endstate_files.ligand_parameter_filename,
                    incrd=config.endstate_files.ligand_coordinate_filename,
                    input_file=config.inputs["min_mdin"],
                    restraint_file=config.inputs["empty_restraint"],
                    directory_args={
                        "runtype": "minimization",
                        "filename": "min",
                        "topology": config.endstate_files.ligand_parameter_filename,
                        "topdir": config.system_settings.top_directory_path,
                    },
                    working_directory=config.system_settings.working_directory,
                    memory=config.system_settings.memory,
                    disk=config.system_settings.disk,
                )

                equilibrate_ligand = minimization_ligand.addFollowOn(
                    REMDSimulation(
                        executable=config.system_settings.executable,
                        mpi_command=config.system_settings.mpi_command,
                        num_cores=config.endstate_method.remd_args.nthreads_ligand,
                        ngroups=config.endstate_method.remd_args.ngroups,
                        prmtop=config.endstate_files.ligand_parameter_filename,
                        incrd=minimization_ligand.rv(0),
                        input_file=equil_mdins.rv(),
                        restraint_file=config.inputs["empty_restraint"],
                        runtype="equil",
                        directory_args={
                            "runtype": "equilibration",
                            "topology": config.endstate_files.ligand_parameter_filename,
                            "topdir": config.system_settings.top_directory_path,
                        },
                        working_directory=config.system_settings.working_directory,
                        memory=config.system_settings.memory,
                        disk=config.system_settings.disk,
                    )
                )

                remd_ligand = equilibrate_ligand.addFollowOn(
                    REMDSimulation(
                        executable=config.system_settings.executable,
                        mpi_command=config.system_settings.mpi_command,
                        num_cores=config.endstate_method.remd_args.nthreads_ligand,
                        prmtop=config.endstate_files.ligand_parameter_filename,
                        incrd=equilibrate_ligand.rv(0),
                        input_file=remd_mdins.rv(),
                        restraint_file=config.inputs["empty_restraint"],
                        runtype="remd",
                        ngroups=config.endstate_method.remd_args.ngroups,
                        directory_args={
                            "runtype": "remd",
                            "topology": config.endstate_files.ligand_parameter_filename,
                            "topdir": config.system_settings.top_directory_path,
                        },
                        working_directory=config.system_settings.working_directory,
                        memory=config.system_settings.memory,
                        disk=config.system_settings.disk,
                    )
                )
                # extact target temparture trajetory and last frame
                extract_ligand_traj = remd_ligand.addFollowOn(
                    ExtractTrajectories(
                        config.endstate_files.ligand_parameter_filename,
                        remd_ligand.rv(1),
                        config.endstate_method.remd_args.target_temperature,
                    )
                )
                # config.inputs["endstate_ligand_traj"] = extract_ligand_traj.rv(0)
                # config.inputs["endstate_ligand_lastframe"] = extract_ligand_traj.rv(1)

                if not workflow.ignore_receptor_endstate:
                    minimization_receptor = remd_mdins.addChild(
                        Simulation(
                            executable=config.system_settings.executable,
                            mpi_command=config.system_settings.mpi_command,
                            num_cores=config.num_cores_per_system.receptor_ncores,
                            prmtop=config.endstate_files.receptor_parameter_filename,
                            incrd=config.endstate_files.receptor_coordinate_filename,
                            input_file=config.inputs["min_mdin"],
                            restraint_file=config.inputs["empty_restraint"],
                            directory_args={
                                "runtype": "minimization",
                                "filename": "min",
                                "topology": config.endstate_files.receptor_parameter_filename,
                                "topdir": config.system_settings.top_directory_path,
                            },
                            working_directory=config.system_settings.working_directory,
                            memory=config.system_settings.memory,
                            disk=config.system_settings.disk,
                        )
                    )

                    equilibrate_receptor = minimization_receptor.addFollowOn(
                        REMDSimulation(
                            executable=config.system_settings.executable,
                            mpi_command=config.system_settings.mpi_command,
                            num_cores=config.endstate_method.remd_args.nthreads_receptor,
                            prmtop=config.endstate_files.receptor_parameter_filename,
                            incrd=minimization_receptor.rv(0),
                            input_file = equil_mdins.rv(),
                            restraint_file=config.inputs["empty_restraint"],
                            runtype="equil",
                            ngroups=config.endstate_method.remd_args.ngroups,
                            directory_args={
                                "runtype": "equilibration",
                                "topology": config.endstate_files.receptor_parameter_filename,
                                "topdir": config.system_settings.top_directory_path,
                            },
                            working_directory=config.system_settings.working_directory,
                            memory=config.system_settings.memory,
                            disk=config.system_settings.disk,
                        )
                    )

                    remd_receptor = equilibrate_receptor.addFollowOn(
                        REMDSimulation(
                            executable=config.system_settings.executable,
                            mpi_command=config.system_settings.mpi_command,
                            num_cores=config.endstate_method.remd_args.nthreads_receptor,
                            prmtop=config.endstate_files.receptor_parameter_filename,
                            incrd=equilibrate_receptor.rv(0),
                            input_file = remd_mdins.rv(),
                            restraint_file=config.inputs["empty_restraint"],
                            runtype="remd",
                            ngroups=config.endstate_method.remd_args.ngroups,
                            directory_args={
                                "runtype": "remd",
                                "topology": config.endstate_files.receptor_parameter_filename,
                                "topdir": config.system_settings.top_directory_name,
                            },
                            working_directory=config.system_settings.working_directory,
                            memory=config.system_settings.memory,
                            disk=config.system_settings.disk,
                        )
                    )
                    # extact target temparture trajetory and last frame
                    extract_receptor = remd_receptor.addFollowOn(
                        ExtractTrajectories(
                            config.endstate_files.receptor_parameter_filename,
                            remd_receptor.rv(1),
                            config.endstate_method.remd_args.target_temperature,
                        )
                    )
                    config.inputs["endstate_receptor_traj"] = extract_receptor.rv(0)
                    config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(
                        1
                    )

            # no endstate run
            elif config.endstate_method.endstate_method_type == 0:
                extract_complex_traj = endstate_method.addChild(
                    ExtractTrajectories(
                        config.endstate_files.complex_parameter_filename,
                        config.endstate_files.complex_coordinate_filename,
                    )
                )
                config.inputs["endstate_complex_traj"] = extract_complex_traj.rv(0)
                config.inputs["endstate_complex_lastframe"] = extract_complex_traj.rv(1)

                extract_ligand_traj = endstate_method.addChild(
                    ExtractTrajectories(
                        config.endstate_files.ligand_parameter_filename,
                        config.endstate_files.ligand_coordinate_filename,
                    )
                )
                config.inputs["endstate_ligand_traj"] = extract_ligand_traj.rv(0)
                config.inputs["endstate_ligand_lastframe"] = extract_ligand_traj.rv(1)

                extract_receptor_traj = endstate_method.addChild(
                    ExtractTrajectories(
                        config.endstate_files.receptor_parameter_filename,
                        config.endstate_files.receptor_coordinate_filename,
                    )
                )
                config.inputs["endstate_receptor_traj"] = extract_receptor_traj.rv(0)
                config.inputs["endstate_receptor_lastframe"] = extract_receptor_traj.rv(
                    1
                )

            # split the complex into host and substrate using the endstate lastframe
            split_job = endstate_method.addFollowOnJobFn(
                split_complex_system,
                config.endstate_files.complex_parameter_filename,
                config.inputs["endstate_complex_lastframe"],
                config.amber_masks.ligand_mask,
                config.amber_masks.receptor_mask,
            )

            config.inputs["ligand_endstate_frame"] = split_job.rv(1)
            config.inputs["receptor_endstate_frame"] = split_job.rv(0)

            restraints = split_job.addChild(RestraintMaker(config=config)).rv()

            #         # create independent simulation jobs for each system
            md_jobs = split_job.addFollowOnJobFn(initilized_jobs)
            completed_endstate_ligand = split_job.addFollowOnJobFn(initilized_jobs)

            # restraints = split_job.addChild(RestraintMaker(config=config))
            try:
                completed_endstate_ligand.addChild(minimization_ligand)
                ligand_simulation_jobs = extract_ligand_traj

            except NameError:
                ligand_simulation_jobs = split_job.addFollowOnJobFn(initilized_jobs)

            #         # create a flag for running postprocess endstate method
            complex_endstate_post_workflow = job.wrapJobFn(
                ddm_workflow,
                config,
                inptraj_id=[config.inputs["endstate_complex_traj"]],
                solute="complex",
                dirstuct_traj_args={
                    "traj_state_label": "endstate",
                    "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                    "state_level": 0.0,
                    "trajectory_restraint_conrest": 0.0,
                    "trajectory_restraint_orenrest": 0.0,
                    "filename": "state_8_endstate_postprocess",
                    "runtype": f"Running post process in endstate potential with trajectory: {config.inputs['endstate_complex_traj']}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )
            ligand_endstate_post_workflow = job.wrapJobFn(
                ddm_workflow,
                config,
                inptraj_id=[extract_ligand_traj.rv(0)],
                solute="ligand",
                dirstuct_traj_args={
                    "traj_state_label": "endstate",
                    "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                    "state_level": 0.0,
                    "filename": "state_2_endstate_postprocess",
                    "trajectory_restraint_conrest": 0.0,
                    "runtype": f"Running post process in endstate potential with trajectory: {extract_ligand_traj.rv(0)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            receptor_endstate_post_workflow = job.wrapJobFn(
                ddm_workflow,
                config,
                inptraj_id=[config.inputs["endstate_receptor_traj"]],
                solute="receptor",
                dirstuct_traj_args={
                    "traj_state_label": "endstate",
                    "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                    "state_level": 0.0,
                    "filename": "state_2_endstate_postprocess",
                    "trajectory_restraint_conrest": 0.0,
                    "runtype": f"Running post process in endstate potential with trajectory: {config.inputs['endstate_receptor_traj']}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            # attempt to run ligand_endstate simulation
            try:
                ligand_simulation_jobs.addChild(ligand_endstate_post_workflow)
                ligand_endstate_post_completed = (
                    ligand_endstate_post_workflow.addFollowOnJobFn(
                        run_post_process, ligand_endstate_post_workflow.rv()
                    )
                )
                ligand_df.append(ligand_endstate_post_completed.rv())
            except NameError:
                pass

            try:
                md_jobs.addChild(complex_endstate_post_workflow)

                complex_endstate_post_completed = (
                    complex_endstate_post_workflow.addFollowOnJobFn(
                        run_post_process, complex_endstate_post_workflow.rv()
                    )
                )
                complex_df.append(complex_endstate_post_completed.rv())

            except NameError:
                pass

            try:
                md_jobs.addChild(receptor_endstate_post_workflow)

                receptor_endstate_post_completed = (
                    receptor_endstate_post_workflow.addFollowOnJobFn(
                        run_post_process, receptor_endstate_post_workflow.rv()
                    )
                )
                receptor_df.append(receptor_endstate_post_completed.rv())
            except NameError:
                pass

    if workflow.end_state_postprocess:
        state_label = "endstate"
        restraint_file = config.inputs["empty_restraint"]
        dirstruct = "post_process_apo"
        if solute == "ligand":
            topology = config.endstate_files.ligand_parameter_filename
            if config.endstate_files.ligand_initial_coordinate is not None:
                coordinate = config.endstate_files.ligand_initial_coordinate
            else:
                coordinate = config.endstate_files.ligand_coordinate_filename
            num_cores = config.num_cores_per_system.ligand_ncores
        elif solute == "receptor":
            topology = config.endstate_files.receptor_parameter_filename
            if config.endstate_files.receptor_initial_coordinate is not None:
                coordinate = config.endstate_files.receptor_initial_coordinate
            else:
                coordinate = config.endstate_files.receptor_coordinate_filename
            num_cores = config.num_cores_per_system.receptor_ncores
        else:
            topology = config.endstate_files.complex_parameter_filename
            if config.endstate_files.complex_initial_coordinate is not None:
                coordinate = config.endstate_files.complex_initial_coordinate
            else:
                coordinate = config.endstate_files.complex_coordinate_filename
            num_cores = config.num_cores_per_system.complex_ncores
            dirstruct = "post_process_halo"
            state_label = "endstate"
            restraint_file = config.inputs["flat_bottom_restraint"]
        end_state_args = {
            "topology": topology,
            "state_label": state_label,
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "state_level": 0.0,
            "topdir": config.system_settings.top_directory_path,
        }
        end_state_args.update(dirstuct_traj_args)
        end_state_prod = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=num_cores,
            prmtop=topology,
            incrd=coordinate,
            input_file=config.inputs["post_mdin"],
            restraint_file=restraint_file,
            directory_args=end_state_args.copy(),
            dirstruct=dirstruct,
            inptraj=inptraj_id,
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        calc_list.append(end_state_prod)

    # define max conformational and restraint forces
    max_con_force = max(config.intermidate_args.conformational_restraints_forces)
    max_orien_force = max(config.intermidate_args.orientational_restriant_forces)

    max_con_exponent = float(
        max(config.intermidate_args.exponent_conformational_forces)
    )
    max_orien_exponent = float(
        max(config.intermidate_args.exponent_orientational_forces)
    )

    # turning off the solvent for ligand simulation, set max force of conformational restraints
    if workflow.remove_GB_solvent_ligand:
        no_solv_args = {
            "topology": config.endstate_files.ligand_parameter_filename,
            "state_label": "no_gb",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "state_level": 0.0,
            "filename": "state_4_prod",
            "runtype": f"Running production Simulation in state 4 (No GB). Max conformational force: {max_con_force} ",
            "topdir": config.system_settings.top_directory_path,
        }
        no_solv_args.update(dirstuct_traj_args)
        no_solv_ligand = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.ligand_ncores,
            prmtop=config.endstate_files.ligand_parameter_filename,
            incrd=config.inputs["ligand_endstate_frame"],
            input_file=mdin_no_solv,
            restraint_file=restraints,  # config.inputs[f"ligand_{max_con_force}_rst"]
            directory_args=no_solv_args.copy(),
            dirstruct=ligand_receptor_dirstruct,
            inptraj=inptraj_id,
            restraint_key=f"ligand_{max_con_force}_rst",
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )

        if not post_process:
            ligand_simulation_jobs.addChild(no_solv_ligand)
            post_process_ligand = no_solv_ligand.addFollowOnJobFn(
                ddm_workflow,
                config,
                inptraj_id=no_solv_ligand.rv(1),
                solute="ligand",
                dirstuct_traj_args={
                    "traj_state_label": "no_gb",
                    "trajectory_restraint_conrest": max_con_exponent,
                    "traj_igb": "igb_6",
                    "filename": "state_4_postprocess",
                    "runtype": f"Running post process with for ligand system with GB=6. inptraj: {no_solv_ligand.rv(1)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            ligand_df.append(
                post_process_ligand.addFollowOnJobFn(
                    run_post_process, post_process_ligand.rv()
                ).rv()
            )

        else:
            calc_list.append(no_solv_ligand)

    # set ligand overall charge to 0
    # if config.workflow.remove_ligand_charges:
    if workflow.remove_ligand_charges:
        ligand_no_charge_args = {
            "topology": config.endstate_files.ligand_parameter_filename,
            "state_label": "no_charges",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "filename": "state_5_prod",
            "runtype": "Production Simulation. In vacuum and ligand charges set to 0",
            "topdir": config.system_settings.top_directory_path,
        }
        ligand_no_charge_args.update(dirstuct_traj_args)

        ligand_no_charge = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.ligand_ncores,
            prmtop=config.inputs["ligand_no_charge_parm_ID"],
            incrd=config.inputs["ligand_endstate_frame"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=ligand_no_charge_args,
            dirstruct=ligand_receptor_dirstruct,
            inptraj=inptraj_id,
            restraint_key=f"ligand_{max_con_force}_rst",
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        if not post_process:
            ligand_simulation_jobs.addChild(ligand_no_charge)
            post_no_charge = ligand_no_charge.addFollowOnJobFn(
                ddm_workflow,
                config,
                inptraj_id=ligand_no_charge.rv(1),
                solute="ligand",
                dirstuct_traj_args={
                    "traj_state_label": "no_charges",
                    "trajectory_restraint_conrest": max_con_exponent,
                    "traj_igb": "igb_6",
                    "filename": "state_5_postprocess",
                    "runtype": f"Running post process with GB=6 and ligand_charge=0. inptraj: {ligand_no_charge.rv(1)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            ligand_df.append(
                post_no_charge.addFollowOnJobFn(
                    run_post_process, post_no_charge.rv()
                ).rv()
            )

        else:
            calc_list.append(ligand_no_charge)

    # Desolvation of receptor
    # if config.workflow.remove_GB_solvent_receptor:
    if workflow.remove_GB_solvent_receptor:
        no_solv_args_receptor = {
            "topology": config.endstate_files.receptor_parameter_filename,
            "state_label": "no_gb",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "filename": "state_4_prod",
            "runtype": "Running production simulation in state 4: Receptor only",
            "topdir": config.system_settings.top_directory_path,
        }

        no_solv_args_receptor.update(dirstuct_traj_args)

        no_solv_receptor = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.receptor_ncores,
            prmtop=config.endstate_files.receptor_parameter_filename,
            incrd=config.inputs["receptor_endstate_frame"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=no_solv_args_receptor,
            dirstruct=ligand_receptor_dirstruct,
            inptraj=inptraj_id,
            working_directory=config.system_settings.working_directory,
            restraint_key=f"receptor_{max_con_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        # receptor_jobs.addChild(no_solv_receptor)
        if not post_process:
            md_jobs.addChild(no_solv_receptor)
            post_no_solv_receptor = no_solv_receptor.addFollowOnJobFn(
                ddm_workflow,
                config,
                inptraj_id=no_solv_receptor.rv(1),
                solute="receptor",
                dirstuct_traj_args={
                    "traj_state_label": "no_gb",
                    "trajectory_restraint_conrest": max_con_exponent,
                    "traj_igb": "igb_6",
                    "filename": "state_4_postprocess",
                    "runtype": f"Running post process with GB=6 for receptor. inptraj: {no_solv_receptor.rv(1)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            receptor_df.append(
                post_no_solv_receptor.addFollowOnJobFn(
                    run_post_process, post_no_solv_receptor.rv()
                ).rv()
            )
        else:
            calc_list.append(no_solv_receptor)

    # Complex simulations
    # Exclusions turned on, no electrostatics, in gas phase
    # if config.workflow.complex_ligand_exclusions:
    if workflow.complex_ligand_exclusions:
        complex_ligand_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "no_interactions",
            "igb": "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7_prod",
            "topdir": config.system_settings.top_directory_path,
            "runtype": "Running production simulation in state 7: No iteractions with receptor/guest and in vacuum",
        }

        complex_ligand_exclusions_args.update(dirstuct_traj_args)

        complex_no_interactions = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.complex_ncores,
            prmtop=config.inputs["complex_no_ligand_interaction_ID"],
            incrd=config.inputs["endstate_complex_lastframe"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_ligand_exclusions_args,
            dirstruct=complex_dirstruct,
            inptraj=inptraj_id,
            working_directory=config.system_settings.working_directory,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        if not post_process:
            md_jobs.addChild(complex_no_interactions)

            complex_no_interactions_post = complex_no_interactions.addFollowOnJobFn(
                ddm_workflow,
                config,
                inptraj_id=complex_no_interactions.rv(1),
                solute="complex",
                dirstuct_traj_args={
                    "traj_state_label": "no_interactions",
                    "trajectory_restraint_conrest": max_con_exponent,
                    "trajectory_restraint_orenrest": max_orien_exponent,
                    "traj_igb": "igb_6",
                    "filename": "state_7_postprocess",
                    "runtype": f"Running post process in no interactions complex. inptraj: {complex_no_interactions.rv(1)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            complex_df.append(
                complex_no_interactions_post.addFollowOnJobFn(
                    run_post_process, complex_no_interactions_post.rv()
                ).rv()
            )
        else:
            job.log(f"update complex args {complex_ligand_exclusions_args}")
            calc_list.append(complex_no_interactions)

    # No electrostatics and in the gas phase
    if workflow.complex_turn_off_exclusions:
        complex_turn_off_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "interactions",
            "igb": "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7a_prod",
            "topdir": config.system_settings.top_directory_path,
            "runtype": "Running production simulation in state 7a: Turing back on interactions with recetor and guest in vacuum",
        }
        complex_turn_off_exclusions_args.update(dirstuct_traj_args)

        complex_no_electrostatics = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.complex_ncores,
            prmtop=config.inputs["complex_ligand_no_charge_ID"],
            incrd=config.inputs["endstate_complex_lastframe"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_turn_off_exclusions_args,
            dirstruct=complex_dirstruct,
            inptraj=inptraj_id,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        if not post_process:
            md_jobs.addChild(complex_no_electrostatics)

            complex_no_electrostatics_post = complex_no_electrostatics.addFollowOnJobFn(
                ddm_workflow,
                config,
                inptraj_id=complex_no_electrostatics.rv(1),
                solute="complex",
                dirstuct_traj_args={
                    "traj_state_label": "interactions",
                    "trajectory_restraint_conrest": max_con_exponent,
                    "trajectory_restraint_orenrest": max_orien_exponent,
                    "traj_igb": "igb_6",
                    "filename": "state_7a_postprocess",
                    "runtype": f"Running post process with interactions back. inptraj: {complex_no_electrostatics.rv(1)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            complex_df.append(
                complex_no_electrostatics_post.addFollowOnJobFn(
                    run_post_process, complex_no_electrostatics_post.rv()
                ).rv()
            )
        else:
            job.log(
                f"update complex_turn_off_exclusions_args {complex_turn_off_exclusions_args}"
            )
            calc_list.append(complex_no_electrostatics)

    # Turn ligand charges and in the gas phase

    if workflow.complex_turn_on_ligand_charges:
        complex_turn_on_ligand_charges_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "electrostatics",
            "igb": "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7b_prod",
            "runtype": "Running production simulation in state 7b: Turning back on ligand charges, still in vacuum",
            "topdir": config.system_settings.top_directory_path,
        }
        complex_turn_on_ligand_charges_args.update(dirstuct_traj_args)

        complex_turn_on_ligand_charges = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.complex_ncores,
            prmtop=config.endstate_files.complex_parameter_filename,
            incrd=config.inputs["endstate_complex_lastframe"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_turn_on_ligand_charges_args,
            dirstruct=complex_dirstruct,
            inptraj=inptraj_id,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        if not post_process:
            md_jobs.addChild(complex_turn_on_ligand_charges)
            complex_turn_on_ligand_charges_post = complex_turn_on_ligand_charges.addFollowOnJobFn(
                ddm_workflow,
                config,
                inptraj_id=complex_turn_on_ligand_charges.rv(1),
                solute="complex",
                dirstuct_traj_args={
                    "traj_state_label": "electrostatics",
                    "trajectory_restraint_conrest": max_con_exponent,
                    "trajectory_restraint_orenrest": max_orien_exponent,
                    "traj_igb": "igb_6",
                    "filename": "state_7b_postprocess",
                    "runtype": f"Running post process with full charge ligand atoms. inptraj: {complex_turn_on_ligand_charges.rv(1)}",
                }.copy(),
                post_process=True,
                restraints=restraints,
            )

            complex_df.append(
                complex_turn_on_ligand_charges_post.addFollowOnJobFn(
                    run_post_process, complex_turn_on_ligand_charges_post.rv()
                ).rv()
            )

        else:
            calc_list.append(complex_turn_on_ligand_charges)

    # Lambda window interate through conformational and orientational restraint forces
    for (con_force, orien_force) in zip(
        config.intermidate_args.conformational_restraints_forces,
        config.intermidate_args.orientational_restriant_forces,
    ):
        # add conformational restraints
        # if config.workflow.add_ligand_conformational_restraints:
        exponent_conformational = np.log2(con_force)
        exponent_orientational = np.log2(orien_force)
        if workflow.add_ligand_conformational_restraints:
            ligand_window_args = {
                "topology": config.endstate_files.ligand_parameter_filename,
                "state_label": "lambda_window",
                "conformational_restraint": exponent_conformational,
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, Conformational restraint: {con_force}",
                "topdir": config.system_settings.top_directory_path,
            }
            ligand_window_args.update(dirstuct_traj_args)

            ligand_windows = Simulation(
                executable=config.system_settings.executable,
                mpi_command=config.system_settings.mpi_command,
                num_cores=config.num_cores_per_system.ligand_ncores,
                prmtop=config.endstate_files.ligand_parameter_filename,
                incrd=config.inputs["ligand_endstate_frame"],
                input_file=default_mdin,
                restraint_file=restraints,
                directory_args=ligand_window_args,
                dirstruct=ligand_receptor_dirstruct,
                inptraj=inptraj_id,
                restraint_key=f"ligand_{con_force}_rst",
                working_directory=config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            if not post_process:
                ligand_simulation_jobs.addChild(ligand_windows)
                ligand_windows_post = ligand_windows.addFollowOnJobFn(
                    ddm_workflow,
                    config,
                    inptraj_id=ligand_windows.rv(1),
                    solute="ligand",
                    dirstuct_traj_args={
                        "traj_state_label": "lambda_window",
                        "trajectory_restraint_conrest": exponent_conformational,
                        "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                        "filename": "state_2_postprocess",
                        "runtype": f"Running post process in lambda window {con_force}: {ligand_windows.rv(1)}",
                    }.copy(),
                    post_process=True,
                    restraints=restraints,
                )

                ligand_df.append(
                    ligand_windows_post.addFollowOnJobFn(
                        run_post_process, ligand_windows_post.rv()
                    ).rv()
                )
            else:
                calc_list.append(ligand_windows)

        if workflow.add_receptor_conformational_restraints:

            receptor_window_args = {
                "topology": config.endstate_files.receptor_parameter_filename,
                "state_label": "lambda_window",
                "conformational_restraint": exponent_conformational,
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, conformational restraint: {con_force}",
                "topdir": config.system_settings.top_directory_path,
            }

            receptor_window_args.update(dirstuct_traj_args)

            receptor_windows = Simulation(
                executable= config.system_settings.executable,
                mpi_command= config.system_settings.mpi_command,
                num_cores= config.num_cores_per_system.receptor_ncores,
                prmtop= config.endstate_files.receptor_parameter_filename,
                incrd = config.inputs["receptor_endstate_frame"],
                input_file= default_mdin,
                restraint_file=restraints,
                directory_args=receptor_window_args,
                dirstruct=ligand_receptor_dirstruct,
                inptraj=inptraj_id,
                restraint_key=f"receptor_{con_force}_rst",
                working_directory = config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            if not post_process:
                md_jobs.addChild(receptor_windows)
                receptor_windows_post = receptor_windows.addFollowOnJobFn(
                    ddm_workflow,
                    config,
                    inptraj_id=receptor_windows.rv(1),
                    solute="receptor",
                    dirstuct_traj_args={
                        "traj_state_label": "lambda_window",
                        "trajectory_restraint_conrest": exponent_conformational,
                        "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                        "filename": "state_2_postprocess",
                        "runtype": f"Running post process with trajectory: {receptor_windows.rv(1)}",
                    }.copy(),
                    post_process=True,
                    restraints=restraints,
                )

                receptor_df.append(
                    (
                        receptor_windows_post.addFollowOnJobFn(
                            run_post_process, receptor_windows_post.rv()
                        )
                    ).rv()
                )

            else:
                calc_list.append(receptor_windows)

        # slowly remove conformational and orientational restraints
        # if config.workflow.complex_remove_restraint:
        if workflow.complex_remove_restraint:
            remove_restraints_args = {
                "topology": config.endstate_files.complex_parameter_filename,
                "state_label": "lambda_window",
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "conformational_restraint": exponent_conformational,
                "orientational_restraints": exponent_orientational,
                "filename": f"state_8_{con_force}_{orien_force}_prod",
                "runtype": f"Running restraint window. Conformational restraint: {con_force} and orientational restraint: {exponent_orientational}",
                "topdir": config.system_settings.top_directory_path,
            }

            remove_restraints_args.update(dirstuct_traj_args)

            remove_restraints = Simulation(
                executable= config.system_settings.executable,
                mpi_command= config.system_settings.mpi_command,
                num_cores= config.num_cores_per_system.complex_ncores,
                prmtop= config.endstate_files.complex_parameter_filename,
                incrd= config.inputs["endstate_complex_lastframe"],
                input_file= default_mdin,
                restraint_file=restraints,
                directory_args=remove_restraints_args,
                dirstruct=complex_dirstruct,
                inptraj=inptraj_id,
                restraint_key=f"complex_{con_force}_{orien_force}_rst",
                working_directory = config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            if not post_process:
                md_jobs.addChild(remove_restraints)
                remove_restraints_windows_post = remove_restraints.addFollowOnJobFn(
                    ddm_workflow,
                    config,
                    inptraj_id=remove_restraints.rv(1),
                    solute="complex",
                    dirstuct_traj_args={
                        "traj_state_label": "lambda_window",
                        "trajectory_restraint_conrest": exponent_conformational,
                        "trajectory_restraint_orenrest": exponent_orientational,
                        "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                        "filename": "state_8_postprocess",
                        "runtype": f"Running post process in lambda window {con_force}&{orien_force}. inptraj: {remove_restraints.rv(1)}",
                    }.copy(),
                    post_process=True,
                    restraints=restraints,
                )

                complex_df.append(
                    remove_restraints_windows_post.addFollowOnJobFn(
                        run_post_process, remove_restraints_windows_post.rv()
                    ).rv()
                )

            else:
                calc_list.append(remove_restraints)

    if post_process:
        return calc_list

    # run pyMBAR and consolidate output.
    if config.workflow.post_treatment:

        job.addFollowOnJobFn(
            consolidate_output,
            ligand_simulation_jobs.addFollowOn(
                PostTreatment(
                    ligand_df,
                    config.intermidate_args.temperature,
                    system="ligand",
                    max_conformation_force=max_con_exponent,
                )
            ).rv(),
            md_jobs.addFollowOn(
                PostTreatment(
                    receptor_df,
                    config.intermidate_args.temperature,
                    system="receptor",
                    max_conformation_force=max_con_exponent,
                )
            ).rv(),
            md_jobs.addFollowOn(
                PostTreatment(
                    complex_df,
                    config.intermidate_args.temperature,
                    system="complex",
                    max_conformation_force=max_con_exponent,
                    max_orientational_force=max_orien_exponent,
                )
            ).rv(),
            restraints,
        )


def run_post_process(job, sims: List[Simulation]):
    # ->list[pd.DataFrame]
    output_data = []
    for sim in sims:
        output_sims = job.addChild(sim)
        data_frame = output_sims.addFollowOnJobFn(
            create_mdout_dataframe, sim.directory_args, sim.dirstruct, sim.output_dir
        )
        output_data.append(data_frame.rv())

    return output_data


def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return


def main():

    parser = Job.Runner.getDefaultArgumentParser()
    parser.add_argument(
        "--config_file",
        nargs="*",
        type=str,
        required=True,
        help="configuartion file with input parameters",
    )
    parser.add_argument(
        "--ignore_receptor",
        action="store_true",
        help=" Receptor MD caluculations with not be performed.",
    )
    options = parser.parse_args()
    options.logLevel = "INFO"
    options.clean = "onSuccess"
    config_file = options.config_file[0]
    ignore_receptor = options.ignore_receptor

    start = time.perf_counter()

    try:
        with open(config_file) as f:
            config_file = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

    # setup configuration dataclass
    config = Config.from_config(config_file)

    # create top level directory to write output files
    if not os.path.exists(config.system_settings.top_directory_path):
        os.makedirs(config.system_settings.top_directory_path)

    complex_name = re.sub(
        r"\..*",
        "",
        os.path.basename(
            config_file["endstate_parameter_files"]["complex_parameter_filename"]
        ),
    )
    # create unique workflow log file
    job_number = 1
    while os.path.exists(
        f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    ):
        job_number += 1
    Path(
        f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    ).touch()

    options.logFile = f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    # setup toil workflow
    with Toil(options) as toil:

        config.workflow.ignore_receptor_endstate = ignore_receptor

        # log the performance time
        file_handler = logging.FileHandler(
            os.path.join(
                config.system_settings.top_directory_path,
                f"{complex_name}_workflow_performance.log",
            ),
            mode="w",
        )
        formatter = logging.Formatter(
            "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s"
        )
        file_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        if config.endstate_method.endstate_method_type != 0:
            config.get_receptor_ligand_topologies()
        else:
            config.endstate_files.get_inital_coordinate()

        if not toil.options.restart:
            config.endstate_files.toil_import_parmeters(toil=toil)

            if config.endstate_method.endstate_method_type == "remd":
                config.endstate_method.remd_args.toil_import_replica_mdin(toil=toil)

            if config.intermidate_args.guest_restraint_files is not None:
                config.intermidate_args.toil_import_user_restriants(toil=toil)

            config.inputs["min_mdin"] = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        os.path.dirname(os.path.realpath(__file__))
                        + "/templates/min.mdin"
                    )
                )
            )

            toil.start(Job.wrapJobFn(ddm_workflow, config))
            logger.info(
                f" Total workflow time: {time.perf_counter() - start} seconds\n"
            )
            logger.info(f"options.workDir {options.workDir}")

        else:
            toil.restart()


if __name__ == "__main__":
    main()

"""
config.endstate_files.complex_parameter_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(config.endstate_files.complex_parameter_filename)
                )
            )
            config.endstate_files.complex_coordinate_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(config.endstate_files.complex_coordinate_filename)
                )
            )
            config.endstate_files.ligand_coordinate_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(config.endstate_files.ligand_coordinate_filename)
                )
            )
            config.endstate_files.ligand_parameter_filename = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(config.endstate_files.ligand_parameter_filename)
                )
            )
"""
