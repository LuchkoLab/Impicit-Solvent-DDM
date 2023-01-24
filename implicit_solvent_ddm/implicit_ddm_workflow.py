# from implicit_solvent_ddm.remd import run_remd
import itertools
import logging
import os
import os.path
import re
import time
from pathlib import Path

import numpy as np
import yaml
from toil.common import Toil
from toil.job import Job, JobFunctionWrappingJob

from implicit_solvent_ddm.alchemical import (alter_topology,
                                             split_complex_system)
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.mdin import generate_replica_mdin, get_mdins
from implicit_solvent_ddm.postTreatment import (PostTreatment,
                                                consolidate_output)
from implicit_solvent_ddm.restraints import (FlatBottom, RestraintMaker,
                                             write_empty_restraint)
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.simulations import (ExtractTrajectories,
                                              REMDSimulation, Simulation)

working_directory = os.getcwd()


def ddm_workflow(job: JobFunctionWrappingJob, config: Config):

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
    ligand_simulations = []
    receptor_simuations = []
    complex_simuations = []
    workflow = config.workflow
    ligand_receptor_dirstruct = "dirstruct_apo"
    complex_dirstruct = "dirstruct_halo"

    # setup_inputs = job.wrapJobFn(
    #     get_intermidate_parameter_files,
    #     config.endstate_files.complex_parameter_filename,
    #     config.endstate_files.complex_coordinate_filename,
    #     config.amber_masks.ligand_mask,
    #     config.amber_masks.receptor_mask,
    # )
    # job.addChild(setup_inputs)

    # #set parameter files
    # config.inputs["ligand_no_charge_parm_ID"] = setup_inputs.rv(0)
    # config.inputs["complex_ligand_no_charge_ID"] = setup_inputs.rv(1)
    # config.inputs["complex_no_ligand_interaction_ID"] = setup_inputs.rv(2)

    # set intermidate mdin files
    mdins = job.addChildJobFn(
        get_mdins, config.intermidate_args.mdin_intermidate_config
    )
    # fill in intermidate mdin
    config.inputs["default_mdin"] = mdins.rv(0)
    config.inputs["no_solvent_mdin"] = mdins.rv(1)
    config.inputs["post_mdin"] = mdins.rv(2)
    config.inputs["post_nosolv_mdin"] = mdins.rv(3)
    default_mdin = mdins.rv(0)
    mdin_no_solv = mdins.rv(1)
    # write empty restraint.RST
    empty_restraint = mdins.addChildJobFn(write_empty_restraint)
    config.inputs["empty_restraint"] = empty_restraint.rv()

    # flat bottom restraints potential restraints
    flat_bottom_template = mdins.addChild(FlatBottom(config=config))

    config.inputs["flat_bottom_restraint"] = flat_bottom_template.rv()

    if workflow.run_endstate_method:

        endstate_method = mdins.addFollowOnJobFn(initilized_jobs)

        if config.endstate_method.endstate_method_type == "remd":

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
                    working_directory=config.system_settings.working_directory,
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

            # run minimization at the end states for ligand system only
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
                minimization_receptor = remd_complex.addChild(
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
                        input_file=equil_mdins.rv(),
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
                        input_file=remd_mdins.rv(),
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
                config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(1)
            # use loaded receptor completed trajectory
            else:
                extract_receptor = remd_complex.addChild(
                    ExtractTrajectories(
                        config.endstate_files.receptor_parameter_filename,
                        config.endstate_files.receptor_coordinate_filename,
                    )
                )
                config.inputs["endstate_receptor_traj"] = extract_receptor.rv(0)
                config.inputs["endstate_receptor_lastframe"] = extract_receptor.rv(1)
                config.endstate_files.receptor_coordinate_filename = (
                    extract_receptor.rv(1)
                )

    # no endstate run
    else:
        endstate_method = mdins.addFollowOn(
            ExtractTrajectories(
                config.endstate_files.complex_parameter_filename,
                config.endstate_files.complex_coordinate_filename,
            )
        )
        config.inputs["endstate_complex_traj"] = endstate_method.rv(0)
        config.inputs["endstate_complex_lastframe"] = endstate_method.rv(1)

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
        config.inputs["endstate_receptor_lastframe"] = extract_receptor_traj.rv(1)

    if config.workflow.vina_dock:
        config.inputs[
            "endstate_complex_lastframe"
        ] = config.endstate_files.complex_coordinate_filename
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

    # create independent simulation jobs for each system
    complex_host_jobs = split_job.addFollowOnJobFn(initilized_jobs)
    completed_endstate_ligand = split_job.addFollowOnJobFn(initilized_jobs)

    try:
        completed_endstate_ligand.addChild(minimization_ligand)
        ligand_simulation_jobs = extract_ligand_traj

    except NameError:
        ligand_simulation_jobs = split_job.addFollowOnJobFn(initilized_jobs)

    if workflow.end_state_postprocess:
        complex_coordinate = config.endstate_files.complex_coordinate_filename

        if config.endstate_files.complex_initial_coordinate is not None:
            complex_coordinate = config.endstate_files.complex_initial_coordinate

        complex_endstate_dirstruct = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "endstate",
            "traj_charge": 1.0,
            "charge": 1.0,
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "igb_value": config.intermidate_args.igb_solvent,
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
            "topdir": config.system_settings.top_directory_path,
            "filename": "state_8_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
            "trajectory_restraint_orenrest": 0.0,
        }
        complex_endstate_postprocess = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.complex_ncores,
            prmtop=config.endstate_files.complex_parameter_filename,
            incrd=complex_coordinate,
            input_file=config.inputs["post_mdin"],
            restraint_file=config.inputs["flat_bottom_restraint"],
            directory_args=complex_endstate_dirstruct,
            system_type="complex",
            dirstruct="post_process_halo",
            inptraj=[config.inputs["endstate_complex_traj"]],
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        complex_simuations.append(complex_endstate_postprocess)

        ligand_coordiante = config.endstate_files.ligand_coordinate_filename

        if config.endstate_files.ligand_initial_coordinate is not None:
            ligand_coordiante = config.endstate_files.ligand_initial_coordinate

        ligand_endstate_dirstruct = {
            "topology": config.endstate_files.ligand_parameter_filename,
            "state_label": "endstate",
            "charge": 1.0,
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "igb_value": config.intermidate_args.igb_solvent,
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_charge": 1.0,
            "topdir": config.system_settings.top_directory_path,
            "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
            "filename": "state_2_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
        }

        ligand_endstate_postprocess = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.ligand_ncores,
            prmtop=config.endstate_files.ligand_parameter_filename,
            incrd=ligand_coordiante,
            input_file=config.inputs["post_mdin"],
            restraint_file=config.inputs["empty_restraint"],
            directory_args=ligand_endstate_dirstruct,
            system_type="ligand",
            dirstruct="post_process_apo",
            inptraj=[extract_ligand_traj.rv(0)],
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        ligand_simulations.append(ligand_endstate_postprocess)

        receptor_coordiate = config.endstate_files.receptor_coordinate_filename
        if config.endstate_files.receptor_initial_coordinate is not None:
            receptor_coordiate = config.endstate_files.receptor_initial_coordinate

        receptor_endstate_dirstruct = {
            "topology": config.endstate_files.receptor_parameter_filename,
            "state_label": "endstate",
            "charge": 1.0,
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "igb_value": config.intermidate_args.igb_solvent,
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_charge": 1.0,
            "topdir": config.system_settings.top_directory_path,
            "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
            "filename": "state_2_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
        }
        receptor_endstate_postprocess = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.receptor_ncores,
            prmtop=config.endstate_files.receptor_parameter_filename,
            incrd=receptor_coordiate,
            input_file=config.inputs["post_mdin"],
            restraint_file=config.inputs["empty_restraint"],
            directory_args=receptor_endstate_dirstruct,
            system_type="receptor",
            dirstruct="post_process_apo",
            inptraj=[config.inputs["endstate_receptor_traj"]],
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        receptor_simuations.append(receptor_endstate_postprocess)

    # define max conformational and restraint forces
    max_con_force = max(config.intermidate_args.conformational_restraints_forces)
    max_orien_force = max(config.intermidate_args.orientational_restriant_forces)

    max_con_exponent = float(
        max(config.intermidate_args.exponent_conformational_forces)
    )
    max_orien_exponent = float(
        max(config.intermidate_args.exponent_orientational_forces)
    )
    # interpolate charges of the ligand
    for index, charge in enumerate(config.intermidate_args.charges_lambda_window):  # type: ignore
        # IGB=6
        if workflow.remove_GB_solvent_ligand:
            no_solv_args = {
                "topology": config.endstate_files.ligand_parameter_filename,
                "state_label": "electrostatics",
                "conformational_restraint": max_con_exponent,
                "igb": "igb_6",
                "charge": charge,
                "igb_value": 6,
                "filename": "state_4_prod",
                "runtype": f"Running production Simulation in state 4 (No GB). Max conformational force: {max_con_force} ",
                "topdir": config.system_settings.top_directory_path,
            }
            no_solv_ligand = Simulation(
                executable=config.system_settings.executable,
                mpi_command=config.system_settings.mpi_command,
                num_cores=config.num_cores_per_system.ligand_ncores,
                prmtop=ligand_simulation_jobs.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.ligand_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.ligand_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=charge,
                ).rv(),
                incrd=config.inputs["ligand_endstate_frame"],
                input_file=mdin_no_solv,
                restraint_file=restraints,  # config.inputs[f"ligand_{max_con_force}_rst"]
                directory_args=no_solv_args.copy(),
                system_type="ligand",
                dirstruct=ligand_receptor_dirstruct,
                restraint_key=f"ligand_{max_con_force}_rst",
                working_directory=config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            ligand_simulations.append(no_solv_ligand)

        # alter the ligand charges in the complex
        if workflow.complex_turn_on_ligand_charges:
            complex_turn_on_ligand_charges_args = {
                "topology": config.endstate_files.complex_parameter_filename,
                "state_label": "electrostatics",
                "charge": charge,
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "igb_value": config.intermidate_args.igb_solvent,
                "conformational_restraint": max_con_exponent,
                "orientational_restraints": max_orien_exponent,
                "filename": "state_7b_prod",
                "runtype": "Running production simulation in state 7b: Turning back on ligand charges, still in vacuum",
                "topdir": config.system_settings.top_directory_path,
            }
            complex_turn_on_ligand_charges = Simulation(
                executable=config.system_settings.executable,
                mpi_command=config.system_settings.mpi_command,
                num_cores=config.num_cores_per_system.complex_ncores,
                prmtop=complex_host_jobs.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.complex_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=charge,
                ).rv(),
                incrd=config.inputs["endstate_complex_lastframe"],
                input_file=default_mdin,
                restraint_file=restraints,
                directory_args=complex_turn_on_ligand_charges_args,
                system_type="complex",
                dirstruct=complex_dirstruct,
                restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
                working_directory=config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            complex_simuations.append(complex_turn_on_ligand_charges)

        # if workflow.remove_ligand_charges:
        #     ligand_no_charge_args = {
        #         "topology": config.endstate_files.ligand_parameter_filename,
        #         "state_label": "no_charges",
        #         "conformational_restraint": max_con_exponent,
        #         "igb": "igb_6",
        #         "igb_value": 6,
        #         "charge": 1.0,
        #         "filename": "state_5_prod",
        #         "runtype": "Production Simulation. In vacuum and ligand charges set to 0",
        #         "topdir": config.system_settings.top_directory_path,
        #     }

        # ligand_no_charge = Simulation(
        #     executable=config.system_settings.executable,
        #     mpi_command=config.system_settings.mpi_command,
        #     num_cores=config.num_cores_per_system.ligand_ncores,
        #     prmtop=config.inputs["ligand_no_charge_parm_ID"],
        #     incrd=config.inputs["ligand_endstate_frame"],
        #     input_file=mdin_no_solv,
        #     restraint_file=restraints,
        #     directory_args=ligand_no_charge_args,
        #     system_type="ligand",
        #     dirstruct=ligand_receptor_dirstruct,
        #     restraint_key=f"ligand_{max_con_force}_rst",
        #     working_directory=config.system_settings.working_directory,
        #     memory=config.system_settings.memory,
        #     disk=config.system_settings.disk,
        # )
        # ligand_simulations.append(ligand_no_charge)

    # Desolvation of receptor
    if workflow.remove_GB_solvent_receptor:
        no_solv_args_receptor = {
            "topology": config.endstate_files.receptor_parameter_filename,
            "state_label": "no_gb",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "igb_value": 6,
            "charge": 1.0,
            "filename": "state_4_prod",
            "runtype": "Running production simulation in state 4: Receptor only",
            "topdir": config.system_settings.top_directory_path,
        }
        no_solv_receptor = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.receptor_ncores,
            prmtop=config.endstate_files.receptor_parameter_filename,
            incrd=config.inputs["receptor_endstate_frame"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=no_solv_args_receptor,
            system_type="receptor",
            dirstruct=ligand_receptor_dirstruct,
            working_directory=config.system_settings.working_directory,
            restraint_key=f"receptor_{max_con_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        receptor_simuations.append(no_solv_receptor)

    # Exclusions turned on, no electrostatics, in gas phase
    if workflow.complex_ligand_exclusions:
        complex_ligand_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "no_interactions",
            "igb": "igb_6",
            "igb_value": 6,
            "charge": 0.0,
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7_prod",
            "topdir": config.system_settings.top_directory_path,
            "runtype": "Running production simulation in state 7: No iteractions with receptor/guest and in vacuum",
        }
        complex_no_interactions = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.complex_ncores,
            prmtop=complex_host_jobs.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
                exculsions=True,
            ).rv(),
            incrd=config.inputs["endstate_complex_lastframe"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_ligand_exclusions_args,
            system_type="complex",
            dirstruct=complex_dirstruct,
            working_directory=config.system_settings.working_directory,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        complex_simuations.append(complex_no_interactions)

    # Turn on LJ potentials bewteen ligand and host. (IGB=6 and ligand charge = 0)
    if workflow.complex_turn_off_exclusions:
        complex_turn_off_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "interactions",
            "igb": "igb_6",
            "igb_value": 6,
            "charge": 0,
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7a_prod",
            "topdir": config.system_settings.top_directory_path,
            "runtype": "Running production simulation in state 7a: Turing back on interactions with recetor and guest in vacuum",
        }
        complex_no_electrostatics = Simulation(
            executable=config.system_settings.executable,
            mpi_command=config.system_settings.mpi_command,
            num_cores=config.num_cores_per_system.complex_ncores,
            prmtop=complex_host_jobs.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
            ).rv(),
            incrd=config.inputs["endstate_complex_lastframe"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_turn_off_exclusions_args,
            system_type="complex",
            dirstruct=complex_dirstruct,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            working_directory=config.system_settings.working_directory,
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        complex_simuations.append(complex_no_electrostatics)

    # # Turn on GB enviroment but ligand charges are off
    # if workflow.complex_turn_on_GB_enviroment:
    #     complex_GB_enviroment= {
    #         "topology": config.endstate_files.complex_parameter_filename,
    #         "state_label": f"GB_enviroment",
    #         "igb": f"igb_{config.intermidate_args.igb_solvent}",
    #         "igb_value": config.intermidate_args.igb_solvent,
    #         "conformational_restraint": max_con_exponent,
    #         "orientational_restraints": max_orien_exponent,
    #         "filename": "state_7b_prod",
    #         "runtype": "Running production simulation in state 7b: Turning back on GB enviroment",
    #         "topdir": config.system_settings.top_directory_path,
    #     }
    #     complex_turn_on_GB_env = Simulation(
    #         executable=config.system_settings.executable,
    #         mpi_command=config.system_settings.mpi_command,
    #         num_cores=config.num_cores_per_system.complex_ncores,
    #         prmtop=config.inputs["complex_ligand_no_charge_ID"],
    #         incrd=config.inputs["endstate_complex_lastframe"],
    #         input_file=default_mdin,
    #         restraint_file=restraints,
    #         directory_args=complex_GB_enviroment,
    #         system_type="complex",
    #         dirstruct=complex_dirstruct,
    #         restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
    #         working_directory=config.system_settings.working_directory,
    #         memory=config.system_settings.memory,
    #         disk=config.system_settings.disk,
    #     )
    #     complex_simuations.append(complex_turn_on_GB_env)

    # if workflow.complex_turn_on_ligand_charges:
    #     complex_turn_on_ligand_charges_args = {
    #         "topology": config.endstate_files.complex_parameter_filename,
    #         "state_label": "electrostatics",
    #         "igb": "igb_6",
    #         "igb_value": 6,
    #         "conformational_restraint": max_con_exponent,
    #         "orientational_restraints": max_orien_exponent,
    #         "filename": "state_7b_prod",
    #         "runtype": "Running production simulation in state 7b: Turning back on ligand charges, still in vacuum",
    #         "topdir": config.system_settings.top_directory_path,
    #     }
    #     complex_turn_on_ligand_charges = Simulation(
    #         executable=config.system_settings.executable,
    #         mpi_command=config.system_settings.mpi_command,
    #         num_cores=config.num_cores_per_system.complex_ncores,
    #         prmtop=config.endstate_files.complex_parameter_filename,
    #         incrd=config.inputs["endstate_complex_lastframe"],
    #         input_file=mdin_no_solv,
    #         restraint_file=restraints,
    #         directory_args=complex_turn_on_ligand_charges_args,
    #         system_type="complex",
    #         dirstruct=complex_dirstruct,
    #         restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
    #         working_directory=config.system_settings.working_directory,
    #         memory=config.system_settings.memory,
    #         disk=config.system_settings.disk,
    #     )
    #     complex_simuations.append(complex_turn_on_ligand_charges)

    # intermidate_simulationsappend(complex_turn_on_ligand_charges)
    # lambda window interate through conformational and orientational restraint forces
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
                "charge": 1.0,
                "igb_value": config.intermidate_args.igb_solvent,
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, Conformational restraint: {con_force}",
                "topdir": config.system_settings.top_directory_path,
            }

            ligand_windows = Simulation(
                executable=config.system_settings.executable,
                mpi_command=config.system_settings.mpi_command,
                num_cores=config.num_cores_per_system.ligand_ncores,
                prmtop=config.endstate_files.ligand_parameter_filename,
                incrd=config.inputs["ligand_endstate_frame"],
                input_file=default_mdin,
                restraint_file=restraints,
                directory_args=ligand_window_args,
                system_type="ligand",
                dirstruct=ligand_receptor_dirstruct,
                restraint_key=f"ligand_{con_force}_rst",
                working_directory=config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            ligand_simulations.append(ligand_windows)

        if workflow.add_receptor_conformational_restraints:

            receptor_window_args = {
                "topology": config.endstate_files.receptor_parameter_filename,
                "state_label": "lambda_window",
                "charge": 1.0,
                "conformational_restraint": exponent_conformational,
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "igb_value": config.intermidate_args.igb_solvent,
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, conformational restraint: {con_force}",
                "topdir": config.system_settings.top_directory_path,
            }

            receptor_windows = Simulation(
                executable=config.system_settings.executable,
                mpi_command=config.system_settings.mpi_command,
                num_cores=config.num_cores_per_system.receptor_ncores,
                prmtop=config.endstate_files.receptor_parameter_filename,
                incrd=config.inputs["receptor_endstate_frame"],
                input_file=default_mdin,
                restraint_file=restraints,
                directory_args=receptor_window_args,
                system_type="receptor",
                dirstruct=ligand_receptor_dirstruct,
                restraint_key=f"receptor_{con_force}_rst",
                working_directory=config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            receptor_simuations.append(receptor_windows)

        # slowly remove conformational and orientational restraints
        # turn back on ligand charges
        # if config.workflow.complex_remove_restraint:
        if workflow.complex_remove_restraint and max_con_force != con_force:
            remove_restraints_args = {
                "topology": config.endstate_files.complex_parameter_filename,
                "state_label": "lambda_window",
                "charge": 1.0,
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "igb_value": config.intermidate_args.igb_solvent,
                "conformational_restraint": exponent_conformational,
                "orientational_restraints": exponent_orientational,
                "filename": f"state_8_{con_force}_{orien_force}_prod",
                "runtype": f"Running restraint window. Conformational restraint: {con_force} and orientational restraint: {exponent_orientational}",
                "topdir": config.system_settings.top_directory_path,
            }
            remove_restraints = Simulation(
                executable=config.system_settings.executable,
                mpi_command=config.system_settings.mpi_command,
                num_cores=config.num_cores_per_system.complex_ncores,
                prmtop=config.endstate_files.complex_parameter_filename,
                incrd=config.inputs["endstate_complex_lastframe"],
                input_file=default_mdin,
                restraint_file=restraints,
                directory_args=remove_restraints_args,
                system_type="complex",
                dirstruct=complex_dirstruct,
                restraint_key=f"complex_{con_force}_{orien_force}_rst",
                working_directory=config.system_settings.working_directory,
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            complex_simuations.append(remove_restraints)

    intermidate_complex = complex_host_jobs.addFollowOn(
        IntermidateRunner(
            complex_simuations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
        )
    )

    intermidate_receptor = complex_host_jobs.addFollowOn(
        IntermidateRunner(
            receptor_simuations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
        )
    )
    intermidate_ligand = ligand_simulation_jobs.addFollowOn(
        IntermidateRunner(
            ligand_simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
        )
    )
    if config.workflow.post_treatment:
        job.addFollowOnJobFn(
            consolidate_output,
            intermidate_ligand.addFollowOn(
                PostTreatment(
                    intermidate_ligand.rv(),
                    config.intermidate_args.temperature,
                    system="ligand",
                    max_conformation_force=max_con_exponent,
                )
            ).rv(),
            intermidate_receptor.addFollowOn(
                PostTreatment(
                    intermidate_receptor.rv(),
                    config.intermidate_args.temperature,
                    system="receptor",
                    max_conformation_force=max_con_exponent,
                )
            ).rv(),
            intermidate_complex.addFollowOn(
                PostTreatment(
                    intermidate_complex.rv(),
                    config.intermidate_args.temperature,
                    system="complex",
                    max_conformation_force=max_con_exponent,
                    max_orientational_force=max_orien_exponent,
                )
            ).rv(),
            restraints,
        )


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

        # if config.endstate_method.endstate_method_type != 0:
        #     config.get_receptor_ligand_topologies()
        # else:
        #     config.endstate_files.get_inital_coordinate()

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
