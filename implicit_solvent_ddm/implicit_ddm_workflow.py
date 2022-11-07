# from implicit_solvent_ddm.remd import run_remd
import itertools
import logging
import os
import os.path
import queue
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from platform import system_alias

import numpy as np
import pandas as pd
import yaml
from toil.common import Toil
from toil.job import Job, JobFunctionWrappingJob

from implicit_solvent_ddm.alchemical import (get_intermidate_parameter_files,
                                             split_complex_system)
from implicit_solvent_ddm.config import Config, IntermidateStatesArgs
from implicit_solvent_ddm.mdin import get_mdins
from implicit_solvent_ddm.postTreatment import (PostTreatment,
                                                consolidate_output,
                                                create_mdout_dataframe)
from implicit_solvent_ddm.restraints import (FlatBottom, RestraintMaker,
                                             write_empty_restraint)
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.simulations import (ExtractTrajectories,
                                              REMDSimulation, Simulation)

working_directory = os.getcwd()


def ddm_workflow(job: JobFunctionWrappingJob, config: Config):

    intermidate_simulations = []
    queue_simulations = {} #{simulation: rank} rank=0 (highest priority)
    receptor_simuations = []
    complex_simuations = []
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

    if workflow.run_endstate_method:

        endstate_method = setup_inputs.addFollowOnJobFn(initilized_jobs)

        if config.endstate_method.endstate_method_type == "remd":

            # run endstate method for complex system
            minimization_complex = Simulation(
                config.system_settings.executable,
                config.system_settings.mpi_command,
                config.num_cores_per_system.complex_ncores,
                config.endstate_files.complex_parameter_filename,
                config.endstate_files.complex_coordinate_filename,
                config.inputs["min_mdin"],
                config.inputs["flat_bottom_restraint"],
                {
                    "runtype": "minimization",
                    "filename": "min",
                    "topology": config.endstate_files.complex_parameter_filename,
                },
                system_type="complex",
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            endstate_method.addChild(minimization_complex)

            # config.endstate_method.remd_args.nthreads
            equilibrate_complex = minimization_complex.addFollowOn(
                REMDSimulation(
                    config.system_settings.executable,
                    config.system_settings.mpi_command,
                    config.endstate_method.remd_args.nthreads_complex,
                    config.endstate_files.complex_parameter_filename,
                    minimization_complex.rv(0),
                    config.endstate_method.remd_args.equilibration_replica_mdins,
                    config.inputs["flat_bottom_restraint"],
                    "equil",
                    config.endstate_method.remd_args.ngroups,
                    {
                        "runtype": "equilibration",
                        "topology": config.endstate_files.complex_parameter_filename,
                    },
                    memory=config.system_settings.memory,
                    disk=config.system_settings.disk,
                )
            )

            remd_complex = equilibrate_complex.addFollowOn(
                REMDSimulation(
                    config.system_settings.executable,
                    config.system_settings.mpi_command,
                    config.endstate_method.remd_args.nthreads_complex,
                    config.endstate_files.complex_parameter_filename,
                    equilibrate_complex.rv(0),
                    config.endstate_method.remd_args.remd_mdins,
                    config.inputs["flat_bottom_restraint"],
                    "remd",
                    config.endstate_method.remd_args.ngroups,
                    directory_args={
                        "runtype": "remd",
                        "topology": config.endstate_files.complex_parameter_filename,
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

            minimization_ligand = minimization_complex.addFollowOn(
                Simulation(
                    config.system_settings.executable,
                    config.system_settings.mpi_command,
                    config.num_cores_per_system.ligand_ncores,
                    config.endstate_files.ligand_parameter_filename,
                    config.endstate_files.ligand_coordinate_filename,
                    config.inputs["min_mdin"],
                    config.inputs["empty_restraint"],
                    {
                        "runtype": "minimization",
                        "filename": "min",
                        "topology": config.endstate_files.ligand_parameter_filename,
                    },
                    system_type="ligand",
                    memory=config.system_settings.memory,
                    disk=config.system_settings.disk,
                )
            )

            equilibrate_ligand = minimization_ligand.addFollowOn(
                REMDSimulation(
                    config.system_settings.executable,
                    config.system_settings.mpi_command,
                    config.endstate_method.remd_args.nthreads_ligand,
                    config.endstate_files.ligand_parameter_filename,
                    minimization_ligand.rv(0),
                    config.endstate_method.remd_args.equilibration_replica_mdins,
                    config.inputs["empty_restraint"],
                    "equil",
                    config.endstate_method.remd_args.ngroups,
                    directory_args={
                        "runtype": "equilibration",
                        "topology": config.endstate_files.ligand_parameter_filename,
                    },
                    memory=config.system_settings.memory,
                    disk=config.system_settings.disk,
                )
            )

            remd_ligand = equilibrate_ligand.addFollowOn(
                REMDSimulation(
                    config.system_settings.executable,
                    config.system_settings.mpi_command,
                    config.endstate_method.remd_args.nthreads_ligand,
                    config.endstate_files.ligand_parameter_filename,
                    equilibrate_ligand.rv(0),
                    config.endstate_method.remd_args.remd_mdins,
                    config.inputs["empty_restraint"],
                    "remd",
                    config.endstate_method.remd_args.ngroups,
                    directory_args={
                        "runtype": "remd",
                        "topology": config.endstate_files.ligand_parameter_filename,
                    },
                    memory=config.system_settings.memory,
                    disk=config.system_settings.disk,
                )
            )

            # extact target temparture trajetory and last frame
            extract_ligand = remd_ligand.addFollowOn(
                ExtractTrajectories(
                    config.endstate_files.ligand_parameter_filename,
                    remd_ligand.rv(1),
                    config.endstate_method.remd_args.target_temperature,
                )
            )
            config.inputs["endstate_ligand_traj"] = extract_ligand.rv(0)
            config.inputs["endstate_ligand_lastframe"] = extract_ligand.rv(1)

            if not workflow.ignore_receptor_endstate:
                minimization_receptor = endstate_method.addChild(
                    Simulation(
                        config.system_settings.executable,
                        config.system_settings.mpi_command,
                        config.num_cores_per_system.receptor_ncores,
                        config.endstate_files.receptor_parameter_filename,
                        config.endstate_files.receptor_coordinate_filename,
                        config.inputs["min_mdin"],
                        config.inputs["empty_restraint"],
                        {
                            "runtype": "minimization",
                            "filename": "min",
                            "topology": config.endstate_files.receptor_parameter_filename,
                        },
                        system_type="receptor",
                        memory=config.system_settings.memory,
                        disk=config.system_settings.disk,
                    )
                )

                equilibrate_receptor = minimization_receptor.addFollowOn(
                    REMDSimulation(
                        config.system_settings.executable,
                        config.system_settings.mpi_command,
                        config.endstate_method.remd_args.nthreads_receptor,
                        config.endstate_files.receptor_parameter_filename,
                        minimization_receptor.rv(0),
                        config.endstate_method.remd_args.equilibration_replica_mdins,
                        config.inputs["empty_restraint"],
                        "equil",
                        config.endstate_method.remd_args.ngroups,
                        {
                            "runtype": "equilibration",
                            "topology": config.endstate_files.receptor_parameter_filename,
                        },
                        memory=config.system_settings.memory,
                        disk=config.system_settings.disk,
                    )
                )

                remd_receptor = equilibrate_receptor.addFollowOn(
                    REMDSimulation(
                        config.system_settings.executable,
                        config.system_settings.mpi_command,
                        config.endstate_method.remd_args.nthreads_receptor,
                        config.endstate_files.receptor_parameter_filename,
                        equilibrate_receptor.rv(0),
                        config.endstate_method.remd_args.remd_mdins,
                        config.inputs["empty_restraint"],
                        "remd",
                        config.endstate_method.remd_args.ngroups,
                        directory_args={
                            "runtype": "remd",
                            "topology": config.endstate_files.receptor_parameter_filename,
                        },
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

    # fill in orientational and conformational forces within templates
    else:
        endstate_method = setup_inputs.addFollowOn(
            ExtractTrajectories(
                config.endstate_files.complex_parameter_filename,
                config.endstate_files.complex_coordinate_filename,
            )
        )
        config.inputs["endstate_complex_traj"] = endstate_method.rv(0)
        config.inputs["endstate_complex_lastframe"] = endstate_method.rv(1)
        ligand_extract = endstate_method.addChild(
            ExtractTrajectories(
                config.endstate_files.ligand_parameter_filename,
                config.endstate_files.ligand_coordinate_filename,
            )
        )
        config.inputs["endstate_ligand_traj"] = ligand_extract.rv(0)
        config.inputs["endstate_ligand_lastframe"] = ligand_extract.rv(1)

        receptor_extract = endstate_method.addChild(
            ExtractTrajectories(
                config.endstate_files.receptor_parameter_filename,
                config.endstate_files.receptor_coordinate_filename,
            )
        )
        config.inputs["endstate_receptor_traj"] = receptor_extract.rv(0)
        config.inputs["endstate_receptor_lastframe"] = receptor_extract.rv(1)

    # split the coordinates from endstate simulations complex
    split_job = endstate_method.addFollowOnJobFn(
        split_complex_system,
        config.endstate_files.complex_parameter_filename,
        config.inputs["endstate_complex_lastframe"],
        config.amber_masks.ligand_mask,
        config.amber_masks.receptor_mask,
    )

    config.inputs["ligand_endstate_frame"] = split_job.rv(1)
    config.inputs["receptor_endstate_frame"] = split_job.rv(0)

    # create orientational and conformational restraints

    restraints = split_job.addChild(RestraintMaker(config=config)).rv()

    md_jobs = split_job.addFollowOnJobFn(initilized_jobs)

    if workflow.end_state_postprocess:
        complex_coordinate = config.endstate_files.complex_coordinate_filename

        if config.endstate_files.complex_initial_coordinate is not None:
            complex_coordinate = config.endstate_files.complex_initial_coordinate

        complex_endstate_dirstruct = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "endstate",
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "state_level": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
            "filename": "state_8_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
            "trajectory_restraint_orenrest": 0.0,
        }
        complex_endstate_postprocess = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.complex_ncores,
            config.endstate_files.complex_parameter_filename,
            complex_coordinate,
            config.inputs["post_mdin"],
            config.inputs["flat_bottom_restraint"],
            directory_args=complex_endstate_dirstruct,
            system_type="complex",
            dirstruct="post_process_halo",
            inptraj=[config.inputs["endstate_complex_traj"]],
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[complex_endstate_postprocess] = 0
        #intermidate_simulationsappend(complex_endstate_postprocess)

        ligand_coordiante = config.endstate_files.ligand_coordinate_filename

        if config.endstate_files.ligand_initial_coordinate is not None:
            ligand_coordiante = config.endstate_files.ligand_initial_coordinate

        ligand_endstate_dirstruct = {
            "topology": config.endstate_files.ligand_parameter_filename,
            "state_label": "endstate",
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "state_level": 0.0,
            "runtype": "remd",
            "state_level": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
            "filename": "state_2_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
        }

        ligand_endstate_postprocess = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.ligand_ncores,
            config.endstate_files.ligand_parameter_filename,
            ligand_coordiante,
            config.inputs["post_mdin"],
            config.inputs["empty_restraint"],
            directory_args=ligand_endstate_dirstruct,
            system_type="ligand",
            dirstruct="post_process_apo",
            inptraj=[config.inputs["endstate_ligand_traj"]],
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[ligand_endstate_postprocess] = 2
        ##intermidate_simulationsappend(ligand_endstate_postprocess)

        receptor_coordiate = config.endstate_files.receptor_coordinate_filename
        if config.endstate_files.receptor_initial_coordinate is not None:
            receptor_coordiate = config.endstate_files.receptor_initial_coordinate

        job.log(f"RECEPTOR coordinate {receptor_coordiate}")
        receptor_endstate_dirstruct = {
            "topology": config.endstate_files.receptor_parameter_filename,
            "state_label": "endstate",
            "igb": f"igb_{config.intermidate_args.igb_solvent}",
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0,
            "state_level": 0.0,
            "runtype": "remd",
            "state_level": 0.0,
            "runtype": "remd",
            "traj_state_label": "endstate",
            "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
            "filename": "state_2_endstate_postprocess",
            "trajectory_restraint_conrest": 0.0,
        }
        receptor_endstate_postprocess = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.receptor_ncores,
            config.endstate_files.receptor_parameter_filename,
            receptor_coordiate,
            config.inputs["post_mdin"],
            config.inputs["empty_restraint"],
            directory_args=receptor_endstate_dirstruct,
            system_type="receptor",
            dirstruct="post_process_apo",
            inptraj=[config.inputs["endstate_receptor_traj"]],
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[receptor_endstate_postprocess] = 1
        #intermidate_simulationsappend(receptor_endstate_postprocess)
    
    # turning off the solvent for ligand simulation, set max force of conformational restraints
    # define max conformational and restraint forces
    max_con_force = max(config.intermidate_args.conformational_restraints_forces)
    max_orien_force = max(config.intermidate_args.orientational_restriant_forces)

    max_con_exponent = float(
        max(config.intermidate_args.exponent_conformational_forces)
    )
    max_orien_exponent = float(
        max(config.intermidate_args.exponent_orientational_forces)
    )

    if workflow.remove_GB_solvent_ligand:
        no_solv_args = {
            "topology": config.endstate_files.ligand_parameter_filename,
            "state_label": "no_gb",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "state_level": 0.0,
            "filename": "state_4_prod",
            "runtype": f"Running production Simulation in state 4 (No GB). Max conformational force: {max_con_force} ",
        }
        no_solv_ligand = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.ligand_ncores,
            config.endstate_files.ligand_parameter_filename,
            config.inputs["ligand_endstate_frame"],
            input_file=mdin_no_solv,
            restraint_file=restraints,
            directory_args=no_solv_args.copy(),
            system_type="ligand",
            dirstruct=ligand_receptor_dirstruct,
            restraint_key=f"ligand_{max_con_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[no_solv_ligand] = 2
        #intermidate_simulationsappend(no_solv_ligand)

    if workflow.remove_ligand_charges:
        ligand_no_charge_args = {
            "topology": config.endstate_files.ligand_parameter_filename,
            "state_label": "no_charges",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "filename": "state_5_prod",
            "runtype": "Production Simulation. In vacuum and ligand charges set to 0",
        }

        ligand_no_charge = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.ligand_ncores,
            config.inputs["ligand_no_charge_parm_ID"],
            config.inputs["ligand_endstate_frame"],
            mdin_no_solv,
            restraints,
            directory_args=ligand_no_charge_args,
            system_type="ligand",
            dirstruct=ligand_receptor_dirstruct,
            restraint_key=f"ligand_{max_con_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        # ligand_jobs.addChild(ligand_no_charge)
        queue_simulations[ligand_no_charge] = 2
        #intermidate_simulationsappend(ligand_no_charge)

    if workflow.remove_GB_solvent_receptor:
        no_solv_args_receptor = {
            "topology": config.endstate_files.receptor_parameter_filename,
            "state_label": "no_gb",
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "filename": "state_4_prod",
            "runtype": "Running production simulation in state 4: Receptor only",
        }
        no_solv_receptor = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.receptor_ncores,
            config.endstate_files.receptor_parameter_filename,
            config.inputs["receptor_endstate_frame"],
            mdin_no_solv,
            restraint_file=restraints,
            directory_args=no_solv_args_receptor,
            system_type="receptor",
            dirstruct=ligand_receptor_dirstruct,
            restraint_key=f"receptor_{max_con_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[no_solv_receptor] = 1
        #intermidate_simulationsappend(no_solv_receptor)

    # Exclusions turned on, no electrostatics, in gas phase
    if workflow.complex_ligand_exclusions:
        complex_ligand_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "no_interactions",
            "igb": "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7_prod",
            "runtype": "Running production simulation in state 7: No iteractions with receptor/guest and in vacuum",
        }
        complex_no_interactions = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.complex_ncores,
            config.inputs["complex_no_ligand_interaction_ID"],
            config.inputs["endstate_complex_lastframe"],
            mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_ligand_exclusions_args,
            system_type="complex",
            dirstruct=complex_dirstruct,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[complex_no_interactions] = 0
        #intermidate_simulationsappend(complex_no_interactions)

    # No electrostatics and in the gas phase
    if workflow.complex_turn_off_exclusions:
        complex_turn_off_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "interactions",
            "igb": "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7a_prod",
            "runtype": "Running production simulation in state 7a: Turing back on interactions with recetor and guest in vacuum",
        }
        complex_no_electrostatics = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.complex_ncores,
            config.inputs["complex_ligand_no_charge_ID"],
            config.inputs["endstate_complex_lastframe"],
            mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_turn_off_exclusions_args,
            system_type="complex",
            dirstruct=complex_dirstruct,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[complex_no_electrostatics] =  0
        #intermidate_simulationsappend(complex_no_electrostatics)

    # Turn on ligand charges/vacuum
    if workflow.complex_turn_on_ligand_charges:
        complex_turn_on_ligand_charges_args = {
            "topology": config.endstate_files.complex_parameter_filename,
            "state_label": "electrostatics",
            "igb": "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7b_prod",
            "runtype": "Running production simulation in state 7b: Turning back on ligand charges, still in vacuum",
        }
        complex_turn_on_ligand_charges = Simulation(
            config.system_settings.executable,
            config.system_settings.mpi_command,
            config.num_cores_per_system.complex_ncores,
            config.endstate_files.complex_parameter_filename,
            config.inputs["endstate_complex_lastframe"],
            mdin_no_solv,
            restraint_file=restraints,
            directory_args=complex_turn_on_ligand_charges_args,
            system_type="complex",
            dirstruct=complex_dirstruct,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            memory=config.system_settings.memory,
            disk=config.system_settings.disk,
        )
        queue_simulations[complex_turn_on_ligand_charges] =  0
        #intermidate_simulationsappend(complex_turn_on_ligand_charges)
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
            }

            ligand_windows = Simulation(
                config.system_settings.executable,
                config.system_settings.mpi_command,
                config.num_cores_per_system.ligand_ncores,
                config.endstate_files.ligand_parameter_filename,
                config.inputs["ligand_endstate_frame"],
                default_mdin,
                restraints,
                directory_args=ligand_window_args,
                system_type="ligand",
                dirstruct=ligand_receptor_dirstruct,
                restraint_key=f"ligand_{con_force}_rst",
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            queue_simulations[ligand_windows] = 2
            #intermidate_simulationsappend(ligand_windows)

        if workflow.add_receptor_conformational_restraints:

            receptor_window_args = {
                "topology": config.endstate_files.receptor_parameter_filename,
                "state_label": "lambda_window",
                "conformational_restraint": exponent_conformational,
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, conformational restraint: {con_force}",
            }

            receptor_windows = Simulation(
                config.system_settings.executable,
                config.system_settings.mpi_command,
                config.num_cores_per_system.receptor_ncores,
                config.endstate_files.receptor_parameter_filename,
                config.inputs["receptor_endstate_frame"],
                default_mdin,
                restraint_file=restraints,
                directory_args=receptor_window_args,
                system_type="receptor",
                dirstruct=ligand_receptor_dirstruct,
                restraint_key=f"receptor_{con_force}_rst",
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            queue_simulations[receptor_windows] = 1
            #intermidate_simulationsappend(receptor_windows)

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
            }
            remove_restraints = Simulation(
                config.system_settings.executable,
                config.system_settings.mpi_command,
                config.num_cores_per_system.complex_ncores,
                config.endstate_files.complex_parameter_filename,
                config.inputs["endstate_complex_lastframe"],
                default_mdin,
                restraint_file=restraints,
                directory_args=remove_restraints_args,
                system_type="complex",
                dirstruct=complex_dirstruct,
                restraint_key=f"complex_{con_force}_{orien_force}_rst",
                memory=config.system_settings.memory,
                disk=config.system_settings.disk,
            )
            queue_simulations[remove_restraints] = 0
            #intermidate_simulationsappend(remove_restraints)

    intermidate_runner = md_jobs.addChild(
        IntermidateRunner(
            queue_simulations,
            restraints,
            post_process_mdin=config.inputs["post_nosolv_mdin"],
            post_process_distruct="post_process_apo",
        )
    )
    
    if workflow.post_treatment:
        job.addFollowOnJobFn(
            consolidate_output,
            intermidate_runner.addFollowOn(
                PostTreatment(
                    intermidate_runner.rv(0),
                    config.intermidate_args.temperature,
                    system="ligand",
                    max_conformation_force=max_con_exponent,
                )
            ).rv(),
            intermidate_runner.addFollowOn(
                PostTreatment(
                    intermidate_runner.rv(1),
                    config.intermidate_args.temperature,
                    system="receptor",
                    max_conformation_force=max_con_exponent,
                )
            ).rv(),
            intermidate_runner.addFollowOn(
                PostTreatment(
                    intermidate_runner.rv(2),
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
    work_dir = os.getcwd()

    try:
        with open(config_file) as f:
            config_file = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

 
    work_dir  = working_directory
    if not os.path.exists(
        os.path.join(work_dir, "mdgb/structs/ligand")):
            os.makedirs(
                os.path.join(
                    work_dir, "mdgb/structs/ligand"
                )
            )
    if not os.path.exists(
        os.path.join(work_dir, "mdgb/structs/receptor")):
            os.makedirs(
                os.path.join(
                    work_dir, "mdgb/structs/receptor"
                )
            )
    
    complex_name = re.sub(r"\..*", "", os.path.basename(config_file["endstate_parameter_files"]["complex_parameter_filename"]))
    # create a log file

    # log the performance time
    file_handler = logging.FileHandler(
        os.path.join(work_dir, f"mdgb/{complex_name}_workflow_performance.log"),
        mode="w",
    )
    formatter = logging.Formatter(
        "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s"
    )
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    job_number = 1
    while os.path.exists(f"mdgb/log_job_{job_number:03}.txt"):
        job_number += 1
    Path(f"mdgb/{complex_name}_job{job_number:03}.txt").touch()

    options.logFile = f"mdgb/{complex_name}_job{job_number:03}.txt"

    with Toil(options) as toil:
        # setup config
        config = Config.from_config(config_file)

        config.workflow.ignore_receptor_endstate = ignore_receptor

        if config.endstate_method.endstate_method_type != 0:
            config.get_receptor_ligand_topologies()
        else:
            config.endstate_files.get_inital_coordinate()

        if not toil.options.restart:
            config.endstate_files.toil_import_parmeters(toil=toil)

            if config.endstate_method.endstate_method_type == "remd":
                config.endstate_method.remd_args.toil_import_replica_mdins(toil=toil)

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
            logger.info(f" Total workflow time: {time.perf_counter() - start} seconds\n")
            
        else:
            toil.restart()


if __name__ == "__main__":
    main()
