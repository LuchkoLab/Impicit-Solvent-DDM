# from implicit_solvent_ddm.remd import run_remd
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

from implicit_solvent_ddm.adaptive_restraints import (
    adaptive_lambda_windows,
    run_exponential_averaging,
)
from implicit_solvent_ddm.alchemical import alter_topology, split_complex_system
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.run_endstate import (
    run_remd,
    run_basic_md,
    user_defined_endstate,
)
from implicit_solvent_ddm.setup_simulations import SimulationSetup
from implicit_solvent_ddm.mdin import get_mdins, generate_extdiel_mdin
from implicit_solvent_ddm.postTreatment import ConsolidateData
from implicit_solvent_ddm.restraints import (
    BoreschRestraints,
    FlatBottom,
    RestraintMaker,
    write_empty_restraint,
)
from implicit_solvent_ddm.runner import IntermidateRunner

logger = logging.getLogger(__name__)
working_directory = os.getcwd()


def ddm_workflow(
    job: JobFunctionWrappingJob, config: Config
) -> tuple[Config, Config, Config]:
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

    workflow = config.workflow
    # set intermediate mdin files
    mdins = job.addChildJobFn(get_mdins, config.intermediate_args.mdin_intermediate_file)
    # fill in intermediate mdin
    config.inputs["default_mdin"] = mdins.rv(0)
    config.inputs["no_solvent_mdin"] = mdins.rv(1)
    config.inputs["post_mdin"] = mdins.rv(2)
    config.inputs["post_nosolv_mdin"] = mdins.rv(3)

    # write empty restraint.RST
    empty_restraint = mdins.addChildJobFn(write_empty_restraint)
    config.inputs["empty_restraint"] = empty_restraint.rv()

    # flat bottom restraints potential restraints
    flat_bottom_template = mdins.addChild(FlatBottom(config=config))

    config.inputs["flat_bottom_restraint"] = flat_bottom_template.rv(0)

    if workflow.run_endstate_method:
        if config.endstate_method.endstate_method_type == "remd":
            endstate_job = mdins.addFollowOnJobFn(run_remd, config)
            # use loaded receptor completed trajectory
        # run basic MD
        else:
            endstate_job = mdins.addFollowOnJobFn(run_basic_md, config)
    # no endstate run
    else:
        endstate_job = mdins.addFollowOnJobFn(user_defined_endstate, config)

    # split the complex into host and substrate using the endstate lastframe
    split_job = endstate_job.addFollowOnJobFn(
        split_complex_system,
        config.endstate_files.complex_parameter_filename,
        endstate_job.rv(0),
        config.amber_masks.ligand_mask,
        config.amber_masks.receptor_mask,
    )

    # create parent job for flat bottom contribution
    flat_bottom_contribution = split_job.addFollowOnJobFn(initilized_jobs, message="Preparing Flat bottom contribution")
    # generate Boresch orientational restraints
    boresh_restraints = split_job.addChild(
        BoreschRestraints(
            complex_prmtop=config.endstate_files.complex_parameter_filename,
            complex_coordinate=endstate_job.rv(0),
            restraint_type=config.intermediate_args.restraint_type,
            ligand_mask=config.amber_masks.ligand_mask,
            receptor_mask=config.amber_masks.receptor_mask,
            K_r=config.intermediate_args.max_conformational_restraint,
            K_thetaA=config.intermediate_args.max_orientational_restraint,
            K_thetaB=config.intermediate_args.max_orientational_restraint,
            K_phiA=config.intermediate_args.max_orientational_restraint,
            K_phiB=config.intermediate_args.max_orientational_restraint,
            K_phiC=config.intermediate_args.max_orientational_restraint,
        )
    )
    # manage restraints
    restraints = boresh_restraints.addChild(
        RestraintMaker(
            config=config,
            complex_binding_mode=endstate_job.rv(0),
            boresch_restraints=boresh_restraints.rv(),
            flat_bottom=flat_bottom_template.rv(1),
        )
    ).rv()

    runner_jobs = boresh_restraints.addFollowOnJobFn(initilized_jobs, message="Setup MD simulations")

    # setup system dependent cycle steps
    complex_simulations = SimulationSetup(
        config=config,
        system_type="complex",
        endstate_traj=endstate_job.rv(1),
        binding_mode=endstate_job.rv(0),
        restraints=restraints,
    )

    receptor_simulations = SimulationSetup(
        config=config,
        system_type="receptor",
        restraints=restraints,
        binding_mode=split_job.rv(0),
        endstate_traj=endstate_job.rv(2),
    )
    ligand_simulations = SimulationSetup(
        config=config,
        system_type="ligand",
        restraints=restraints,
        binding_mode=split_job.rv(1),
        endstate_traj=endstate_job.rv(3),
    )
    flat_bottom_setup = SimulationSetup(
        config=config,
        system_type="complex",
        endstate_traj=endstate_job.rv(1),
        binding_mode=endstate_job.rv(0),
        restraints=restraints,
    )
    if workflow.end_state_postprocess:
        # Setup EXP ->  flat bottom contribution
        flat_bottom_setup.setup_post_endstate_simulation(flat_bottom=True)
        flat_bottom_setup.setup_post_endstate_simulation()

        # Setup endstate post-process analysis at the endstates
        complex_simulations.setup_post_endstate_simulation(flat_bottom=True)
        receptor_simulations.setup_post_endstate_simulation()
        ligand_simulations.setup_post_endstate_simulation()

    # define max conformational and restraint forces
    max_con_force = max(config.intermediate_args.conformational_restraints_forces)
    max_orien_force = max(config.intermediate_args.orientational_restraint_forces)
    max_con_exponent = float(
        round(max(config.intermediate_args.exponent_conformational_forces), 3)
    )
    max_orien_exponent = float(
        round(max(config.intermediate_args.exponent_orientational_forces), 3)
    )
    # interpolate charges of the ligand
    for index, charge in enumerate(config.intermediate_args.charges_lambda_window):  # type: ignore
        # IGB=6
        # scale ligand charges
        ligand_simulations.setup_ligand_charge_simulation(
            prmtop=runner_jobs.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.ligand_parameter_filename,
                solute_amber_coordinate=config.endstate_files.ligand_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=charge,
            ).rv(),
            charge=charge,
            restraint_key=f"ligand_{max_con_force}_rst",
        )
        # scale ligand charges within the complex
        complex_simulations.setup_ligand_charge_simulation(
            prmtop=runner_jobs.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=charge,
            ).rv(),
            charge=charge,
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
        )

    # desolvation of receptor only
    if workflow.remove_GB_solvent_receptor:
        receptor_simulations.setup_remove_gb_solvent_simulation(
            restraint_key=f"receptor_{max_con_force}_rst",
            prmtop=config.endstate_files.receptor_parameter_filename,
        )

    # exclusions turned on, no electrostatics and in gas phase
    if workflow.complex_ligand_exclusions:
        complex_simulations.setup_remove_gb_solvent_simulation(
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            prmtop=runner_jobs.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
                exculsions=True,
            ).rv(),
        )

    # Turn on LJ potentials bewteen ligand and host. (IGB=6 and ligand charge = 0)
    if workflow.complex_turn_off_exclusions:
        complex_simulations.setup_lj_interations_simulation(
            restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
            prmtop=runner_jobs.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
            ).rv(),
        )
    # turn on GB solvent
    if workflow.gb_extdiel_windows:
        # create complex with ligand electrostatics = 0
        complex_ligand_no_charge = runner_jobs.addChildJobFn(
            alter_topology,
            solute_amber_parm=config.endstate_files.complex_parameter_filename,
            solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
            ligand_mask=config.amber_masks.ligand_mask,
            receptor_mask=config.amber_masks.receptor_mask,
            set_charge=0.0,
        ).rv()
        # interpolate GB external dielectric constant
        for dielectric in config.intermediate_args.gb_extdiel_windows:
            complex_simulations.setup_gb_external_dielectric(
                restraint_key=f"complex_{max_con_force}_{max_orien_force}_rst",
                prmtop=complex_ligand_no_charge,
                extdiel=dielectric,
                mdin=runner_jobs.addChildJobFn(
                    generate_extdiel_mdin,
                    user_mdin_ID=config.intermediate_args.mdin_intermediate_file,
                    gb_extdiel=dielectric,
                ).rv(),
            )

    # lambda window interate through conformational and orientational restraint forces
    for con_force, orien_force in zip(
        config.intermediate_args.conformational_restraints_forces,
        config.intermediate_args.orientational_restraint_forces,
    ):
        exponent_conformational = round(np.log2(con_force), 3)
        exponent_orientational = round(np.log2(orien_force), 3)

        # add conformational restraints
        if config.workflow.add_ligand_conformational_restraints:
            ligand_simulations.setup_apply_restraint_windows(
                restraint_key=f"ligand_{con_force}_rst",
                exponent_conformational=exponent_conformational,
            )

        if workflow.add_receptor_conformational_restraints:
            receptor_simulations.setup_apply_restraint_windows(
                restraint_key=f"receptor_{con_force}_rst",
                exponent_conformational=exponent_conformational,
            )

        # slowly remove conformational and orientational restraints
        # turn back on ligand charges
        if workflow.complex_remove_restraint and max_con_force != con_force:
            complex_simulations.setup_apply_restraint_windows(
                restraint_key=f"complex_{con_force}_{orien_force}_rst",
                exponent_conformational=exponent_conformational,
                exponent_orientational=exponent_orientational,
            )

    # Flat bottom contribution
    flat_bottom_analysis = flat_bottom_contribution.addChild(
        IntermidateRunner(
            flat_bottom_setup.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=workflow.post_analysis_only,
            config=config,
        )
    )
    # perform exponntial averaging -> flat bottom restraint contribution
    flat_bottom_exp = flat_bottom_analysis.addFollowOnJobFn(
        run_exponential_averaging,
        system_runner=flat_bottom_analysis.rv(),
        temperature=config.intermediate_args.temperature,
    )

    # place the binding models within the config.inputs dictionary
    updated_config = runner_jobs.addChildJobFn(
        update_config, config, endstate_job.rv(0), split_job.rv(0), split_job.rv(1)
    ).rv()
    md_jobs = runner_jobs.addFollowOnJobFn(initilized_jobs, message="Running MD simulations")
    # Run all initial left and right side MD cycle steps.
    intermediate_complex = md_jobs.addChild(
        IntermidateRunner(
            complex_simulations.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=False,
            config=updated_config,
        )
    )
    intermediate_receptor = md_jobs.addChild(
        IntermidateRunner(
            receptor_simulations.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=False,
            config=updated_config,
        )
    )

    intermediate_ligand = md_jobs.addChild(
        IntermidateRunner(
            ligand_simulations.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=False,
            config=updated_config,
        )
    )
    # Job holder for MD jobs completed
    MD_jobs_completed = md_jobs.addFollowOnJobFn(initilized_jobs, message="Completed MD simulations-moving to post-processing")

    # Once MD jobs are completed, perform post-analysis
    post_analyses_intermediate_complex = MD_jobs_completed.addChild(
        IntermidateRunner(
            complex_simulations.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_halo",
            post_only=True,
            config=updated_config,
        )
    )
    post_analyses_intermediate_receptor = MD_jobs_completed.addChild(
        IntermidateRunner(
            receptor_simulations.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=True,
            config=updated_config,
        )
    )
    post_analyses_intermediate_ligand = MD_jobs_completed.addChild(
        IntermidateRunner(
            ligand_simulations.simulations,
            restraints,
            post_process_no_solv_mdin=config.inputs["post_nosolv_mdin"],
            post_process_mdin=config.inputs["post_mdin"],
            post_process_distruct="post_process_apo",
            post_only=True,
            config=updated_config,
        )
    )
    # Improve any poor space phase overlap between adjacent windows
    # adaptive process for restraints and ligand charge scaling.
    if workflow.run_adaptive_windows:

        # first scale gb external dielectric
        complex_adaptive_gb_extdiel_job = post_analyses_intermediate_complex.addFollowOnJobFn(
            adaptive_lambda_windows,
            post_analyses_intermediate_complex.rv(),
            updated_config,
            "complex",
            gb_scaling=True,
        )
        # then scale restraints
        complex_adaptive_charge_job = complex_adaptive_gb_extdiel_job.addFollowOnJobFn(
            adaptive_lambda_windows,
            complex_adaptive_gb_extdiel_job.rv(2),
            complex_adaptive_gb_extdiel_job.rv(1),
            "complex",
            charge_scaling=True,
        )
        # finally scale restraints
        complex_adaptive_restraints_job = complex_adaptive_charge_job.addFollowOnJobFn(
            adaptive_lambda_windows,
            complex_adaptive_charge_job.rv(2),
            complex_adaptive_charge_job.rv(1),
            "complex",
            restraints_scaling=True,
        )
        # adaptive process for restraints and ligand charge scaling for ligand system steps.
        ligand_adaptive_restraints_job = post_analyses_intermediate_ligand.addFollowOnJobFn(
            adaptive_lambda_windows,
            post_analyses_intermediate_ligand.rv(),
            updated_config,
            "ligand",
            restraints_scaling=True,
        )
        ligand_adaptive_charges_job = ligand_adaptive_restraints_job.addFollowOnJobFn(
            adaptive_lambda_windows,
            ligand_adaptive_restraints_job.rv(2),
            ligand_adaptive_restraints_job.rv(1),
            "ligand",
            charge_scaling=True,
        )
        # adaptive process for restraints only.
        receptor_adaptive_job = post_analyses_intermediate_receptor.addFollowOnJobFn(
            adaptive_lambda_windows,
            post_analyses_intermediate_receptor.rv(),
            updated_config,
            "receptor",
            restraints_scaling=True,
        )
        # Once reached, we are done just export results :)
        if workflow.consolidate_output:
            job.addFollowOn(
                ConsolidateData(
                    complex_adative_run=complex_adaptive_restraints_job.rv(0),
                    ligand_adaptive_run=ligand_adaptive_charges_job.rv(0),
                    receptor_adaptive_run=receptor_adaptive_job.rv(0),
                    flat_botton_run=flat_bottom_exp.rv(),
                    temperature=config.intermediate_args.temperature,
                    max_conformation_force=max_con_exponent,
                    max_orientational_force=max_orien_exponent,
                    boresch_df=restraints,
                    complex_filename=config.endstate_files.complex_parameter_filename,
                    ligand_filename=config.endstate_files.ligand_parameter_filename,
                    receptor_filename=config.endstate_files.receptor_parameter_filename,
                    working_path=config.system_settings.cache_directory_output,
                    plot_overlap_matrix=config.workflow.plot_overlap_matrix,
                )
            )
        return (
            complex_adaptive_gb_extdiel_job.rv(1),
            ligand_adaptive_charges_job.rv(1),
            receptor_adaptive_job.rv(1),
        )

    return config


def update_config(
    job,
    config: Config,
    complex_binding_mode,
    receptor_binding_mode,
    ligand_binding_mode,
):
    config.inputs["endstate_complex_lastframe"] = complex_binding_mode
    config.inputs["receptor_endstate_frame"] = receptor_binding_mode
    config.inputs["ligand_endstate_frame"] = ligand_binding_mode

    return config


def initilized_jobs(job, message:str):
    "Place holder to schedule jobs for MD and post-processing"
    job.fileStore.logToMaster(message)
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
    options.logLevel = options.clean
    options.clean = "onSuccess"
    config_file = options.config_file[0]
    ignore_receptor = options.ignore_receptor

    start = time.perf_counter()
    logger.info("LOADED CONFIG??? ")
    try:
        with open(config_file) as f:
            config_file = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

    # setup configuration dataclass
    logger.info(f'[CONFIG] Loading in config')
    config = Config.from_config(config_file)
    logger.info(f'[CONFIG] Finished loading in config')

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
                f"{complex_name}_{job_number}_workflow_performance.log",
            ),
            mode="w",
        )
        formatter = logging.Formatter(
            "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s"
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        # if config.endstate_method.endstate_method_type != 0:
        #     config.get_receptor_ligand_topologies()
        # else:
        #     config.endstate_files.get_inital_coordinate()

        if not toil.options.restart:
            config.endstate_files.toil_import_parameters(toil=toil)
            config.intermediate_args.toil_import_user_mdin(toil=toil)
            # if the user doesn't provide there own endstate simulation
            if config.endstate_method.endstate_method_type != 0:
                # import files for remd
                if config.endstate_method.endstate_method_type == "remd":
                    config.endstate_method.remd_args.toil_import_replica_mdin(toil=toil)
                # import files for basic MD
                else:
                    config.endstate_method.basic_md_args.toil_import_basic_mdin(
                        toil=toil
                    )

            if config.intermediate_args.guest_restraint_files is not None:
                config.intermediate_args.toil_import_user_restraints(toil=toil)

            config.inputs["min_mdin"] = str(
                toil.import_file(
                    "file://"
                    + os.path.abspath(
                        os.path.dirname(os.path.realpath(__file__))
                        + "/templates/min.mdin"
                    )
                )
            )
            logger.info(f"config.endstate_files.complex_parameter_filename: {config.endstate_files.complex_parameter_filename}")
            update_config = toil.start(Job.wrapJobFn(ddm_workflow, config))
            logger.info(
                f" Total workflow time: {time.perf_counter() - start} seconds\n"
            )
            # logger.info(f"options.workDir {options.workDir}")
            # if len(
            #     update_config.intermediate_args.exponent_conformational_forces
            # ) != len(config.intermediate_args.exponent_conformational_forces):
            #     logger.info(
            #         f"""Restraints windows were added to config file: \n
            #                     original conformational & orientational windows: {config.intermediate_args.exponent_conformational_forces} & {config.intermediate_args.exponent_orientational_forces}\n
            #                     updated conformational & orientational windows: {update_config.intermediate_args.exponent_conformational_forces} & {update_config.intermediate_args.exponent_orientational_forces}\n
            #                 """
            #     )

        else:
            toil.restart()


if __name__ == "__main__":
    logger.info("EXECUTE MAIN()")
    main()
