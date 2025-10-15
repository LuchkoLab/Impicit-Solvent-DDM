"""
Double Decoupling Method (DDM) workflow for free energy calculations.
"""

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

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.workflow_phases import (
    setup_workflow_components,
    run_endstate_simulations,
    decompose_system_and_generate_restraints,
    setup_intermediate_simulations,
    run_post_analysis_intermediate_simulations,
    compute_free_energy_and_consolidate,
    run_intermediate_simulations,       
    initilized_jobs,
)

logger = logging.getLogger(__name__)
working_directory = os.getcwd()


def ddm_workflow(
    job: JobFunctionWrappingJob, config: Config
) -> tuple[Config, Config, Config]:
    """
    Double Decoupling Method (DDM) workflow for free energy calculations.

    This workflow performs a complete DDM calculation including:
    1. Setup and preparation of simulation components
    2. Endstate simulations (complex, receptor, ligand)
    3. System decomposition and restraint generation
    4. Intermediate state simulations with alchemical transformations
    5. Post-processing and analysis

    Parameters
    ----------
    job : JobFunctionWrappingJob
        Toil job wrapper for workflow execution
    config : Config
        Configuration object containing all simulation parameters

    Returns
    -------
    tuple[Config, Config, Config]
        Updated configuration objects for complex, ligand, and receptor systems
    """

    # Phase 1: Setup and Preparation
    setup_jobs = job.addChildJobFn(setup_workflow_components, config)
    updated_config = setup_jobs.rv()

    setup_jobs.addFollowOnJobFn(
        initilized_jobs, 
        message="✓ Phase 1 Complete: Workflow components setup finished"
    )
    setup_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 2: Endstate Simulations"
    )
    
    # Phase 2: Endstate Simulations (depends on setup)
    endstate_jobs = setup_jobs.addFollowOnJobFn(
        run_endstate_simulations, 
        updated_config
    )
    endstate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 2 Complete: Endstate simulations finished"
    )
    endstate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 3: System Decomposition and Restraint Generation"
    )
    # Phase 3: System Decomposition and Restraint Generation (depends on endstate)
    decomposition_jobs = endstate_jobs.addFollowOnJobFn(
        decompose_system_and_generate_restraints,
        endstate_jobs.rv(),
        updated_config
    )
    decomposition_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 3 Complete: System decomposition and restraint generation finished"
    )
    decomposition_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 4: Intermediate State Simulations"
    )
    # Phase 4: Intermediate State Simulations (depends on decomposition)
    setup_intermediate_jobs = decomposition_jobs.addFollowOnJobFn(
        setup_intermediate_simulations,
        decomposition_jobs.rv(), 
        endstate_jobs.rv(),
        updated_config
    )

    setup_intermediate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 4 Complete: Intermidate simulations setup finished"
    )
    setup_intermediate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 5: Intermediate State Simulations"
    )
    # Phase 5: Run intermediate state simulations
    run_intermediate_jobs = setup_intermediate_jobs.addFollowOnJobFn(
        run_intermediate_simulations,
        setup_intermediate_jobs.rv(0), # config 
        setup_intermediate_jobs.rv(1), # complex simulations
        setup_intermediate_jobs.rv(2), # receptor simulations   
        setup_intermediate_jobs.rv(3), # ligand simulations
        setup_intermediate_jobs.rv(4), # flat bottom simulations
    )
    run_intermediate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 5 Complete: Intermediate state simulations finished"
    )
    run_intermediate_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 6: Energy post-processing and analysis"
    )
    # Phase 6: Post-processing and Analysis (depends on intermediate)
    analysis_jobs = run_intermediate_jobs.addFollowOnJobFn(
        run_post_analysis_intermediate_simulations,
        setup_intermediate_jobs.rv(0), # config 
        setup_intermediate_jobs.rv(1), # complex simulations
        setup_intermediate_jobs.rv(2), # receptor simulations
        setup_intermediate_jobs.rv(3), # ligand simulations
        setup_intermediate_jobs.rv(4), # flat bottom simulations
    )
    analysis_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 6 Complete: Energy post-processing and analysis finished"
    )
    post_analysis_complete = analysis_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="--> Moving to phase 7: Free energy computation and consolidation"
    )

    # Phase 7: Compute Free Energy and Consolidate Results
    free_energy_difference_jobs = analysis_jobs.addFollowOnJobFn(
        compute_free_energy_and_consolidate,
        analysis_jobs.rv(0), # complex post-analysis results
        analysis_jobs.rv(1), # receptor post-analysis results
        analysis_jobs.rv(2), # receptor post-analysis results
        analysis_jobs.rv(3), # ligand post-analysis results
        setup_intermediate_jobs.rv(0), # config 
    )

    free_energy_difference_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="✓ Phase 7 Complete: Free energy computation and consolidation finished"
    )

    # Final workflow completion message
    
    free_energy_difference_jobs.addFollowOnJobFn(
        initilized_jobs,
        message="🎉 DDM Workflow Complete: All phases finished successfully!"
    )
    
    return free_energy_difference_jobs


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
    options.clean = "onSuccess"
    options.logLevel = "INFO"
    config_file = options.config_file[0]
    ignore_receptor = options.ignore_receptor

    start = time.perf_counter()
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

        else:
            toil.restart()


if __name__ == "__main__":
    main()
