import os
from pathlib import Path
import re
import shutil

import pytest
import yaml
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow
from toil.common import Toil
from toil.job import Job


working_directory = os.getcwd()


@pytest.fixture(scope="session")
def run_workflow():
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    options.workDir = working_directory

    yaml_file = os.path.join("implicit_solvent_ddm/tests/input_files/config.yaml")
    with open(yaml_file) as yml:
        config_file = yaml.safe_load(yml)

    config = Config.from_config(config_file)

    # create top level directory to write output files
    if not os.path.exists(config.system_settings.top_directory_path):
        os.makedirs(config.system_settings.top_directory_path)
    # # create unique workflow log file
    # job_number = 1
    # complex_name = "cb7_mol01"
    # while os.path.exists(
    #     f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    # ):
    #     job_number += 1
    # Path(
    #     f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    # ).touch()

    # options.logFile = f"{config.system_settings.top_directory_path}/{complex_name}_job_{job_number:03}.txt"
    with Toil(options) as toil:
        config.workflow.ignore_receptor_endstate = False

        config.endstate_files.toil_import_parmeters(toil=toil)
        config.inputs["min_mdin"] = str(
            toil.import_file(
                "file://"
                + os.path.abspath(
                    os.path.dirname(os.path.realpath(__file__))
                    + "/input_files/min.mdin"
                )
            )
        )

        updated_config = toil.start(Job.wrapJobFn(ddm_workflow, config))

    # cleanup
    yield updated_config
    shutil.rmtree(config.system_settings.top_directory_path)
    shutil.rmtree(".cache/")


@pytest.fixture(scope="module")
def get_ligand_config(run_workflow):
    """_summary_

    Args:
        run_workflow (_type_): _description_

    Returns:
        _type_: _description_
    """
    return run_workflow[1]


@pytest.fixture(scope="module")
def get_complex_config(run_workflow):
    """_summary_

    Args:
        run_workflow (_type_): _description_

    Returns:
        _type_: _description_
    """
    return run_workflow[0]


@pytest.fixture(scope="module")
def get_receptor_config(run_workflow):
    """_summary_

    Args:
        run_workflow (_type_): _description_

    Returns:
        _type_: _description_
    """
    return run_workflow[2]
