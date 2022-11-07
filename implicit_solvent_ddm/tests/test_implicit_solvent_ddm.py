import logging
import os
import random
import re
import shutil
import time

import parmed as pmd
import pytest
import yaml
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow
from matplotlib.pyplot import get
from toil.common import Toil
from toil.job import Job

working_directory = os.getcwd()


@pytest.fixture(scope="module")
def run_workflow():
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    yaml_file = os.path.join("implicit_solvent_ddm/tests/input_files/config.yaml")
    with open(yaml_file) as yml:
        config_file = yaml.safe_load(yml)

    config = Config.from_config(config_file)

    if options.workDir:
        config.system_settings.working_directory = os.path.abspath(options.workDir)
    else:
        config.system_settings.working_directory = os.getcwd()

    config.ignore_receptor = False

    if not os.path.exists(
        os.path.join(config.system_settings.working_directory, "mdgb/structs/ligand")
    ):
        os.makedirs(
            os.path.join(
                config.system_settings.working_directory, "mdgb/structs/ligand"
            )
        )
    if not os.path.exists(
        os.path.join(config.system_settings.working_directory, "mdgb/structs/receptor")
    ):
        os.makedirs(
            os.path.join(
                config.system_settings.working_directory, "mdgb/structs/receptor"
            )
        )

    # config.get_receptor_ligand_topologies()

    with Toil(options) as toil:
        if not toil.options.restart:
            if config.endstate_method.endstate_method_type == 0:
                config.endstate_files.get_inital_coordinate()

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

            # if not config.ignore_receptor:
            #     config.endstate_files.receptor_parameter_filename = str(toil.import_file("file://" + os.path.abspath("implicit_solvent_ddm/tests/structs/CB7.parm7")))
            #     config.endstate_files.receptor_coordinate_filename = str(toil.import_file("file://" + os.path.abspath("implicit_solvent_ddm/tests/structs/CB7.ncrst.1")))

            toil.start(Job.wrapJobFn(ddm_workflow, config))
            # postprocess analysis

        else:
            toil.restart()
    # cleanup
    yield
    shutil.rmtree('mdgb/')


#     yield mdout_files

# load topology without any changes
parm_file_path = os.path.join("implicit_solvent_ddm/tests/structs/")


@pytest.fixture
def load_topology():
    return pmd.load_file(os.path.join(parm_file_path, "cb7-mol01.parm7"))


# load in complex topology file with vdw/lj-potentials and no ligand charges
@pytest.fixture
def load_exclusions_no_charge():
    return pmd.load_file(
        os.path.join(parm_file_path, "exclusions_no_charge_cb7-M01.parm7")
    )


# load complex topology with Lj-pontential btw Host & Ligand but the ligand charges are set to 0
@pytest.fixture
def load_no_charge_topology():
    return pmd.load_file(os.path.join(parm_file_path, "no_charge_cb7-M01.parm7"))


# walk through all directories and collect mdout files from workflow
@pytest.fixture(scope="module", params=["mdgb/M01/", "mdgb/cb7-mol01/", "mdgb/CB7/"])
def get_mdouts(request, run_workflow):
    mdout_files = []
    for root, dirs, files in os.walk(request.param, topdown=False):
        for name in files:
            if "mdout" == name:
                mdout_files.append(os.path.join(root, name))
    yield mdout_files


@pytest.mark.parametrize("test_input, expected", [("mdgb/M01/", 5), ("mdgb/CB7/", 4), ("mdgb/cb7-mol01/", 6)])
def test_completed_workflow(test_input, expected, run_workflow):
    print('test input:', test_input)
    mdout_files = []
    for root, dirs, files in os.walk(test_input, topdown=False):
        for name in files:
            if "mdout" == name:
                mdout_files.append(os.path.join(root, name))
    
    assert len(mdout_files) == expected
                
# read in all mdout files and check if run was completed
def test_successful_simulations(get_mdouts):
    """
    Executes the entire workflow and does a check if all output files were completed.
    """
    for mdout in get_mdouts:
        complete = False
        with open(mdout) as output:
            data = output.readlines()
            for line in data:
                if "Total time" in line:
                    complete = True

            assert complete


# check complex parameter potential that host and guest have no interactions and ligand charge = 0
def test_complex_exclusions_no_charge(load_exclusions_no_charge):

    output_solute_traj = get_output_file("mdgb/cb7-mol01/no_interactions/")
    assert_exclusions_charges(output_solute_traj, load_exclusions_no_charge)


def test_complex_no_charge_no_gb(load_no_charge_topology):

    output_solute_traj = get_output_file("mdgb/cb7-mol01/interactions/")
    assert_exclusions_charges(output_solute_traj, load_no_charge_topology)


@pytest.mark.parametrize(
    "no_igb_path",
    [
        "mdgb/M01/no_charges/",
        "mdgb/M01/no_gb/",
        "mdgb/CB7/no_gb/",
        "mdgb/cb7-mol01/no_interactions/",
        "mdgb/cb7-mol01/interactions/",
        "mdgb/cb7-mol01/electrostatics/",
    ],
)
def test_no_solvent_complex(no_igb_path):
    for root, dirs, files in os.walk(no_igb_path):
        no_igb = False
        for name in files:
            if "mdout" == name:
                with open(os.path.join(root, name)) as output:
                    data = output.readlines()
                for line in data:
                    if "igb = 6" in line:
                        no_igb = True
                assert no_igb


def assert_exclusions_charges(output, correct_output):

    assert (
        output.parm_data["NUMBER_EXCLUDED_ATOMS"]
        == correct_output.parm_data["NUMBER_EXCLUDED_ATOMS"]
    )

    assert (
        output[":M01"].parm_data["CHARGE"] == correct_output[":M01"].parm_data["CHARGE"]
    )


def get_output_file(output_path):
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if bool(re.match(r".*\.parm7", file)):
                solute = os.path.join(root, file)
            if bool(re.match(r".*\.rst7", file)):
                coordinate = os.path.join(root, file)

    return pmd.load_file(solute, xyz=coordinate)
