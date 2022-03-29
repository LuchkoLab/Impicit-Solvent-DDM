import pytest
import yaml
import shutil
import os 
import re 
import random
import time
import parmed as pmd 
from implicit_solvent_ddm.simulations import run_intermidate
from implicit_solvent_ddm.restraints import make_restraint_files
from toil.common import Toil
from toil.job import Job


@pytest.fixture(scope="module")
def setUp(config_workflow):
    
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    complex_topology_file = os.path.abspath(config_workflow[0]["parameters"]["complex_parameter_filename"][0])
    complex_coordinate_file = os.path.abspath(config_workflow[0]["parameters"]["complex_coordinate_filename"][0])
    
    with Toil(options) as toil: 
        toil_coord_import = [toil.import_file(f"file://{complex_coordinate_file}")]
        config_workflow[0]['complex_parameter_filename'] = [toil.import_file(f"file://{complex_topology_file}")]
        config_workflow[0]["parameters"]["receptor_parameter_filename"] = ["receptor_testing.parm7"]
        config_workflow[0]["parameters"]["ligand_parameter_filename"] = ["ligand_testing.parm7"]
        config_workflow[0]["parameters"]["ignore_receptor"] = False 
        
        make_template_restraints = Job.wrapJobFn(make_restraint_files, toil_coord_import, config_workflow[0], config_workflow[0])
        
        toil.start(make_template_restraints)
           
        yield [toil, toil_coord_import, config_workflow[0], config_workflow[1]]
    #cleanup 
    shutil.rmtree('mdgb/')
      
@pytest.fixture(scope="module")
def config_workflow():
    work_dir = os.getcwd()
    with open('implicit_solvent_ddm/tests/mdgb.yaml') as argSet:
        config = yaml.safe_load(argSet)
    with open('implicit_solvent_ddm/tests/workflow.yaml') as run_args:
        intermidate_workflow = yaml.safe_load(run_args)
    
    intermidate_mdin = yaml.safe_load(open(config["parameters"]["mdin_intermidate_config"]))
    
    config["parameters"]["mdin_intermidate_config"] = intermidate_mdin
    config["workDir"] = work_dir
    
    return [config, intermidate_workflow]

@pytest.fixture(scope="module", params=["complex_ligand_exclusions","complex_turn_off_exclusions"])
def run_complex_intermidate(setUp, request):
    file_name = request.param
    setUp[3]["jobs"][file_name]["args"]["topology"] = setUp[2]['complex_parameter_filename'][0]  
    setUp[3]["jobs"][file_name]["args"]["conformational_restraint"] = random.choice(setUp[2]["parameters"]["freeze_restraints_forces"])
    setUp[3]["jobs"][file_name]["args"]["orientational_restraints"] = random.choice(setUp[2]["parameters"]["orientational_restriant_forces"])
    
    complex_exclusions_no_charge_no_gb_job = Job.wrapJobFn(run_intermidate, 
                                                    setUp[1], setUp[2], 
                                                    setUp[3]["jobs"][file_name]["args"].copy())
    
    setUp[0].start(complex_exclusions_no_charge_no_gb_job)
    
    
file_path ='implicit_solvent_ddm/tests/'
#load in topology file with exclusions and without charge
@pytest.fixture 
def load_exclusions_no_charge():
    return pmd.load_file(file_path + 'solutes/exclusions_no_charge_cb7-M01.parm7', xyz= file_path + 'solutes/cb7-mol01.ncrst')

# load topology without exclusions and charge 
@pytest.fixture
def load_no_charge_topology():
    return pmd.load_file(file_path +'solutes/no_charge_cb7-M01.parm7',xyz= file_path +'solutes/cb7-mol01.ncrst') 

# load topology without any changes 
@pytest.fixture
def load_topology():
    return pmd.load_file(file_path +'solutes/cb7-mol01.parm7', xyz= file_path +'solutes/cb7-mol01.ncrst')


def test_complex_exclusions_no_charge_no_gb(run_complex_intermidate, load_exclusions_no_charge):
   
    output_solute_traj = get_output_file('mdgb/cb7-mol01/7/')          
    assert_exclusions_charges(output_solute_traj, load_exclusions_no_charge)


def test_complex_no_charge_no_gb(load_no_charge_topology):
    
    output_solute_traj = get_output_file('mdgb/cb7-mol01/7a/') 
    assert_exclusions_charges(output_solute_traj, load_no_charge_topology)

#@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_no_solvent_complex():
    
    for root, dirs, files in os.walk('mdgb/cb7-mol01/'):
        for file in files:
            if bool(re.match(r"mdout", file)):
                with open(os.path.join(root,file)) as output:
                    data = output.readlines()
                    for line in data:
                        if 'igb = 6' in line:
                            assert True
                             
def assert_exclusions_charges(output, correct_output):

    assert output.parm_data['NUMBER_EXCLUDED_ATOMS'] == correct_output.parm_data['NUMBER_EXCLUDED_ATOMS']

    assert output[':M01'].parm_data['CHARGE'] == correct_output[':M01'].parm_data['CHARGE']

def get_output_file(output_path):
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if bool(re.match(r".*\.parm7", file)):
                solute = os.path.join(root, file) 
            if bool(re.match(r".*\.rst7", file)):
                coordinate = os.path.join(root, file)
                 
    output_solute_traj = pmd.load_file(solute, xyz=coordinate)
    
    return output_solute_traj