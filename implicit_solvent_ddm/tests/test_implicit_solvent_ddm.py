
import os
from _pytest import outcomes
from numpy import logaddexp 
import pytest
import shutil
import subprocess
import tempfile 
import parmed as pmd 

#local imports 
from implicit_solvent_ddm.alchemical import alter_topology_file


class Implicit_solvent_Tests:

    def __init__(self):
        self.work_dir = tempfile.mkdtemp()
        self.workflow_command = ['run_implicit_ddm.py', 'file:jobstore', '--config_file', 'mdgb.yaml', '--workDir', str(self.work_dir)]
    def _run(self):
        subprocess.check_call(self.workflow_command)

    def tearDown(self):
        shutil.rmtree('mdgb/')

@pytest.fixture
def workflow():
    workflow = Implicit_solvent_Tests()
    return workflow

#load in topology file with exclusions and without charge
@pytest.fixture 
def load_exclusions_no_charge():
    return pmd.load_file('solutes/exclusions_no_charge_cb7-M01.parm7', xyz= 'solutes/cb7-mol01.ncrst')

# load topology without exclusions and charge 
@pytest.fixture
def load_no_charge_topology():
    return pmd.load_file('solutes/no_charge_cb7-M01.parm7',xyz= 'solutes/cb7-mol01.ncrst') 

# load topology without any changes 
@pytest.fixture
def load_topology():
    return pmd.load_file('solutes/cb7-mol01.parm7', xyz= 'solutes/cb7-mol01.ncrst')

# test complex w/ exclusions & w/o charges/GB
def test_complex_exclusions_no_charge_no_gb(load_exclusions_no_charge, workflow):
    workflow._run()
    solute_parm = 'mdgb/cb7-mol01/7/4.0_16.0/charges_off_exculsions_cb7-mol01.parm7'
    solute_resrt = 'mdgb/cb7-mol01/7/4.0_16.0/restrt'
    output_solute_traj = pmd.load_file(solute_parm, xyz=solute_resrt)
    #test exclusions and charges  
    assert_exclusions_charges(output_solute_traj, load_exclusions_no_charge)

#test for complex w/o exclusions, charge and GB 
def test_complex_no_charge_no_gb(load_no_charge_topology):
    solute_parm = 'mdgb/cb7-mol01/7a/4.0_16.0/charges_off_cb7-mol01.parm7'
    solute_resrt = 'mdgb/cb7-mol01/7a/4.0_16.0/restrt'
    output_solute = pmd.load_file(solute_parm, xyz=solute_resrt)
    #test exclusions and charges 
    assert_exclusions_charges(output_solute, load_no_charge_topology)


def test_complex_no_exclusions(load_topology):
    solute_parm = 'mdgb/cb7-mol01/9/0/cb7-mol01.parm7'
    solute_resrt = 'mdgb/cb7-mol01/9/0/restrt'
    output_solute = pmd.load_file(solute_parm, xyz=solute_resrt)
    # test exclusions and charges 
    assert_exclusions_charges(output_solute,load_topology)

#checks for both state 4 and 5 without GB solvent 

# unittest for altering the charges or adding exclusion 
def test_charge_off_exclusions(load_exclusions_no_charge, workflow):
    
    parm_file = alter_topology_file('solutes/cb7-mol01.parm7', 'solutes/cb7-mol01.ncrst', ':M01', receptor_mask = ':CB7', turn_off_charges = True, add_exclusions = True)

    parm_traj = pmd.load_file(parm_file, xyz = 'solutes/cb7-mol01.ncrst')

    for charge in parm_traj[':M01'].parm_data['CHARGE']:
        assert 0.0 == charge

    assert parm_traj.parm_data['NUMBER_EXCLUDED_ATOMS'] == load_exclusions_no_charge.parm_data['NUMBER_EXCLUDED_ATOMS']
    
    os.remove(parm_file)
    workflow.tearDown()

# test for running restraints for receptor, ligand and complex 


def assert_exclusions_charges(output, correct_output):

    assert output.parm_data['NUMBER_EXCLUDED_ATOMS'] == correct_output.parm_data['NUMBER_EXCLUDED_ATOMS']

    assert output[':M01'].parm_data['CHARGE'] == correct_output[':M01'].parm_data['CHARGE']

#search for EGB average 
#re.search(r"EGB\s+=\s+(.\d+\.\d+)",zero_egb).group(1).strip()