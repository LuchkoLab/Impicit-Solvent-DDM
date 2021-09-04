

import simrun
import os, os.path
import yaml
import re 

from pathlib import Path
from argparse import ArgumentParser
from toil.common import Toil
from toil.job import Job

#local imports 
from implicit_solvent_ddm.toil_parser import input_parser
from implicit_solvent_ddm.toil_parser import get_output_dir
from implicit_solvent_ddm.simulations import run_md
from implicit_solvent_ddm.simulations import initilized_jobs
import implicit_solvent_ddm.restraints as restraints


def ddm_workflow(toil, df, argSet, work_dir):

    if argSet['parameters']['mdin'] is None:
        file_name = 'mdin'
        print('No mdin file specified. Generating one automaticalled called: %s' %str(file_name))
        mdin = make_mdin_file(state_label=9)
        mdin_file = toil.importFile("file://" + os.path.abspath(os.path.join(mdin)))
        mdin_filename= 'mdin'
    else:
        mdin_file = toil.importFile("file://" + os.path.abspath(os.path.join(argSet['parameters']['mdin'])))
        mdin_filename = 'mdin'
          
    end_state_job = Job.wrapJobFn(initilized_jobs, work_dir)


    for n in range(len(df.values)):
        end_state_job.addChildJobFn(run_md, df['ligand_file'][n], df['ligand_filename'][n],df['ligand_rst'][n],df['ligand_rst_filename'][n], mdin_file, mdin_filename, get_output_dir(df['ligand_file'][n],2), 2 , argSet, "ligand job")

        end_state_job.addChildJobFn(run_md, df['receptor_file'][n], df['receptor_filename'][n],df['receptor_rst'][n],df['receptor_rst_filename'][n], mdin_file, mdin_filename, get_output_dir(df['receptor_file'][n],2), 2 , argSet, "receptor job")

        complex_job = end_state_job.addChildJobFn(run_md, df['complex_file'][n], df['complex_filename'][n],df['complex_rst'][n],df['complex_rst_filename'][n], mdin_file, mdin_filename, get_output_dir(df['complex_file'][n],9), 9 , argSet, "complex")
          
        restraint_job = complex_job.addFollowOnJobFn(restraints.make_restraints_file, df['complex_file'][n], df['complex_filename'][n], complex_job.rv(), df['complex_rst_filename'][n], argSet["parameters"]["restraint_type"], work_dir)

    return end_state_job

def main():
    
    parser = Job.Runner.getDefaultArgumentParser()
    parser.add_argument('--config_file', nargs='*', type=str, required=True, help="configuartion file with input parameters")
    options = parser.parse_args()
    options.logLevel = "INFO"
    options.clean = "always"

    config = options.config_file[0]
    
    try:
        with open(config) as f:
            argSet = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

    #argSet = {}
    #argSet.update(config)
    
    run_simrun(argSet)
   #create a log file
    Path('mdgb/log_file.txt').touch()
    options.logFile = "mdgb/log_file.txt"
    work_dir = os.getcwd()
    with Toil(options) as toil:
        #dataFrame
        df_inputs = input_parser(argSet,toil)
        end_state_job = ddm_workflow(toil, df_inputs, argSet, work_dir)

        toil.start(end_state_job)


def run_simrun(argSet, dirstruct = "dirstruct"):
    """
    Creates unique directory structure for all output files when created.

    Parameters
    ----------
    argSet: dict
        A dictionary of parameters from a .yaml configuation file
    dirstruct: str
        dirstruct is a preference File used to create new directory structures when simrun.getRun() is called.

    Returns
    -------
    None
    """
    sim = simrun.SimRun("mdgb", description = '''Perform molecular dynamics with GB or in vacuo''')


    struct = sim.getDirectoryStructure(dirstruct)
   #iterate through solutes

    for key in argSet['parameters']:
        if key == 'complex':
            complex_state = 7
            while complex_state <= 9:
                for complex in argSet['parameters'][key]:
                    argSet['solute'] = complex
                    solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                    pathroot = re.sub(r"\.[^.]*","",argSet['solute'])
                    print('pathroot',pathroot)
                    root = os.path.basename(pathroot)
                    print('root',root)
                    argSet['state_label'] = complex_state
                    run = sim.getRun(argSet)
                complex_state = complex_state + 1
        if key == 'ligand_parm':
            ligand_state = 2
            while ligand_state <= 5:
                for ligand in argSet['parameters'][key]:
                    argSet['solute'] = ligand
                    solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                    argSet['state_label'] = ligand_state
                    run = sim.getRun(argSet)
                ligand_state = ligand_state + 1

        if key == 'receptor_parm':
            receptor_state = 2
            while receptor_state <= 5:
                for receptor in argSet['parameters'][key]:
                    argSet['solute'] = receptor
                    solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                    argSet['state_label'] = receptor_state
                    run = sim.getRun(argSet)
                receptor_state = receptor_state + 1
