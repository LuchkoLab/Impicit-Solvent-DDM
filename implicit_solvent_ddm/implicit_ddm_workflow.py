


import simrun
import os, os.path
import yaml
import re 
import string 
from pathlib import Path
from argparse import ArgumentParser
from toil.common import Toil
from toil.job import Job

#local imports 
import implicit_solvent_ddm.restraints as restraints
from implicit_solvent_ddm.toil_parser import input_parser
from implicit_solvent_ddm.toil_parser import get_output_dir
from implicit_solvent_ddm.simulations import run_md
from implicit_solvent_ddm.simulations import initilized_jobs
from implicit_solvent_ddm.alchemical import split_complex
from implicit_solvent_ddm.toil_parser import get_receptor_ligand_topologies 

def ddm_workflow(toil, df_config_inputs, argSet, work_dir):
    '''
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
    '''
    #run a simple log command 
    end_state_job = Job.wrapJobFn(initilized_jobs, work_dir)

    lambda_windows = [i for i in string.ascii_lowercase] 
    lambda_count = 0 

    #loop through all complexes within the data frame 
    for n in range(len(df_config_inputs)):
        #run simulation for ligand only 
        end_state_job.addChildJobFn(run_md, 
                                    df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n],
                                    df_config_inputs['ligand_coordinate_filename'][n],df_config_inputs['ligand_coordinate_basename'][n], 
                                    get_output_dir(df_config_inputs['ligand_parameter_filename'][n],2), 
                                    argSet, "ligand job")
        #run simulation for receptor only 
        end_state_job.addChildJobFn(run_md, 
                                    df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_parameter_basename'][n],
                                    df_config_inputs['receptor_coordinate_filename'][n],df_config_inputs['receptor_coordinate_basename'][n],  
                                    get_output_dir(df_config_inputs['receptor_parameter_filename'][n],2), 
                                    argSet, "receptor job")
        #run simulation for complex 
        complex_job = end_state_job.addChildJobFn(run_md, 
                                                  df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n],
                                                  df_config_inputs['complex_coordinate_filename'][n], df_config_inputs['complex_coordinate_basename'][n], 
                                                  get_output_dir(df_config_inputs['complex_parameter_filename'][n],9), 
                                                  argSet, "complex")
        #create orentational and conformational restraint templates  
        restraint_job = complex_job.addFollowOnJobFn(restraints.make_restraints_file, 
                                                     df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                     complex_job.rv(0), df_config_inputs['complex_coordinate_basename'][n],  
                                                     df_config_inputs['ligand_parameter_basename'][n], df_config_inputs['receptor_parameter_basename'][n], 
                                                     argSet["parameters"]["ligand_mask"][n], argSet["parameters"]["receptor_mask"], 
                                                     argSet["parameters"]["restraint_type"], work_dir)
        #split the complex coordinates once complex_job is completed 
        split_job = complex_job.addFollowOnJobFn(split_complex, 
                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                 complex_job.rv(0),  
                                                 df_config_inputs['ligand_parameter_basename'][n], df_config_inputs['receptor_parameter_basename'][n],  
                                                 argSet["parameters"]["ligand_mask"][n],  argSet["parameters"]["receptor_mask"], 
                                                 work_dir)
        #loop through conformational restraint forces 
        for conformational_rest in argSet["parameters"]["freeze_restraints_forces"]:
            #begin running intermidate states for ligand 
            ligand_intermidate = split_job.addChildJobFn(run_md, 
                                                         df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n], 
                                                         split_job.rv(0), os.path.basename(str(split_job.rv(0))), 
                                                         get_output_dir(df_config_inputs['ligand_parameter_filename'][n],2), 
                                                         argSet, "running lambda windows: " +str(conformational_rest), 
                                                         work_dir, conformational_restraint = conformational_rest)
            #begin running intermidate states for receptor 
           # receptor_intermidate = split_job.addChildJobFn(run_md,
                                                           #df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_parameter_basename'][n],
                                                           #split_job.rv(1), os.path.basename(str(split_job.rv(1))),
                                                           #mdin_file, mdin_filename,
                                                           #get_output_dir(df_config_inputs['receptor_parameter_filename'][n],2),
                                                           #argSet, "_receptor",
                                                           #work_dir, conformational_rest)
        #turning off the solvent for ligand simulation with force of conformational restraints
        turn_off_solvent_ligand_job = split_job.addChildJobFn(run_md,
                                                              df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n],
                                                              split_job.rv(0), os.path.basename(str(split_job.rv(0))),
                                                              get_output_dir(df_config_inputs['ligand_parameter_filename'][n],4),
                                                              argSet, "solvent off for ligand",
                                                              work_dir, conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], solvent_turned_off=True) 
        #turn off the solvent for receptor simulation with force of conformational restraints
        turn_off_solvent_receptor_job = split_job.addChildJobFn(run_md,
                                                                df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_parameter_basename'][n],
                                                                split_job.rv(1), os.path.basename(str(split_job.rv(1))),
                                                                get_output_dir(df_config_inputs['receptor_parameter_filename'][n],4),
                                                                argSet, "solvent off for receptor",
                                                                work_dir, conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], solvent_turned_off=True)
        #set ligand net charge to 0 with full force of conformational restraints 
        turn_off_ligand_charges_job = split_job.addChildJobFn(run_md,
                                                              df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n],
                                                              split_job.rv(0), os.path.basename(str(split_job.rv(0))),
                                                              get_output_dir(df_config_inputs['ligand_parameter_filename'][n],5),
                                                              argSet, "ligand charge to zero and full conformational",
                                                              work_dir, argSet["parameters"]["ligand_mask"][n], 
                                                              conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], 
                                                              solvent_turned_off=True, charge_off= True,
                                                             )
        # turn on all restraints conformational and orientational
        add_orientational_restraints = restraint_job.addChildJobFn(run_md, 
                                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                                 complex_job.rv(0), os.path.basename(str(complex_job.rv(0))),  
                                                                 get_output_dir(df_config_inputs['complex_parameter_filename'][n],7),
                                                                 argSet, "_orientatinal restraints on",
                                                                 work_dir, argSet["parameters"]["ligand_mask"][n],
                                                                 conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], orientational_restraint = argSet["parameters"]["orientational_restriant_forces"][-1],
                                                                 solvent_turned_off=True, charge_off= True, exculsions=True,
                                                                 
        )
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

    #updates argSet to contain ligand and receptor respective topology and coordinate files. 
    argSet["parameters"].update(get_receptor_ligand_topologies(argSet))
    #create initial directory structure 
    run_simrun(argSet)
    #create a log file
    Path('mdgb/log_file.txt').touch()
    options.logFile = "mdgb/log_file.txt"
    work_dir = os.getcwd()

    with Toil(options) as toil:
        #dataFrame containing absolute paths of topology and coordinate files. Also contains basenames of both file types 
        dataframe_parameter_inputs = input_parser(argSet,toil)
        
        ddm_workflow_job = ddm_workflow(toil, dataframe_parameter_inputs, argSet, work_dir)
        
        #dataframe_parameter_inputs.to_hdf('parameter_data_frame.h5',key='df', mode='w')

        toil.start(ddm_workflow_job)


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

    for key in argSet['parameters'].keys():
        if key == 'complex_parameter_filename':
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
        if key == 'ligand_parameter_filename':
            ligand_state = 2
            while ligand_state <= 5:
                for ligand in argSet['parameters'][key]:
                    argSet['solute'] = ligand
                    solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                    argSet['state_label'] = ligand_state
                    run = sim.getRun(argSet)
                ligand_state = ligand_state + 1

        if key == 'receptor_parameter_filename':
            receptor_state = 2
            while receptor_state <= 5:
                for receptor in argSet['parameters'][key]:
                    argSet['solute'] = receptor
                    solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                    argSet['state_label'] = receptor_state
                    run = sim.getRun(argSet)
                receptor_state = receptor_state + 1
