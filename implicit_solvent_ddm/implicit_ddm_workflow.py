



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
import implicit_solvent_ddm.dirstruct_core as dc 
from implicit_solvent_ddm.toil_parser import input_parser
from implicit_solvent_ddm.toil_parser import get_output_dir
from implicit_solvent_ddm.simulations import run_md
from implicit_solvent_ddm.simulations import initilized_jobs
from implicit_solvent_ddm.alchemical import split_complex
from implicit_solvent_ddm.toil_parser import get_receptor_ligand_topologies 
from implicit_solvent_ddm.remd import remd_workflow
from implicit_solvent_ddm.toil_parser import get_mdins
from implicit_solvent_ddm.toil_parser import import_restraint_files
from implicit_solvent_ddm.remd import run_minimization
#from implicit_solvent_ddm.remd import run_remd

def ddm_workflow(df_config_inputs, argSet, work_dir):
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
    # list of conformational restraint forces 
    con_rest = argSet["parameters"]["freeze_restraints_forces"]
    # list of orientational restraint forces 
    orien_rest = argSet["parameters"]["orientational_restriant_forces"]
    # merge both list into list of tuples
    restraint_tuples = list(map(lambda conformational, orientational:(conformational, orientational), con_rest, orien_rest))

    #loop through all complexes within the data frame 
    for n in range(len(df_config_inputs)):
        #run simulation for ligand only 
        if argSet["replica_exchange_parameters"]["replica_exchange"]:
            complex_job = end_state_job.addChildJobFn(remd_workflow, df_config_inputs, argSet, work_dir)
        #long MD simulations 
        
        else:
            ligand_name = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",df_config_inputs['ligand_parameter_filename'][n])
            
            minimization_ligand = end_state_job.addChildJobFn(run_minimization,
                                    df_config_inputs['ligand_parameter_filename'][n],df_config_inputs['ligand_coordinate_filename'][n], 
                                    argSet, output_path= f"{work_dir}/mdgb/minization/{ligand_name}", COM=False)
            
            ligand_job = minimization_ligand.addFollowOnJobFn(run_md, 
                                        df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n],
                                        minimization_ligand.rv(0),df_config_inputs['ligand_coordinate_basename'][n], 
                                        get_output_dir(df_config_inputs['ligand_parameter_filename'][n],2), 
                                        argSet, "end_state", input_mdin=argSet["parameters"]["end_state_mdin"][0], work_dir=work_dir)
            
            # If the ignore_receptor flag is not called, then run long MD on receptor 
            if not argSet["ignore_receptor"]:
                #run simulation for receptor only 
                receptor_name =  re.sub(r".*/([^/.]*)\.[^.]*",r"\1",df_config_inputs['receptor_parameter_filename'][n])
                
                minimization_receptor = end_state_job.addChildJobFn(run_minimization,
                                        df_config_inputs['receptor_parameter_filename'][n],df_config_inputs['receptor_coordinate_filename'][n], 
                                        argSet, output_path = f"{work_dir}/mdgb/minization/{receptor_name}", COM=False)
                
                receptor_job = minimization_receptor.addFollowOnJobFn(run_md, 
                                            df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_parameter_basename'][n],
                                            minimization_receptor.rv(0),df_config_inputs['receptor_coordinate_basename'][n],  
                                            get_output_dir(df_config_inputs['receptor_parameter_filename'][n],2), 
                                            argSet, "end_state", input_mdin=argSet["parameters"]["end_state_mdin"][0], work_dir=work_dir)
            #run simulation for complex 
            complex_name = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",df_config_inputs['complex_parameter_filename'][n])
            
            minimization_complex = end_state_job.addChildJobFn(run_minimization,
                                    df_config_inputs['complex_parameter_filename'][n],df_config_inputs['complex_coordinate_filename'][n], 
                                    argSet, output_path = f"{work_dir}/mdgb/minization/{complex_name}", COM=True)
            
            complex_job = minimization_complex.addFollowOnJobFn(run_md, 
                                                    df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n],
                                                    minimization_complex.rv(0), df_config_inputs['complex_coordinate_basename'][n], 
                                                    get_output_dir(df_config_inputs['complex_parameter_filename'][n],9), 
                                                    argSet, "end_state", COM=True, input_mdin = argSet["parameters"]["end_state_mdin"][0],
                                                    work_dir=work_dir)
        #create orentational and conformational restraint templates  
        restraint_job = complex_job.addFollowOnJobFn(restraints.make_restraint_files, complex_job.rv(0), argSet, df_config_inputs)

        #split the complex coordinates once complex_job is completed 
        split_job = restraint_job.addFollowOnJobFn(split_complex, 
                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                 complex_job.rv(0),  
                                                 df_config_inputs['ligand_parameter_basename'][n], os.path.basename(argSet["parameters"]["receptor_parameter_filename"][n]),  
                                                 argSet["parameters"]["ligand_mask"][n],  argSet["parameters"]["receptor_mask"], 
                                                 work_dir)
        #loop through conformational restraint forces 
        for conformational_rest in argSet["parameters"]["freeze_restraints_forces"]:
            #begin running intermidate states for ligand 
            ligand_intermidate = split_job.addChildJobFn(run_md, 
                                                         df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n], 
                                                         [split_job.rv(0)], os.path.basename(str(split_job.rv(0))), 
                                                         get_output_dir(df_config_inputs['ligand_parameter_filename'][n],2), 
                                                         argSet, f"lambda_{conformational_rest}", 
                                                         work_dir=work_dir, conformational_restraint = conformational_rest)
            if not argSet["ignore_receptor"]: 
                #begin running intermidate states for receptor 
                receptor_intermidate = split_job.addChildJobFn(run_md,
                                                            df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_parameter_basename'][n],
                                                            [split_job.rv(1)], os.path.basename(str(split_job.rv(1))),
                                                            get_output_dir(df_config_inputs['receptor_parameter_filename'][n],2),
                                                            argSet, f"lambda_{conformational_rest}",
                                                            work_dir=work_dir, conformational_restraint = conformational_rest)
        #turning off the solvent for ligand simulation with force of conformational restraints
        turn_off_solvent_ligand_job = split_job.addChildJobFn(run_md,
                                                              df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n],
                                                              [split_job.rv(0)], os.path.basename(str(split_job.rv(0))),
                                                              get_output_dir(df_config_inputs['ligand_parameter_filename'][n],4),
                                                              argSet, "solvent_off",
                                                              work_dir=work_dir, conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], 
                                                              solvent_turned_off=True) 
        
        #turn off the solvent for receptor simulation with force of conformational restraints
        if not argSet["ignore_receptor"]:
            turn_off_solvent_receptor_job = split_job.addChildJobFn(run_md,
                                                                    df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_parameter_basename'][n],
                                                                    [split_job.rv(1)], os.path.basename(str(split_job.rv(1))),
                                                                    get_output_dir(df_config_inputs['receptor_parameter_filename'][n],4),
                                                                    argSet, "solvent_off",
                                                                    work_dir=work_dir, conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], 
                                                                    solvent_turned_off=True)
        
        # #set ligand net charge to 0 with full force of conformational restraints 
        turn_off_ligand_charges_job = split_job.addChildJobFn(run_md,
                                                              df_config_inputs['ligand_parameter_filename'][n], df_config_inputs['ligand_parameter_basename'][n],
                                                              [split_job.rv(0)], os.path.basename(str(split_job.rv(0))),
                                                              get_output_dir(df_config_inputs['ligand_parameter_filename'][n],5),
                                                              argSet, "ligand charge to zero and full conformational",
                                                              work_dir=work_dir, ligand_mask = argSet["parameters"]["ligand_mask"][n], 
                                                              conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], 
                                                              solvent_turned_off=True, charge_off= True,
                                                             )
        # turn on all restraints conformational/orientational with exclusions 
        add_orientational_restraints = restraint_job.addChildJobFn(run_md, 
                                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                                 complex_job.rv(0), complex_job.rv(0),  
                                                                 get_output_dir(df_config_inputs['complex_parameter_filename'][n],7),
                                                                 argSet, "orientatinal",
                                                                 work_dir=work_dir, ligand_mask = argSet["parameters"]["ligand_mask"][n], receptor_mask = argSet["parameters"]["receptor_mask"],
                                                                 conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], orientational_restraint = argSet["parameters"]["orientational_restriant_forces"][-1],
                                                                 solvent_turned_off=True, charge_off= True, exculsions=True,                                 
                                                             )
        # turn on interactions with receptor and ligand 
        add_back_ligand_receptor_interactions = restraint_job.addChildJobFn(run_md, 
                                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                                 complex_job.rv(0), complex_job.rv(0),  
                                                                 get_output_dir(df_config_inputs['complex_parameter_filename'][n],'7a'),
                                                                 argSet, "exclusion_on",
                                                                 work_dir=work_dir, 
                                                                 ligand_mask = argSet["parameters"]["ligand_mask"][n],receptor_mask = argSet["parameters"]["receptor_mask"],
                                                                 conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], orientational_restraint = argSet["parameters"]["orientational_restriant_forces"][-1],
                                                                 solvent_turned_off=True, charge_off= True, exculsions=False)
        # turn charges back on of the ligand 
        add_back_charges_complex = restraint_job.addChildJobFn(run_md, 
                                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                                 complex_job.rv(0), complex_job.rv(0),  
                                                                 get_output_dir(df_config_inputs['complex_parameter_filename'][n],'7b'),
                                                                 argSet, "charges_on",
                                                                 work_dir=work_dir, ligand_mask = argSet["parameters"]["ligand_mask"][n], receptor_mask =  argSet["parameters"]["receptor_mask"],
                                                                 conformational_restraint = argSet["parameters"]["freeze_restraints_forces"][-1], orientational_restraint = argSet["parameters"]["orientational_restriant_forces"][-1],
                                                                 solvent_turned_off=True, charge_off= False, exculsions=False)
        # slowly turn off the restraints 
        for restraints_forces in restraint_tuples:
            complex_intermidate = restraint_job.addChildJobFn(run_md, 
                                                                 df_config_inputs['complex_parameter_filename'][n], df_config_inputs['complex_parameter_basename'][n], 
                                                                 complex_job.rv(0), complex_job.rv(0),  
                                                                 get_output_dir(df_config_inputs['complex_parameter_filename'][n],'8'),
                                                                 argSet, "_orientatinal restraints on",
                                                                 work_dir=work_dir, ligand_mask = argSet["parameters"]["ligand_mask"][n], receptor_mask = argSet["parameters"]["receptor_mask"],
                                                                 conformational_restraint = restraints_forces[0], orientational_restraint = restraints_forces[1],
                                                                 solvent_turned_off=False, charge_off= False, exculsions=False)

    return end_state_job

def main():
    
    parser = Job.Runner.getDefaultArgumentParser()
    parser.add_argument('--config_file', nargs='*', type=str, required=True, help="configuartion file with input parameters")
    parser.add_argument("--ignore_receptor", action= "store_true", help=" Receptor MD caluculations with not be performed.")
    options = parser.parse_args()
    options.logLevel = "INFO"
    options.clean = "onSuccess"
    config = options.config_file[0]
    
    try:
        with open(config) as f:
            argSet = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

    #updates argSet to contain ligand and receptor respective topology and coordinate files. 
    
    argSet["parameters"]["mdin_intermidate_config"] = os.path.abspath(argSet["parameters"]["mdin_intermidate_config"])
    argSet["ignore_receptor"] = options.ignore_receptor
    #create initial directory structure 
    create_dirstruct(argSet)
     
    #create a log file
    job_number = 1
    while os.path.exists(f"mdgb/log_job_{job_number:03}.txt"):
        job_number +=1
    Path(f"mdgb/log_job_{job_number:03}.txt").touch()
    
    options.logFile = f"mdgb/log_job_{job_number:03}.txt"

    # if not options.workDir: 
    #     work_dir = os.getcwd()
    # else:
    #     work_dir = str(options.workDir)
        
    argSet["parameters"].update(get_receptor_ligand_topologies(argSet)) 
    work_dir = os.getcwd()
    argSet["workDir"] = work_dir
    
    
    with Toil(options) as toil:
        #dataFrame containing absolute paths of topology and coordinate files. Also contains basenames of both file types 
        if not toil.options.restart:
            dataframe_parameter_inputs = input_parser(argSet,toil)
            argSet["parameters"].update(import_restraint_files(argSet, toil))
            
            if argSet["replica_exchange_parameters"]["replica_exchange"]:
                remd_mdins = get_mdins(argSet, toil)
                argSet["replica_exchange_parameters"].update(remd_mdins)
                
                replica_workflow = ddm_workflow(dataframe_parameter_inputs, argSet, work_dir)
                #toil.start(Job.wrapJobFn(remd_workflow))
                toil.start(replica_workflow)
            #run long implicit MD simulation 
            else:
                ddm_workflow_job = ddm_workflow(dataframe_parameter_inputs, argSet, work_dir)
                toil.start(ddm_workflow_job)

        else:
            toil.restart()
    
def create_dirstruct(argSet, dirstruct = "dirstruct"):
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
    sim = dc.Dirstruct("mdgb", description='''Perform molecular dynamics with GB or in vacuo''')
   #iterate through solutes

    for key in argSet['parameters'].keys():
        if key == 'complex_parameter_filename':
            complex_state = 7
            intermidate_state = 7
            while complex_state <= 9:
                for complex in argSet['parameters'][key]:
                    argSet['solute'] = complex
                    solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                    pathroot = re.sub(r"\.[^.]*","",argSet['solute'])
                    #print('pathroot',pathroot)
                    root = os.path.basename(pathroot)
                    #print('root',root)
                    argSet['state_label'] = complex_state
                    run = sim.getRun(argSet)
                    for inter_state in ['a','b','c']:
                        argSet['state_label'] = str(intermidate_state) + inter_state
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
