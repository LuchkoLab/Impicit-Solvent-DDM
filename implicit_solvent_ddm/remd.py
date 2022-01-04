#from genericpath import exists
import subprocess
import string
import os, os.path 
import re 

#from string import Template 
#local imports 
from implicit_solvent_ddm.simulations import run_md
from implicit_solvent_ddm.simulations import initilized_jobs
from implicit_solvent_ddm.IO import export_outputs 
from toil.common import Toil
from toil.job import Job
#from implicit_solvent_ddm.restraints import write_epty_restraint_file
#from implicit_solvent_ddm.restraints import write_restraint_forces
#from implicit_solvent_ddm.alchemical import turn_off_charges 
#from implicit_solvent_ddm.alchemical import alter_topology_file
def remd_workflow(df_config_inputs, argSet, mdins, work_dir):
    
    end_state_job = Job.wrapJobFn(initilized_jobs, work_dir)
    for n in range(len(df_config_inputs)):
        minimization_ligand = end_state_job.addChildJobFn(run_remd,
                                    df_config_inputs['ligand_parameter_filename'][n],df_config_inputs['ligand_coordinate_filename'][n], 
                                    argSet, work_dir, "minimization")

        # equilibratate_ligand = minimization_ligand.addFollowOnJobFn(run_remd,
        #                                                             df_config_inputs['ligand_parameter_filename'][n], minimization_ligand.rv(),
        #                                                             argSet, work_dir, "equil", mdins_config = mdins)
           
        minimization_receptor = end_state_job.addChildJobFn(run_remd, 
                                    df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_coordinate_filename'][n],
                                    argSet, work_dir, "minimization")
        minimization_complex = end_state_job.addChildJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n],df_config_inputs['complex_coordinate_filename'][n],  
                                                  argSet, work_dir, "minimization")
        
    return end_state_job

def run_remd(job, solute_file, solute_rst, arguments, working_directory, runtype, mdins_config=None):
    
    tempDir = job.fileStore.getLocalTempDir()
    solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute_file)
    if not os.path.exists(os.path.join(f"{working_directory}/mdgb/{runtype}/{solu}")):
        os.makedirs(f"{working_directory}/mdgb/{runtype}/{solu}")
        
    output_path = os.path.join(f"{working_directory}/mdgb/{runtype}/{solu}")
    
    #read in parameter and coordinate file 
    solute_filename = os.path.basename(str(solute_file))
    solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
    solute_rst_filename = os.path.basename(str(solute_rst))
    rst = job.fileStore.readGlobalFile(solute_rst, userPath=os.path.join(tempDir, solute_rst_filename))
 
    # mdin_filename = job.fileStore.importFile("file://" + make_remd_mdin_ref(config, runtype))
    # mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
    if runtype == "minimization":
        mdin_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/templates/min.mdin")
        mdin_filename = job.fileStore.importFile("file://" + mdin_path)
        mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
        restart_filename = f"{solu}_{runtype}.rst"
        subprocess.run(["sander", "-O", "-i", mdin, "-p", solute, "-c", rst, "-r", restart_filename ])

        restart_file = export_outputs(job,output_path, mdout='mdout', mdinfo = 'mdinfo', restart = restart_filename)
        #export solute parm
        job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_path, str(os.path.basename(solute)))))
        job.fileStore.exportFile(mdin,"file://" + os.path.abspath(os.path.join(output_path, os.path.basename(mdin))))
        
    else:
        restart_file = ""
        speed = True
        job.log("The current coordinate file is  {} ".format(str(rst)))
        
        if speed:
            write_groupfile = job.fileStore.importFile("file://" + setup_remd(job, tempDir, solute, rst, runtype, mdins_config))
        
            groupfile = job.fileStore.readGlobalFile(write_groupfile, userPath=os.path.join(tempDir, os.path.basename(write_groupfile)))
        
        num_threads = str(arguments["replica_exchange_parameters"]["number_of_threads"])
        exe = arguments["replica_exchange_parameters"]["executable"]
        num_groups = str(arguments["replica_exchange_parameters"]["number_of_replicas"])
        
        job.fileStore.exportFile(groupfile, "file://" + os.path.abspath(os.path.join(output_path, os.path.basename(groupfile))))
                
        subprocess.run(["srun", "-n", num_threads, exe , "-ng", num_groups , "-groupfile", groupfile])
        # mdout_filename = "mdout.rep.000"
        # mdout_file = job.fileStore.writeGlobalFile(mdout_filename)
        # job.fileStore.exportFile(mdout_file,"file://" + os.path.abspath(os.path.join(output_path, mdout_filename)))
        # job.fileStore.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_path, "mdout")))
        
    return restart_file
#function read in all mdins and create the groupfile template?

def setup_remd(md_job, temp_directory, solute, solute_coordinate, runtype, mdins):
    with open('equilibrate.groupfile', 'a+') as group:
        for count, mdin in enumerate(mdins["equilibrate_mdins"]):
            
            read_mdin = md_job.fileStore.readGlobalFile(mdin, userPath=os.path.join(temp_directory, os.path.basename(mdin)))
            solu = re.sub("\..*", "", os.path.basename(solute))
            if runtype == 'equil':
                group.write(f"-O -rem 0 -i {read_mdin} -p {solute} -c {solute_coordinate} -o mdout.rep.{count:03} -r {solu}.rst.{count:03} -x {solu}.mdcrd.{count:03} -inf equilibrate.mdinfo.{count:03} \n")
            elif runtype == 'prod':
                pass 
        
    return os.path.abspath('equilibrate.groupfile')
'''
    #mdin_filename = job.fileStore.importFile("file://" + make_remd_mdin_ref(config, runtype))
    #mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
    solu = re.sub(r"\..*", "", os.basename(solute))    
    
    mdin = make_remd_mdin_ref(config, runtype)
    #a third input file for groupfile 
    temperture = make_temperature_data(config)

    mdins = []
    with open('groufile.ref', 'w') as groufile:
        groupfile.write(f"-O -i mdin.rep.REPNUM -p {solute} -c {solute_coordinate} -o mdout.rep.REPNUM -r rst.rep.REPNUM")

    subprocess.check_call["genremdinputs.py", "-inputs" ,"temperatures.dat" , "-groupfile" , "groupfile.ref" ,"-i", mdin, "-O"]

#will take in config file and open up the mdin.yaml file 

def make_remd_mdin_ref(config,runtype):
    #default values for equilibration 
    default_mdin = {
        'nstlim' : 1000, 
        'irest': 0, 
        'ntx': 1,
        'dt' : 0.002, 
        'ntt': 3,
        'gamma_ln' : 1.0,        
        'ntb' : 0,
        'igb' : 5,
        'cut' : 999.0,
        'rgbmax' : 999.0,
        'ntpr' : 500,
        'ntwx' : 500,
        'ntwr' : 100000,
        'numexchg' : ''}
    #completion of equilibration run production 
    if runtype == 'production':
        production_run = {
            'nstlim': 500,
            'ntpr' : 100,
            'ntwx' : 1000,
            'numexchg' : 1000,
            }
        default_mdin.update(production_run)
        if config["replica_exchange_parameters"]["replica_mdin"]) != None:
            try:
                with open(config["replica_exchange_parameters"]["replica_mdin"]) as f:
                    user_remd_mdin = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
            default_mdin.update(user_remd_mdin)    

    mdin_path = os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)) + "/templates/remd_mdin.ref")

    with open(mdin_path) as t:
        template = Template(t.read())
    
    final_template = template.substitute(
        nstlim = default_mdin['nstlim'], 
        irest = default_mdin['irest'], 
        ntx = default_mdin['ntx'],
        dt = default_mdin['dt'], 
        ntt = default_mdin['ntt'],
        gamma_ln = default_mdin['gamma_ln'],        
        ntb = default_mdin['ntb'],
        igb = default_mdin['igb'],
        cut = default_mdin['cut'],
        rgbmax = default_mdin['rgbmax'],
        ntpr = default_mdin['ntpr'],
        ntwx = default_mdin['ntwx'],
        ntwr = default_mdin['ntwr'],
        numexchg = default_mdin['numexchg']
        restraint = COM.restraint
        )
     with open('mdin.ref', "w") as output:
        output.write(final_template)
    return os.path.abspath('mdin.ref')

def make_temperature_data(config):

    with open('temperature.dat', 'w') as temp:

        temp.write('TEMPERATURE')
        temp.write("Temperature Replica Exchange")
        for temperature in config["replica_exchange_parameters"]["temperatures"]:
            temp.write(str(temperature))
    
    return os.path.abspath('temperature.dat')
'''