#from genericpath import exists
import subprocess
import string
import os, os.path 
import re 
import itertools
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

        equilibrate_ligand = minimization_ligand.addFollowOnJobFn(run_remd,
                                                                    df_config_inputs['ligand_parameter_filename'][n], minimization_ligand.rv(),
                                                                    argSet, work_dir, "equil", mdins_config = mdins)
        remd_ligand = equilibrate_ligand.addFollowOnJobFn(run_remd,
                                                                    df_config_inputs['ligand_parameter_filename'][n], equilibrate_ligand.rv(),
                                                                    argSet, work_dir, "remd", mdins_config = mdins)
           
        minimization_receptor = end_state_job.addChildJobFn(run_remd, 
                                    df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_coordinate_filename'][n],
                                    argSet, work_dir, "minimization")
        
        equilibrate_receptor = minimization_receptor.addFollowOnJobFn(run_remd,
                                                                    df_config_inputs['receptor_parameter_filename'][n], minimization_receptor.rv(),
                                                                    argSet, work_dir, "equil", mdins_config = mdins)
        
        remd_receptor = equilibrate_receptor.addFollowOnJobFn(run_remd,
                                                                    df_config_inputs['receptor_parameter_filename'][n], equilibrate_receptor.rv(),
                                                                    argSet, work_dir, "remd", mdins_config = mdins)
        
        minimization_complex = end_state_job.addChildJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n],df_config_inputs['complex_coordinate_filename'][n],  
                                                  argSet, work_dir, "minimization")
        equilibrate_complex = minimization_complex.addFollowOnJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n], minimization_complex.rv(),  
                                                  argSet, work_dir, "equil",  mdins_config = mdins)
        
        remd_complex = equilibrate_complex.addFollowOnJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n], equilibrate_complex.rv(),  
                                                  argSet, work_dir, "remd",  mdins_config = mdins)
        
    return end_state_job

def run_remd(job, solute_file, solute_rst, arguments, working_directory, runtype, mdins_config=None):
    
    tempDir = job.fileStore.getLocalTempDir()
    solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute_file)
    if not os.path.exists(os.path.join(f"{working_directory}/mdgb/{runtype}/{solu}")):
        os.makedirs(f"{working_directory}/mdgb/{runtype}/{solu}")
        
    output_path = os.path.join(f"{working_directory}/mdgb/{runtype}/{solu}")
    
    #read in parameter file 
    solute_filename = os.path.basename(str(solute_file))
    solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
 
    # mdin_filename = job.fileStore.importFile("file://" + make_remd_mdin_ref(config, runtype))
    # mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
    if runtype == "minimization":
        job.log(f"running minimization the coordinate list is {solute_rst}")
        solute_rst_filename = os.path.basename(str(solute_rst))
        rst = job.fileStore.readGlobalFile(solute_rst, userPath=os.path.join(tempDir, solute_rst_filename))
        mdin_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/templates/min.mdin")
        mdin_filename = job.fileStore.importFile("file://" + mdin_path)
        mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
        restart_filename = f"{solu}_{runtype}.rst"
        files_current_directory = os.listdir(tempDir)
        job.log(f"files in the current directory {files_current_directory}")
        subprocess.run(["sander", "-O", "-i", mdin, "-p", solute, "-c", rst, "-r", restart_filename ])
        restart_file = export_outputs(job, output_path, files_current_directory)
        #restart_file = export_outputs(job,output_path, mdout='mdout', mdinfo = 'mdinfo', restart = restart_filename)
        
        #export solute parm and mdin 
        job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_path, str(os.path.basename(solute)))))
        job.fileStore.exportFile(mdin,"file://" + os.path.abspath(os.path.join(output_path, os.path.basename(mdin))))
        
    else:
        if runtype == 'equil':
            job.log("Running equilibration")
            job.log(f"current is equilibration the coordiate file is {solute_rst[0]}")
            #read in minimization coordinate file 
            #solute_rst_filename = os.path.basename(str(solute_rst[0]))
            #rst = job.fileStore.readGlobalFile(solute_rst[0], userPath=os.path.join(tempDir, solute_rst_filename))
        elif runtype == 'remd':
            job.log("Running Replica Exchange Molecular Dynamics")
        #write a groupfile for certain solute and coorindate files and read it in 
        write_groupfile = job.fileStore.importFile("file://" + setup_remd(job, tempDir, solute, solute_rst, runtype, mdins_config))
        groupfile = job.fileStore.readGlobalFile(write_groupfile, userPath=os.path.join(tempDir, os.path.basename(write_groupfile)))
        job.log(f"groupfile is {groupfile}")
        
        exe = arguments["replica_exchange_parameters"]["executable"].split()
        exe.append(groupfile)
        job.log(f"the execution is {exe}")
        #export groupfile that was written 
        job.fileStore.exportFile(groupfile, "file://" + os.path.abspath(os.path.join(output_path, os.path.basename(groupfile))))
        #current files in the temporary directory
        files_current_directory = os.listdir(tempDir)
        job.log(f"files in the current directory {files_current_directory}")
        
        #subprocess.run(exe)
        submit_job(exe,job)
        
        output_files = os.listdir(tempDir)
        job.log(f"files after subprocess was executed {output_files}")
        restart_file = export_outputs(job, output_path, files_current_directory)
        
    return restart_file
#function read in all mdins and create the groupfile template?

def setup_remd(md_job, temp_directory, solute, solute_coordinate, runtype, mdins):
    if runtype == 'equil':
        solute_rst_filename = os.path.basename(str(solute_coordinate[0]))
        equil_coordinate = md_job.fileStore.readGlobalFile(solute_coordinate[0], userPath=os.path.join(temp_directory, solute_rst_filename))
        #equilibration mdins 
        runtype_mdins = mdins["equilibrate_mdins"]
    elif runtype == 'remd':
        runtype_mdins = mdins["remd_mdins"]

    with open('equilibrate.groupfile', 'a+') as group:
        for count, mdin in enumerate(runtype_mdins):
            read_mdin = md_job.fileStore.readGlobalFile(mdin, userPath=os.path.join(temp_directory, os.path.basename(mdin)))
            solu = re.sub("\..*", "", os.path.basename(solute))
            if runtype == 'equil':
                group.write(f'''-O -rem 0 -i {read_mdin} 
                -p {solute} -c {equil_coordinate} 
                -o equilibrate.mdout.{count:03} -inf equilibrate.mdinfo.{count:03}
                -r {solu}_equilibrate.rst.{count:03} -x {solu}_equilibrate.mdcrd.{count:03}'''.replace('\n', '') +"\n")
            elif runtype == 'remd':
                single_coordinate = [coordinate for coordinate in solute_coordinate if re.search(rf".*.rst.{count:03}", coordinate)]
                read_coordinate = md_job.fileStore.readGlobalFile(single_coordinate[0], userPath=os.path.join(temp_directory, os.path.basename(single_coordinate[0])))
                group.write(f'''-O -rem 1 -remlog rem.log
                    -i {read_mdin} -p {solute} 
                    -c {read_coordinate} -o remd.mdout.rep.{count:03} 
                    -r {solu}_remd.rst.{count:03} -x {solu}_remd.mdcrd.{count:03} 
                    -inf remd.mdinfo.{count:03}'''.replace('\n', '') +"\n")
    return os.path.abspath('equilibrate.groupfile')

    # with open('equilibrate.groupfile', 'a+') as group:
    #     for count, mdin in enumerate(mdins["equilibrate_mdins"]):
    #         read_mdin = md_job.fileStore.readGlobalFile(mdin, userPath=os.path.join(temp_directory, os.path.basename(mdin)))
    #         solu = re.sub("\..*", "", os.path.basename(solute))
    #         if runtype == 'equil':
    #             group.write(f"-O -rem 0 -i {read_mdin} -p {solute} -c {solute_coordinate} -o mdout.rep.{count:03} -r {solu}.rst.{count:03} -x {solu}.mdcrd.{count:03} -inf equilibrate.mdinfo.{count:03} \n")
    #         elif runtype == 'prod':
    #             pass 
    # return os.path.abspath('equilibrate.groupfile')
def submit_job(executable, exe_job):
    try:
        subprocess.run(executable, capture_output=True)
    except subprocess.CalledProcessError as e:
        exe_job.log(e.output)
    
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