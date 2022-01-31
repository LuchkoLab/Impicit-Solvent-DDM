#from genericpath import exists
from ast import arguments
from pickle import NONE
import subprocess
import os, os.path 
import re 
import itertools
import shutil
from string import Template
import parmed as pmd 
#from string import Template 
#local imports 
#from implicit_solvent_ddm.simulations import run_md
#from implicit_solvent_ddm.sim4ulations import initilized_jobs 
from implicit_solvent_ddm.IO import export_outputs 
from toil.common import Toil
from toil.job import Job
#from implicit_solvent_ddm.restraints import write_epty_restraint_file
#from implicit_solvent_ddm.restraints import write_restraint_forces
#from implicit_solvent_ddm.alchemical import turn_off_charges 
#from implicit_solvent_ddm.alchemical import alter_topology_file
def remd_workflow(job, df_config_inputs, argSet, work_dir):
    
    # end_state_job = Job.wrapJobFn(initilized_jobs, work_dir)
    # job.addChild(end_state_job)
    
    for n in range(len(df_config_inputs)):
        
        minimization_ligand = Job.wrapJobFn(run_remd,
                                    df_config_inputs['ligand_parameter_filename'][n],df_config_inputs['ligand_coordinate_filename'][n], 
                                    argSet, work_dir, "minimization")
        job.addChild(minimization_ligand)
        
        equilibrate_ligand = minimization_ligand.addFollowOnJobFn(run_remd,
                                                                    df_config_inputs['ligand_parameter_filename'][n], minimization_ligand.rv(0),
                                                                    argSet, work_dir, "equil")
        job.addChild(equilibrate_ligand)
        
        remd_ligand = equilibrate_ligand.addFollowOnJobFn(run_remd,
                                                                    df_config_inputs['ligand_parameter_filename'][n], equilibrate_ligand.rv(0),
                                                                    argSet, work_dir, "remd")
        ligand_extract = remd_ligand.addFollowOnJobFn(extract_traj, df_config_inputs['ligand_parameter_filename'][n], remd_ligand.rv(1), 
                                                      work_dir, argSet)
        job.addChild(ligand_extract)
    
        minimization_complex = Job.wrapJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n],df_config_inputs['complex_coordinate_filename'][n],  
                                                  argSet, work_dir, "minimization", COM=True)
        job.addChild(minimization_complex)
        equilibrate_complex = minimization_complex.addFollowOnJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n], minimization_complex.rv(0),  
                                                  argSet, work_dir, "equil",  COM=True)
        job.addChild(equilibrate_complex)
        
        remd_complex = equilibrate_complex.addFollowOnJobFn(run_remd, 
                                                  df_config_inputs['complex_parameter_filename'][n], equilibrate_complex.rv(0),  
                                                  argSet, work_dir, "remd", COM=True)
        job.addChild(remd_complex)
        
        complex_extract = remd_complex.addFollowOnJobFn(extract_traj, df_config_inputs['complex_parameter_filename'][n], remd_complex.rv(1), 
                                                      work_dir, argSet)
        job.addChild(complex_extract)
        
        #if the ingore_receptor flag is not given then run REMD on receptor system 
        if not argSet["ignore_receptor"]:
            minimization_receptor = Job.wrapJobFn(run_remd, 
                                        df_config_inputs['receptor_parameter_filename'][n], df_config_inputs['receptor_coordinate_filename'][n],
                                        argSet, work_dir, "minimization")
            job.addChild(minimization_receptor)
            equilibrate_receptor = minimization_receptor.addFollowOnJobFn(run_remd,
                                                                        df_config_inputs['receptor_parameter_filename'][n], minimization_receptor.rv(0),
                                                                        argSet, work_dir, "equil")
            job.addChild(equilibrate_receptor)
            remd_receptor = equilibrate_receptor.addFollowOnJobFn(run_remd,
                                                                        df_config_inputs['receptor_parameter_filename'][n], equilibrate_receptor.rv(0),
                                                                        argSet, work_dir, "remd")
            extract_receptor = remd_receptor.addFollowOnJobFn(extract_traj, df_config_inputs['receptor_parameter_filename'][n], remd_receptor.rv(1), 
                                                        work_dir, argSet)
            job.addChild(extract_receptor)
        
    return complex_extract.rv()

def run_remd(job, solute_file, solute_rst, arguments, working_directory, runtype, COM=False):
    # temporary directory 
    tempDir = job.fileStore.getLocalTempDir()
    solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute_file)
    # make output directory if it does not exist 
    if not os.path.exists(os.path.join(f"{working_directory}/mdgb/{runtype}/{solu}")):
        os.makedirs(f"{working_directory}/mdgb/{runtype}/{solu}")
        
    output_path = os.path.join(f"{working_directory}/mdgb/{runtype}/{solu}")
    
    if runtype == "minimization":
        restart_file = run_minimization(job, solute_file, solute_rst, arguments, output_path, COM) 
    else:
        restart_file = run_MD_groups(job, solute_file, solute_rst, tempDir, runtype, arguments, output_path,COM)
    return restart_file
#function read in all mdins and create the groupfile template?

def run_minimization(min_job, solute_topology_file, solute_coordinate_file, args, output_path, COM):
    
    temporary_directory =  min_job.fileStore.getLocalTempDir()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    #read in topology file 
    solute_filename = os.path.basename(str(solute_topology_file))
    solu = re.sub(r"\..*","",solute_filename)
    solute = min_job.fileStore.readGlobalFile(solute_topology_file,  userPath=os.path.join(temporary_directory, solute_filename))
    #read in coordinate file
    min_job.log(f"running minimization the coordinate list is {solute_coordinate_file}")
    solute_rst_filename = os.path.basename(str(solute_coordinate_file))
    rst = min_job.fileStore.readGlobalFile(solute_coordinate_file, userPath=os.path.join(temporary_directory, solute_rst_filename))
    
    flat_bottom = args["parameters"]["flat_bottom_restraints"][0]
    #if the system is an complex copy -> flat_bottom restraint file
    if COM:
        import_restraint = min_job.fileStore.importFile("file://" + copy(flat_bottom))
    else:
        import_restraint =  min_job.fileStore.importFile("file://" + write_empty_restraint(flat_bottom))
        
    min_job.log(f"the flat bottom import file {flat_bottom}")
    restraint = min_job.fileStore.readGlobalFile(import_restraint, userPath=os.path.join(temporary_directory, os.path.basename(import_restraint)))
    min_job.log(f"read in global flat bottom {restraint}")
    
    
    mdin_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/templates/min.mdin")
    mdin_filename = min_job.fileStore.importFile("file://" + mdin_path)
    mdin = min_job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(temporary_directory, os.path.basename(mdin_filename)))
    
    #check what files are in the current temporary directory  
    files_current_directory = os.listdir(temporary_directory)
    min_job.log(f"files in the current directory {files_current_directory}")
    
    # Name: output final coordinates , velocity, and box dimensions if any -for restarting run
    restart_filename = f"{solu}_minimization.rst7"
    
    #exectute serial Sander for minimization 
    output = subprocess.run(["sander", "-O", "-i", mdin, "-p", solute, "-c", rst, "-r", restart_filename], 
                            capture_output=True)
    min_job.log(f"the min output : {output}")
    minimization_restart = export_outputs(min_job, output_path, files_current_directory)
   
    #export solute parm and mdin 
    min_job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_path, str(os.path.basename(solute)))))
    min_job.fileStore.exportFile(mdin,"file://" + os.path.abspath(os.path.join(output_path, os.path.basename(mdin))))
    min_job.fileStore.exportFile(restraint, "file://" + os.path.abspath(os.path.join(output_path,"COM.restraint")))
    return minimization_restart

def copy(file):
    basename = str(os.path.basename(file))
    copied_file = shutil.copyfile(file, basename)
    return os.path.abspath(copied_file)

def write_empty_restraint(file):
    restraint_basename = os.path.basename(file)
    file = open(restraint_basename,"w+")
    file.write("")
    file.close()
    return os.path.abspath(restraint_basename)
    
def run_MD_groups(md_group_job, solute_topology_file, solute_coordinate_file, temporary_directory, runtype, arguments, output_path,COM):
    #read in topology file 
    solute_filename = os.path.basename(str(solute_topology_file))
    solu = re.sub(r"\..*","",solute_filename)
    solute = md_group_job.fileStore.readGlobalFile(solute_topology_file,  userPath=os.path.join(temporary_directory, solute_filename))
    
    flat_bottom = arguments["parameters"]["flat_bottom_restraints"][0]
    #if the system is an complex copy -> flat_bottom restraint file
    if COM:
        import_restraint = md_group_job.fileStore.importFile("file://" + copy(flat_bottom))
        #if system is complex read in flat bottom else write empty restraint  
        #copy restraint file 
        #copy_flat_bottom = shutil.copyfile(flat_bottom, restraint_basename)
        #import_restraint = min_job.fileStore.importFile("file://" + os.path.abspath(copy_flat_bottom))
    #write an empty restraint 
    else:
        import_restraint =  md_group_job.fileStore.importFile("file://" + write_empty_restraint(flat_bottom))
        
    if runtype == 'equil':
        md_group_job.log("Running equilibration")
        md_group_job.log(f"current is equilibration the coordiate file is {solute_coordinate_file[0]}")
        #read in minimization coordinate file 
        #solute_rst_filename = os.path.basename(str(solute_rst[0]))
        #rst = job.fileStore.readGlobalFile(solute_rst[0], userPath=os.path.join(tempDir, solute_rst_filename))
    elif runtype == 'remd':
        md_group_job.log("Running Replica Exchange Molecular Dynamics")
        #write a groupfile for certain solute and coorindate files and read it in 
    write_groupfile = md_group_job.fileStore.importFile("file://" + setup_remd(md_group_job, temporary_directory, solute, solute_coordinate_file, runtype, arguments))
    groupfile = md_group_job.fileStore.readGlobalFile(write_groupfile, userPath=os.path.join(temporary_directory, os.path.basename(write_groupfile)))
    md_group_job.log(f"groupfile is {groupfile}")
    

    exe = arguments["replica_exchange_parameters"]["executable"].split()
    exe.append(groupfile)
    md_group_job.log(f"the execution is {exe}")
    #export groupfile that was written 
    md_group_job.fileStore.exportFile(groupfile, "file://" + os.path.abspath(os.path.join(output_path, os.path.basename(groupfile))))
    #current files in the temporary directory
    files_current_directory = os.listdir(temporary_directory)
    md_group_job.log(f"files in the current directory {files_current_directory}")
    
    #subprocess.run(exe)
    submit_job(exe,md_group_job)
    
    output_files = os.listdir(temporary_directory)
    md_group_job.log(f"files after subprocess was executed {output_files}")
    restart_file = export_outputs(md_group_job, output_path, files_current_directory)

    return restart_file

def setup_remd(md_job, temp_directory, solute, solute_coordinate, runtype, arguments):
    
    if runtype == 'equil':
        solute_rst_filename = os.path.basename(str(solute_coordinate[0]))
        equil_coordinate = md_job.fileStore.readGlobalFile(solute_coordinate[0], userPath=os.path.join(temp_directory, solute_rst_filename))
        #equilibration mdins 
        runtype_mdins = arguments["replica_exchange_parameters"]["equilibrate_mdins"]
    elif runtype == 'remd':
        runtype_mdins = arguments["replica_exchange_parameters"]["remd_mdins"]

    with open('equilibrate.groupfile', 'a+') as group:
        for count, mdin in enumerate(runtype_mdins):
            read_mdin = md_job.fileStore.readGlobalFile(mdin, userPath=os.path.join(temp_directory, os.path.basename(mdin)))
            solu = re.sub("\..*", "", os.path.basename(solute))
            if runtype == 'equil':
                group.write(f'''-O -rem 0 -i {read_mdin} 
                -p {solute} -c {equil_coordinate} 
                -o equilibrate.mdout.{count:03} -inf equilibrate.mdinfo.{count:03}
                -r {solu}_equilibrate.rst7.{count:03} -x {solu}_equilibrate.nc.{count:03}'''.replace('\n', '') +"\n")
            elif runtype == 'remd':
                single_coordinate = [coordinate for coordinate in solute_coordinate if re.search(rf".*.rst7.{count:03}", coordinate)]
                read_coordinate = md_job.fileStore.readGlobalFile(single_coordinate[0], userPath=os.path.join(temp_directory, os.path.basename(single_coordinate[0])))
                group.write(f'''-O -rem 1 -remlog rem.log
                    -i {read_mdin} -p {solute} 
                    -c {read_coordinate} -o remd.mdout.rep.{count:03} 
                    -r {solu}_remd.rst7.{count:03} -x {solu}_remd.nc.{count:03} 
                    -inf remd.mdinfo.{count:03}'''.replace('\n', '') +"\n")
    return os.path.abspath('equilibrate.groupfile')

def extract_traj(job, solute_topology, traj_files, workdir, arguments):
    tempDir =  job.fileStore.getLocalTempDir()
    job.log(f"extracting trajectories {traj_files}")
    read_trajectories = readfn(job, traj_files)
    read_solute = readfn(job, [solute_topology])
    temperature = arguments["replica_exchange_parameters"]["target_temperature"]
    
    job.log(f"get the bash script with target temp {temperature}")    
    
    bash_script = get_bash(job, solute=read_solute, 
                           coordinate_files=read_trajectories,
                           target_temperature=temperature)
    
    extract_trajectory = run_bash(job, bash_script)
    read_extract_traj = job.fileStore.readGlobalFile(extract_trajectory, userPath=os.path.join(tempDir, os.path.basename(extract_trajectory)))
    
    solu = re.sub(r"\..*", "", os.path.basename(read_solute[0]))
    lastframe = f"{solu}_{temperature}K_lastframe.ncrst"
    lastframe_rst7 = f"{solu}_{temperature}K_lastframe.rst7"
    
    subprocess.run(['cpptraj', '-p', read_solute[0] , '-y', read_extract_traj, '-ya', 'lastframe','-x', lastframe])
    
    output = subprocess.run(['cpptraj', '-p', read_solute[0] , '-y', lastframe, '-x', lastframe_rst7], 
                            capture_output=True)
    job.log(f"{output}")
    output = subprocess.run(['cpptraj', '-p', read_solute[0] , '-y', lastframe_rst7, '-x', lastframe],
                            capture_output=True)
    job.log(f"last subprocess {output}")
    final_traj = job.fileStore.writeGlobalFile(lastframe)
    job.fileStore.exportFile(final_traj,"file://" + 
                             os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/mdgb/", lastframe)))
    ncrst = [final_traj]
    rst7 = [lastframe_rst7]
    return (ncrst, rst7)
   
def run_bash(job, executable_file):
    job.log(f"running bash {executable_file}")
    current_files = os.listdir()
    output = subprocess.run(["bash", executable_file], capture_output=True)
    job.log(f"the capture output {output}")
    for file in os.listdir():
        if file not in current_files:
            output_file = job.fileStore.writeGlobalFile(file)
    job.log(f"the capture output trajectory is {output_file}")
    return output_file

    #job.fileStore.writeGlobalFile("topology_ligand_0_300.0K.nc")

def readfn(job, stage_file):
    job.log(f"readfn will begin reading in {stage_file}")
    job.log(f"the length size for {len(stage_file)}")  
    tempDir =  job.fileStore.getLocalTempDir()
    read_files = []
    if len(stage_file) > 1:
        for file in stage_file:
            file_basename  = os.path.basename(file)
            read_files.append(job.fileStore.readGlobalFile(file,  userPath=os.path.join(tempDir, file_basename)))
    else:
        file_basename  = os.path.basename(stage_file[0])
        read_files.append(job.fileStore.readGlobalFile(stage_file[0],  userPath=os.path.join(tempDir, file_basename)))
    
    job.log(f"the read global files are {read_files}")
    return read_files

def get_bash(job, solute, coordinate_files, target_temperature):
    tempDir =  job.fileStore.getLocalTempDir()
    job.log(f"bash script: the solute {solute} and the coordinate_files")
    #initial starting frame trajectory_name.nc.000    
    initial_coordinate = list(filter(lambda coordinate: re.match(r".*\.nc.000", coordinate), coordinate_files))
    job.log(f"the initial_coordinate file is {initial_coordinate}")
    bash_script = os.path.abspath(os.path.dirname(
            os.path.realpath(__file__)) + "/templates/cpptraj_remd.sh")
    with open(bash_script) as t:
        template = Template(t.read())
    
    solu = re.sub(r"\..*", "", os.path.basename(solute[0]))
    output_trajectory_filename = f"{solu}_{target_temperature}K"
    
    final_template = template.substitute(
        solute = solute[0],
        trajectory = initial_coordinate[0],
        target_temperature = target_temperature,
        temperature_traj = output_trajectory_filename,
    )
    with open(f"cpptraj_extract_{target_temperature}K.x", "w") as output:
        output.write(final_template)
    job.log(f"import bash file: cpptraj_extract_{target_temperature}K.x")
    import_bash = job.fileStore.importFile("file://" + os.path.abspath(f"cpptraj_extract_{target_temperature}K.x"))
    job.log(f"after import {import_bash}")
    read_bash = job.fileStore.readGlobalFile(import_bash,  userPath=os.path.join(tempDir , os.path.basename(import_bash)))
    job.log(f"readGlobal bash script {read_bash}")
    #job.fileStore.exportFile(read_bash,"file://" + os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/mdgb/", os.path.basename(read_bash))))
    
    return read_bash
    #return os.path.abspath(f"cpptraj_extract_{target_temperature}K.x")

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