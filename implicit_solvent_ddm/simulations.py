

from ast import arguments
from genericpath import exists
from platform import architecture
import subprocess
import string
import os, os.path 
import re
import sys
import yaml
import parmed as pmd 
from string import Template 
#local imports 
from implicit_solvent_ddm.restraints import write_empty_restraint_file
from implicit_solvent_ddm.restraints import write_restraint_forces
from implicit_solvent_ddm.toil_parser import get_output_dir
from implicit_solvent_ddm.alchemical import alter_topology_file
from implicit_solvent_ddm.remd import copy 
from implicit_solvent_ddm.remd import write_empty_restraint
from implicit_solvent_ddm.IO import export_outputs

def run_intermidate(job, solute_cordinate, config_args, intermidate_args):
    
    # job.log("this is the current job running {}".format(message))
    # job.log(f"input mdin {input_mdin}")
    tempDir = job.fileStore.getLocalTempDir()
    
    #make a global copy of topology and coordinate file 
    solute_file = intermidate_args["topology"]
    solute_filename = os.path.basename(solute_file)
    solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, os.path.basename(solute_file)))
    rst = job.fileStore.readGlobalFile(solute_cordinate[0], userPath=os.path.join(tempDir, os.path.basename(solute_cordinate[0])))
    #read in intermidate mdin 
    mdin_filename = job.fileStore.importFile("file://" + make_mdin_file(config_args, 
                                                                        intermidate_args["conformational_restraint"], intermidate_args["solvent_turned_off"]))
    mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
    
    if intermidate_args["charge_off"]:      
        altered_solute_filename = job.fileStore.importFile("file://" + alter_topology_file(solute, rst, config_args, intermidate_args))
        #solute = job.fileStore.readGlobalFile(charge_off_solute_filename,  userPath=os.path.join(tempDir, os.path.basename(charge_off_solute_filename)))
        solute = job.fileStore.readGlobalFile(altered_solute_filename, userPath=os.path.join(tempDir, os.path.basename(altered_solute_filename)))
    
    if intermidate_args["conformational_restraint"]:
        conformational_restraint = str(intermidate_args["conformational_restraint"]) 
        job.log("CONFORMATIONAL TURN ON")
        restraint_file = job.fileStore.importFile("file://" + write_restraint_forces(solute_filename, config_args["workDir"], intermidate_args["conformational_restraint"], intermidate_args["orientational_restraints"])) 
        restraint_basename = os.path.basename(restraint_file)
        job.log('restraint_freeze_file : ' + str(restraint_file))
        restraint = job.fileStore.readGlobalFile(restraint_file, userPath=os.path.join(tempDir, restraint_basename))
        #get output directory 
        output_dir = get_output_dir(intermidate_args["topology"],intermidate_args["state_label"], config_args["workDir"])
        job.log(f"output directory {output_dir}")
        if intermidate_args["orientational_restraints"] != None:
            conformational_restraint = intermidate_args["conformational_restraint"]
            orientational_restraint = intermidate_args["orientational_restraints"]            
            if not os.path.exists(output_dir + '/' + str(conformational_restraint) + '_' + str(orientational_restraint)):
                output_dir = os.path.join(output_dir + '/'+ str(conformational_restraint) + '_' + str(orientational_restraint))
                os.makedirs(output_dir)
        else:
        #make directory for specific conformational restraint force 
            if not os.path.exists(f"{output_dir}/{conformational_restraint}"):
                output_dir = os.path.join(f"{output_dir}/{conformational_restraint}")
                os.makedirs(output_dir)
                job.log(f"no orientaional restraints make output dir {output_dir}")
        
        
    files_in_current_directory = os.listdir(tempDir)  
    job.log(f"files before simulations {files_in_current_directory}")
    
    #name restart and trajectory files 
    solu = re.sub(r"\..*","",solute_filename)
    message = intermidate_args["message"]
    restart_filename = f"{message}_{solu}.rst7"
    trajectory_filename = f"{solu}.nc"
    
    run_args = {'solute': solute,
                'coordinate': rst,
                "restart_filename": restart_filename,
                "trajectory_filename": trajectory_filename,
                "mdin": mdin
                }
    submit_job(job, config_args, run_args)
 
    #export files 
    restrt_file,mdcrd_file = export_outputs(job, output_dir, files_in_current_directory)
    job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_dir, str(os.path.basename(solute)))))
    job.fileStore.exportFile(restraint, "file://" + os.path.abspath(os.path.join(output_dir,"restraint.RST")))

    return (restrt_file,mdcrd_file)
    
def run_md(job, solute_file, solute_filename, solute_rst, solute_rst_filename, output_dir, argSet, message, COM=False, input_mdin=None, work_dir= None, ligand_mask = None, receptor_mask = None, conformational_restraint = None, orientational_restraint = None, solvent_turned_off=False, charge_off = False, exculsions=False):
    """
    Locally run AMBER library engines for molecular dynamics

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        A context manager that represents a Toil workflow
    solute_file: toil.fileStore.FileID
        The jobStoreFileID of the imported file is an parameter file (ex: File://tmp/path/solute.parm7)
    solute_filename: str
        Name of solute file (ex: solute.parm7)
    solute_rst: toil.fileStore.FileID
        The jobStoreFileID of the imported file being a coordinate file (ex: File://tmp/path/solute.ncrst)
    solute_rst_filename: str
        Name of coordinate file (ex: solute.ncrst)
    mdin_file: toil.fileStore.FileID
        jobStoreFileID of an imported MD input file
    mdin_filename: str
        Name of MD file
    output_dir: str
        Absolute directory path where output files would be exported into
    state_label: int
        A flag to denote which state/step is currently being ran
    argSet: dict
        Dictionary of key:values obtained from a .yaml configuration file
    message: str
        A unique string that will denote the type of solute. Ex receptor, ligand or complex

    Returns
    -------
    restrt_file: toil.fileStore.FileID
        A jobStoreFileID of a restart file created after the completion of a MD run

    """
    job.log("this is the current job running {}".format(message))
    job.log(f"input mdin {input_mdin}")
    tempDir = job.fileStore.getLocalTempDir()
    if type(solute_rst_filename) == list:
        job.log(f'the list solute complex is {solute_rst_filename}')
        solute_rst_filename = os.path.basename(solute_rst_filename[0])

    if charge_off: 
        temp_solute_filename = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
        rst = job.fileStore.readGlobalFile(solute_rst[0], userPath=os.path.join(tempDir, solute_rst_filename))
        # import ligand parmtop into temporary directory 
        #charge_off_solute_filename = job.fileStore.importFile("file://" + turn_off_charges(temp_solute_filename, rst, ligand_mask))        
        altered_solute_filename = job.fileStore.importFile("file://" + alter_topology_file(temp_solute_filename, rst, ligand_mask, receptor_mask, charge_off, exculsions))
        #solute = job.fileStore.readGlobalFile(charge_off_solute_filename,  userPath=os.path.join(tempDir, os.path.basename(charge_off_solute_filename)))
        solute = job.fileStore.readGlobalFile(altered_solute_filename, userPath=os.path.join(tempDir, os.path.basename(altered_solute_filename)))
    else:
        solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
        rst = job.fileStore.readGlobalFile(solute_rst[0], userPath=os.path.join(tempDir, solute_rst_filename))

    if input_mdin:
        mdin_filename  = job.fileStore.importFile(os.path.join("file://" + work_dir, input_mdin))
        mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(input_mdin)))
        
    else:
        mdin_filename = job.fileStore.importFile("file://" + make_mdin_file(argSet, conformational_restraint, solvent_turned_off))
        mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
    
    flat_bottom = argSet["parameters"]["flat_bottom_restraints"][0]
    
    if COM:
        import_restraint = job.fileStore.importFile("file://" + copy(flat_bottom))
        
    elif conformational_restraint == None:
        import_restraint =  job.fileStore.importFile("file://" + write_empty_restraint(flat_bottom))
            
        # restraint_file = job.fileStore.importFile("file://" + write_empty_restraint_file())

        # restraint_basename = os.path.basename(restraint_file)
        # restraint = job.fileStore.readGlobalFile( restraint_file, userPath=os.path.join(tempDir, restraint_basename))
             
        if not os.path.exists(output_dir + '/0'):
            output_dir = os.path.join(output_dir + '/0')
            os.makedirs(output_dir)

    
    elif conformational_restraint != None:
        
        job.log("CONFORMATIONAL TURN ON")
        restraint_file = job.fileStore.importFile("file://" + write_restraint_forces(solute_filename, work_dir, conformational_restraint, orientational_restraint)) 
        restraint_basename = os.path.basename(restraint_file)
        job.log('restraint_freeze_file : ' + str(restraint_file))
        restraint = job.fileStore.readGlobalFile(restraint_file, userPath=os.path.join(tempDir, restraint_basename))
        
        if orientational_restraint != None:
            if not os.path.exists(output_dir + '/' + str(conformational_restraint) + '_' + str(orientational_restraint)):
                output_dir = os.path.join(output_dir + '/'+ str(conformational_restraint) + '_' + str(orientational_restraint))
                os.makedirs(output_dir)
        else:
        #make directory for specific conformational restraint force 
            if not os.path.exists(output_dir + '/' + str(conformational_restraint)):
                output_dir = os.path.join(output_dir + '/'+ str(conformational_restraint))
                os.makedirs(output_dir)
                job.log(f"no orientaional restraints make output dir {output_dir}")
        
        job.fileStore.exportFile(restraint, "file://" + os.path.abspath(os.path.join(output_dir,"restraint.RST")))
    
    files_in_current_directory = os.listdir(tempDir)  
    job.log(f"files before simulations {files_in_current_directory}")
    
    #name restart and trajectory files 
    solu = re.sub(r"\..*","",solute_filename)
    restart_filename = f"{message}_{solu}.rst7"
    trajectory_filename = f"{solu}.nc"
    
    run_args = {'solute': solute,
                'coordinate': rst,
                "restart_filename": restart_filename,
                "trajectory_filename": trajectory_filename,
                "mdin": mdin
                }
    submit_job(job, argSet, run_args)
    
    # if argSet["parameters"]["mpi"]:
    #     exe = argSet["parameters"]["executable"].split()
    #     exe.append()
    #     np = str(argSet["parameters"]["mpi"])
    #     output = subprocess.run(["srun", "-n", np, exe, "-O", "-i", mdin, 
    #                              "-p", solute, "-c", rst, 
    #                              "-r",restart_filename , "-", trajectory_filename], 
    #                             capture_output=True)
    # else: 
    #     exe = argSet["parameters"]["executable"]
    #     output = subprocess.run([exe, "-O", "-i", mdin, 
    #                              "-p", solute, "-c", rst,
    #                              "-r",restart_filename , "-x", trajectory_filename],
    #                             capture_output=True)
    restrt_file,mdcrd_file = export_outputs(job, output_dir, files_in_current_directory)

    
    # mdout_filename = "mdout"
    # mdinfo_filename = "mdinfo"
    # restrt_filename = "restrt"
    # mdcrd_filename = "mdcrd"


    # mdout_file = job.fileStore.writeGlobalFile(mdout_filename)
    # mdinfo_file = job.fileStore.writeGlobalFile(mdinfo_filename)
    # restrt_file = job.fileStore.writeGlobalFile(restrt_filename)
    # mdcrd_file = job.fileStore.writeGlobalFile(mdcrd_filename)
    
    # if conformational_restraint != None:
    #     job.log("running current conformational restraint value : " +str(conformational_restraint))
    #     job.log("conformational restraint mdout file: " + str(mdout_file))
    
    job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_dir, str(os.path.basename(solute)))))
    return (restrt_file,mdcrd_file)
    # #export all files 
    # job.fileStore.exportFile(mdin,"file://" + os.path.abspath(os.path.join(output_dir, "mdin")))
    # job.fileStore.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    # job.fileStore.exportFile(mdinfo_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdinfo")))
    # job.fileStore.exportFile(restrt_file, "file://" + os.path.abspath(os.path.join(output_dir,"restrt")))
    # job.fileStore.exportFile(mdcrd_file, "file://" + os.path.abspath(os.path.join(output_dir,"mdcrd")))
    


def initilized_jobs(job, work_dir):
    #job.log("Hello world, I have a message: {}".format(message))
    job.log(f'initialized job, the current working directory is {work_dir}')


def submit_job(job, arguments, run_arguments):
    
    system = pmd.load_file(run_arguments["solute"], run_arguments["coordinate"])
     
    argument_list = [
        "-O",
        "-i", run_arguments["mdin"],
        "-p", run_arguments["solute"], 
        "-c", run_arguments["coordinate"], 
        "-r",run_arguments["restart_filename"] , 
        "-x", run_arguments["trajectory_filename"]
    ]
    exe = arguments["parameters"]["executable"].split()
    
    if len(system.residues) > 1:
        #exe.append("-O")
        exe += argument_list
        #np = str(arguments["parameters"]["mpi"])
        output = subprocess.run(exe,capture_output=True)
    else:
        if bool(re.search(r"\.", arguments["parameters"]["executable"])): 
            serial_exe = [re.sub(r"\..*","",exe[-1])]
        else:
            serial_exe = exe
        #exe.append("-O")
        serial_exe += argument_list
        output = subprocess.run(serial_exe,capture_output=True)
    job.log(f"the output error {output}")
    
def make_mdin_file(arguments, turn_on_conformational_rest, turn_off_solvent):
    """ Creates an molecular dynamics input file

    Function will fill a template and write an MD input file

    Parameters
    ----------
    state_label: str
        This is a placeholder to denote which state of the MD cycle

    Returns
    -------
    mdin: str
        Absolute path where the MD input file was created.
    """
    user_inputs = arguments["parameters"]["mdin_intermidate_config"]
    mdin_path = os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)) + "/templates/mdgb.mdin")
    
    with open(mdin_path) as t:
        template = Template(t.read())
    
    if turn_off_solvent: 
        final_template = template.substitute(
            nstlim=user_inputs["nstlim"],
            ntx=1,
            irest=0,
            dt=user_inputs["dt"],
            igb = 6,
            saltcon = 0.0,
            rgbmax=user_inputs["rgbmax"],
            gbsa=user_inputs["gbsa"],
            temp0=user_inputs["temp0"],
            ntpr=user_inputs["ntpr"],
            ntwx=user_inputs["ntwx"],
            cut=user_inputs["cut"],
            ntc= user_inputs["ntc"],
            nmropt=1
            )
        
    else:
        final_template = template.substitute(
            nstlim=user_inputs["nstlim"],
            ntx=1,
            irest=0,
            dt=user_inputs["dt"],
            igb =user_inputs["igb"],
            saltcon =user_inputs["saltcon"],
            rgbmax=user_inputs["rgbmax"],
            gbsa=user_inputs["gbsa"],
            temp0=user_inputs["temp0"],
            ntpr=user_inputs["ntpr"],
            ntwx=user_inputs["ntwx"],
            cut=user_inputs["cut"],
            ntc= user_inputs["ntc"],
            nmropt=1
            )


    with open('mdin', "w") as output:
        output.write(final_template)
    return os.path.abspath('mdin')
