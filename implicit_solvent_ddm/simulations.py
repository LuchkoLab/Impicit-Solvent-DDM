

import subprocess
import string
import os, os.path 

from string import Template 
#local imports 
from implicit_solvent_ddm.restraints import write_empty_restraint_file
from implicit_solvent_ddm.restraints import write_conformational_restraints
from implicit_solvent_ddm.alchemical import turn_off_charges 

def run_md(job, solute_file, solute_filename, solute_rst, solute_rst_filename, output_dir, argSet, message, work_dir=None, conformational_restraint = None, solvent_turned_off=None, charge_off = None, ligand_mask = None):
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

    tempDir = job.fileStore.getLocalTempDir()
    

    if charge_off != None: 
        temp_solute_filename = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
        rst = job.fileStore.readGlobalFile(solute_rst, userPath=os.path.join(tempDir, solute_rst_filename))

        charge_off_solute_filename = job.fileStore.importFile("file://" + turn_off_charges(temp_solute_filename, rst, ligand_mask))        
        solute = job.fileStore.readGlobalFile(charge_off_solute_filename,  userPath=os.path.join(tempDir, os.path.basename(charge_off_solute_filename)))
        
    else:
        solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
        rst = job.fileStore.readGlobalFile(solute_rst, userPath=os.path.join(tempDir, solute_rst_filename))

    if argSet["parameters"]["mdin"]:
        mdin_filename  = job.fileStore.importFile(os.path.join("file://" + work_dir, argSet["parameters"]["mdin"]))
        mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(argSet["parameters"]["mdin"])))
        
    else:
        mdin_filename = job.fileStore.importFile("file://" + make_mdin_file(conformational_restraint, solvent_turned_off))
        mdin = job.fileStore.readGlobalFile(mdin_filename, userPath=os.path.join(tempDir, os.path.basename(mdin_filename)))
        
    if conformational_restraint == None:
        restraint_file = job.fileStore.importFile("file://" + write_empty_restraint_file())
        restraint = job.fileStore.readGlobalFile( restraint_file, userPath=os.path.join(tempDir,'restraint.RST'))
             
        if not os.path.exists(output_dir + '/0'):
            output_dir = os.path.join(output_dir + '/0')
            os.makedirs(output_dir)

    
    if conformational_restraint != None:
        restraint_file = job.fileStore.importFile("file://" + write_conformational_restraints(solute_filename,conformational_restraint, work_dir))
        restraint_basename = os.path.basename(restraint_file)
        job.log('restraint_freeze_file : ' + str(restraint_file))
        restraint = job.fileStore.readGlobalFile(restraint_file, userPath=os.path.join(tempDir,'restraint.RST'))

        #make directory for specific conformational restraint force 
        if not os.path.exists(output_dir + '/' + str(conformational_restraint)):
            output_dir = os.path.join(output_dir + '/'+ str(conformational_restraint))
            os.makedirs(output_dir)

    if argSet["parameters"]["mpi"]:
        exe = argSet["parameters"]["executable"]
        np = str(argSet["parameters"]["mpi"])
        subprocess.check_call(["mpirun", "-np", np, exe, "-O", "-i", mdin, "-p", solute, "-c", rst])
    else: 
        exe = argSet["parameters"]["executable"]
        subprocess.check_call([exe, "-O", "-i", mdin, "-p", solute, "-c", rst])

    mdout_filename = "mdout"
    mdinfo_filename = "mdinfo"
    restrt_filename = "restrt"
    mdcrd_filename = "mdcrd"


    mdout_file = job.fileStore.writeGlobalFile(mdout_filename)
    mdinfo_file = job.fileStore.writeGlobalFile(mdinfo_filename)
    restrt_file = job.fileStore.writeGlobalFile(restrt_filename)
    mdcrd_file = job.fileStore.writeGlobalFile(mdcrd_filename)
    if conformational_restraint != None:
        job.log("running current conformational restraint value : " +str(conformational_restraint))
        job.log("conformational restraint mdout file: " + str(mdout_file))

    job.fileStore.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    job.fileStore.exportFile(mdinfo_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdinfo")))
    job.fileStore.exportFile(restrt_file, "file://" + os.path.abspath(os.path.join(output_dir,"restrt")))
    job.fileStore.exportFile(mdcrd_file, "file://" + os.path.abspath(os.path.join(output_dir,"mdcrd")))
    job.fileStore.exportFile(restraint, "file://" + os.path.abspath(os.path.join(output_dir,"restraint.RST")))
    job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_dir, str(os.path.basename(solute)))))

    if message == 'complex':
         
        return restrt_file, mdcrd_file

def initilized_jobs(job, work_dir):
    #job.log("Hello world, I have a message: {}".format(message))
    job.log(f'initialized job, the current working directory is {work_dir}')


def make_mdin_file(turn_on_conformational_rest=None, turn_off_solvent=None):
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

    mdin_path = os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)) + "/templates/mdgb.mdin")
    with open(mdin_path) as t:
        template = Template(t.read())
    
    if turn_on_conformational_rest == None:
        final_template = template.substitute(
            nstlim=1000,
            ntx=1,
            irest=0,
            dt=0.001,
            igb = 2,
            saltcon = 0.3,
            gbsa=0,
            temp0=298,
            ntpr=10,
            ntwx=10,
            cut=999,
            nmropt=1
            )


    if turn_on_conformational_rest != None and turn_off_solvent == None:
         final_template = template.substitute(
            nstlim=100,
            ntx=1,
            irest=0,
            dt=0.001,
            igb = 2,
            saltcon = 0.3,
            gbsa=0,
            temp0=298,
            ntpr=10,
            ntwx=10,
            cut=999,
            nmropt=1
            )
    
    if turn_on_conformational_rest != None and turn_off_solvent != None:
        final_template = template.substitute(
            nstlim=100,
            ntx=1,
            irest=0,
            dt=0.001,
            igb = 6,
            saltcon = 0.0,
            gbsa=0,
            temp0=298,
            ntpr=10,
            ntwx=10,
            cut=999,
            nmropt=1
            )
    with open('mdin', "w") as output:
        output.write(final_template)
    return os.path.abspath('mdin')
