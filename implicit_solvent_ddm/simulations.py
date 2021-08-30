
from implicit_solvent_ddm.restraints import write_empty_restraint_file
import subprocess
import os, os.path 

def run_md(job, solute_file, solute_filename, solute_rst, solute_rst_filename, mdin_file, mdin_filename, output_dir, state_label, argSet,message):
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
    solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
    rst = job.fileStore.readGlobalFile(solute_rst, userPath=os.path.join(tempDir, solute_rst_filename))
    mdin = job.fileStore.readGlobalFile(mdin_file , userPath=os.path.join(tempDir, 'mdin'))
    if state_label == 9 or 2 or 7:
        restraint_file = job.fileStore.importFile("file://" + write_empty_restraint_file())
        restraint = job.fileStore.readGlobalFile( restraint_file, userPath=os.path.join(tempDir,'restraint.RST'))

    #if state_label == 3:
        # restraint_file =
    #subprocess.check_call(["mpirun", "-np","2","pmemd.MPI", "-O", "-i", mdin, "-p", solute, "-c", rst])
    subprocess.check_call(["pmemd", "-O", "-i", mdin, "-p", solute, "-c", rst])
    mdout_filename = "mdout"
    mdinfo_filename = "mdinfo"
    restrt_filename = "restrt"
    mdcrd_filename = "mdcrd"


    mdout_file = job.fileStore.writeGlobalFile(mdout_filename)
    mdinfo_file = job.fileStore.writeGlobalFile(mdinfo_filename)
    restrt_file = job.fileStore.writeGlobalFile(restrt_filename)
    mdcrd_file = job.fileStore.writeGlobalFile(mdcrd_filename)


    job.fileStore.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    job.fileStore.exportFile(mdinfo_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdinfo")))
    job.fileStore.exportFile(restrt_file, "file://" + os.path.abspath(os.path.join(output_dir,"restrt")))
    job.fileStore.exportFile(mdcrd_file, "file://" + os.path.abspath(os.path.join(output_dir,"mdcrd")))
    if message == 'complex':
         
        return restrt_file
