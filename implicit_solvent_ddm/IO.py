import os 
import re 
def export_outputs(toil_job, output_dir, files_to_ignore):
    toil_job.log(f"Exporting files to {output_dir} ")
    restart_files = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name in files_to_ignore:
                continue
            output_file = toil_job.fileStore.writeGlobalFile(name)
            toil_job.fileStore.exportFile(output_file,"file://" + os.path.abspath(os.path.join(output_dir, name)))
            if re.match(r".*.rst.*", name):
                restart_files.append(str(output_file))
                
    return restart_files
    # if runtype == 'min':
    #     for arg in kwargs:
    #         output_filename = os.path.basename(kwargs[arg])
    #         output_file = toil_job.fileStore.writeGlobalFile(output_filename)
    #         toil_job.fileStore.exportFile(output_file,"file://" + os.path.abspath(os.path.join(output_dir, output_filename)))
    #         if arg == 'restart':
    #             coordinate_restart = output_file
    #     return coordinate_restart
    
    '''
    mdout_filename = "mdout"
    mdinfo_filename = "mdinfo"
    restrt_filename = "restrt"
    mdcrd_filename = "mdcrd"


    mdout_file = toil_job.fileStore.writeGlobalFile(mdout_filename)
    mdinfo_file = toil_job.fileStore.writeGlobalFile(mdinfo_filename)
    restrt_file = toil_job.fileStore.writeGlobalFile(restrt_filename)
    #mdcrd_file = toil_job.fileStore.writeGlobalFile(mdcrd_filename)
    
    #export all files 
    toil_job.fileStore.exportFile(mdin,"file://" + os.path.abspath(os.path.join(output_dir, os.path.basename(mdin))))
    toil_job.fileStore.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    toil_job.fileStore.exportFile(mdinfo_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdinfo")))
    toil_job.fileStore.exportFile(restrt_file, "file://" + os.path.abspath(os.path.join(output_dir,"restrt")))
    #toil_job.fileStore.exportFile(mdcrd_file, "file://" + os.path.abspath(os.path.join(output_dir,"mdcrd")))
    #job.fileStore.exportFile(restraint, "file://" + os.path.abspath(os.path.join(output_dir,"restraint.RST")))
    toil_job.fileStore.exportFile(solute, "file://" + os.path.abspath(os.path.join(output_dir, str(os.path.basename(solute)))))
    '''