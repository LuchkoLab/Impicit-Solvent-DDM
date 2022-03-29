import os 
import re 
def export_outputs(toil_job, output_dir, files_to_ignore):
    toil_job.log(f"Exporting files to {output_dir} ")
    restart_files = []
    traj_files = [] 
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name in files_to_ignore:
                continue
            output_file = toil_job.fileStore.writeGlobalFile(name)
            toil_job.fileStore.exportFile(output_file,"file://" + os.path.abspath(os.path.join(output_dir, name)))
            if re.match(r".*.rst7.*", name):
                restart_files.append(str(output_file))
            if re.match(r".*.nc.*", name):
                traj_files.append(str(output_file))
    toil_job.log(f"the current trajectory files {traj_files}")
    toil_job.log(f"the restart files: {restart_files}")
    return restart_files, traj_files