
import parmed as pmd
import pytraj as pt 
import os 
import re 


def split_complex(job, complex_topology_filename, complex_topology_basename, complex_coordinate_restart_filename, ligand_topology_basename, receptor_topology_basename, ligand_mask, receptor_mask, work_dir):

    '''
    Splitting a restart coordinate file of an complex into individual receptor and ligand topology and coordinate files 

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        context manager that represents a Toil workflow
    complex_topology_filename: str 
        complex parameter topology file absolute path 
    complex_topology_basename: str 
        basename for complex parameter topology file 
    complex_coordinate_restart_filename: str 
        absolute path for restart coordinate file of the complex
    ligand_topology_basename: str 
        basename for ligand parameter topology file 
    receptor_topology_basename: str 
        basename for receptor parameter topology file 
    ligand_mask: str 
        Amber mask which select ligand atoms from the complex
    receptor_mask: str 
        Amber mask which selects receptor atoms from the complex 
    work_dir: str 
        absolute working path 

    Returns
    -------
    ligand_traj_file: str
        jobStoreFileID representing ligand coordinate file in the file store
    receptor_traj_file: str 
        jobStoreFileID representing receptor coordinate file in the file store
    '''

    tempDir = job.fileStore.getLocalTempDir()
    complex_traj_filename = os.path.basename(complex_coordinate_restart_filename)
    solute = job.fileStore.readGlobalFile(complex_topology_filename, userPath=os.path.join(tempDir, complex_topology_basename))
    traj_file = job.fileStore.readGlobalFile(complex_coordinate_restart_filename, userPath=os.path.join(tempDir, complex_traj_filename))
   
    
    
    
    if not os.path.exists(work_dir + '/mdgb/split_complex_folder/ligand/'):
        os.makedirs(work_dir + '/mdgb/split_complex_folder/ligand/')
    
    if not os.path.exists(work_dir + '/mdgb/split_complex_folder/receptor/'):
        os.makedirs(work_dir + '/mdgb/split_complex_folder/receptor/')

    job.log("starting to split the complex")

    traj = pt.load(traj_file, solute)
    
    receptor = pt.strip(traj, ligand_mask)
    
    #receptor = traj[receptor_mask]

    job.log("the receptor trajectory is :" + str(receptor))

    ligand = pt.strip(traj, receptor_mask)
    #ligand = traj[ligand_mask]
    job.log("the ligand trajectory is: " + str(ligand))

    receptor_name = re.sub(r"\..*","",os.path.basename(receptor_topology_basename))
    
    ligand_name = re.sub(r"\..*","",os.path.basename(ligand_topology_basename))

    #pt.write_traj(work_dir + '/mdgb/split_complex_folder/ligand/'+ 'split_'+ ligand_name + '.ncrst', ligand, frame_indices=[ligand.n_frames-1], overwrite=True)
    pt.write_traj(work_dir + '/mdgb/split_complex_folder/ligand/'+ 'split_'+ ligand_name + '.ncrst', ligand, overwrite=True)
    pt.write_traj(work_dir + '/mdgb/split_complex_folder/receptor/'+ 'split_'+ receptor_name + '.ncrst', receptor, overwrite=True)

    ligand_path = os.path.join(work_dir + '/mdgb/split_complex_folder/ligand/'+ 'split_'+ ligand_name + '.ncrst.1')
    
    receptor_path = os.path.join(work_dir + '/mdgb/split_complex_folder/receptor/'+ 'split_' + receptor_name + '.ncrst.1')
    
    ligand_traj_file = job.fileStore.importFile( "File://"+ligand_path)
    receptor_traj_file =  job.fileStore.importFile( "File://"+receptor_path)

    job.log('writing global ligand file to jobstore: '+ str(ligand_traj_file))
    job.log('writing global rec file to jobstore: ' + str(receptor_traj_file))

    return ligand_traj_file, receptor_traj_file



    
def turn_off_charges(ligand_topology_filename, ligand_coordinate_filename, ligand_mask):

    '''
    Setting the atomic charge of every atom in a ligand parameter toplogy file to a zero charge

    Parameters
    ----------
    ligand_topology_filename: str
        absolute file path in job store to ligand parameter topology file 
    ligand_coorindate_filename: str 
        absolute file path in job store to ligand coordinate file 
    ligand_mask: str
        Amber mask which selects all ligand atoms from the complex
   
    Returns 
    -------
    ligand_zero_charge_topology_filename: str 
        absolute path to ligand topology file containing zero charge atoms   
    '''
    ligand_traj = pmd.load_file(ligand_topology_filename, xyz=ligand_coordinate_filename)
    pmd.tools.actions.change(ligand_traj, 'charge', ligand_mask, 0).execute()
    
    ligand_traj.save("charges_off_" + str(os.path.basename(ligand_topology_filename)))
    ligand_zero_charge_topology_filename = os.path.abspath("charges_off_" + str(os.path.basename(ligand_topology_filename)))

    return ligand_zero_charge_topology_filename

def alter_topology_file(solute_topology_filename, solute_coordinate_filename, ligand_mask, receptor_mask, turn_off_charges, add_exclusions):
    '''
    Altering the ligand charge to zero and non-bonded interactions with receptor w/ligand will not be computed.

    Parameters
    ----------
    solute_topology_filename: str
        absolute file path in job store to ligand parameter topology file 
    solute_coordinate_filename: str 
        absolute file path in job store to ligand coordinate file 
    ligand_mask: str
        Amber mask which selects all ligand atoms from the complex
    receptor_mask: str
        Amber mask which selects all recetor atoms from the complex 
    turn_off_charges: bool
        If True the atom charges within the ligand will be set to zero.
    add_exclusions: bool
        If True no non-bonded interactions between the ligand and receptor will be computed 
   
    Returns 
    -------
    solute_altered_filename: str 
        absolute path to ligand topology file containing all modified parameters   
    '''
    solute_traj = pmd.load_file(solute_topology_filename, xyz=solute_coordinate_filename)
    if turn_off_charges:
        pmd.tools.actions.change(solute_traj, 'charge', ligand_mask, 0).execute()
        saved_filename = "charges_off_"
    if add_exclusions:
        pmd.tools.actions.addExclusions(solute_traj, ligand_mask, receptor_mask)
        saved_filename = saved_filename + "exculsions_"
    
    solute_traj.save(saved_filename + str(os.path.basename(solute_topology_filename)))
    
    solute_altered_filename = os.path.abspath(saved_filename + str(os.path.basename(solute_topology_filename)))

    return solute_altered_filename
