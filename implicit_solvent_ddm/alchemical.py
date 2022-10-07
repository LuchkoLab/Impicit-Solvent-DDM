
import copy
import os
import re
from typing import Tuple

import parmed as pmd
import pytraj as pt


def split_complex_system(job, endstate_complex_parameter_filename, endstate_coordiante_complex_filename, ligand_mask, receptor_mask):
    '''
    Splitting a restart coordinate file of an complex into individual receptor and ligand topology and coordinate files 

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        context manager that represents a Toil workflow
    endstate_complex_parameter_filename: str 
        complex parameter topology file absolute path 
    endstate_coordiante_complex_filename: str 
        basename for complex parameter topology file 
    ligand_mask: str 
        Amber mask which select ligand atoms from the complex
    receptor_mask: str 
        Amber mask which selects receptor atoms from the complex 
 
    Returns
    -------
    receptor_ID: str
        A jobStoreFileID to access receptor_ID globally during the workflow 
    ligand_ID: str 
        A jobStoreFileID to access ligand_ID globally during the workflow 
    '''
    temp_dir = job.fileStore.getLocalTempDir()
    #read files into temporary directory 
    solute_file = job.fileStore.readGlobalFile(endstate_complex_parameter_filename, userPath= os.path.join(temp_dir, os.path.basename(endstate_complex_parameter_filename)))
    coordinate_file = job.fileStore.readGlobalFile(endstate_coordiante_complex_filename, userPath=os.path.join(temp_dir, os.path.basename(endstate_coordiante_complex_filename)))
    
    #load in endstate lastframe coordinate of complex file 
    traj = pt.load(coordinate_file, solute_file)
    #strip into receptor and ligand respectivlity 
    receptor = pt.strip(traj, ligand_mask)
    ligand = pt.strip(traj, receptor_mask)
    
    receptor_name = receptor_mask.strip(":")
    ligand_name = ligand_mask.strip(":")
    #write files into temporary directory 
    pt.write_traj(f"{temp_dir}/split_{receptor_name}.ncrst", receptor, overwrite=True)
    pt.write_traj(f"{temp_dir}/split_{ligand_name}.ncrst", ligand, overwrite=True)
    #write global copies of receptor and ligand coordiante files into job-store 
    receptor_ID = job.fileStore.writeGlobalFile(f"{temp_dir}/split_{receptor_name}.ncrst.1")
    ligand_ID = job.fileStore.writeGlobalFile(f"{temp_dir}/split_{ligand_name}.ncrst.1")
    
    return receptor_ID, ligand_ID

def get_intermidate_parameter_files(job, complex_prmtop, complex_coordinate, ligand_mask, receptor_mask)->Tuple[str, str, str]:
    '''
    Altering the ligand charge to zero and non-bonded interactions with receptor w/ligand will not be computed.

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        A context manager that represents a Toil workflow
    complex_prmtop: toil.fileStores.FileID
        file path to job store of a complex parameter file
    complex_coordinate: toil.fileStores.FileID
        file path to job store of a complex coordinate file
    ligand_mask: str
        Amber mask which selects all ligand atoms from the complex
    receptor_mask: str
        Amber mask which selects all recetor atoms from the complex 
        
    Returns 
    -------
    ligand_no_vdw_ID: toil.fileStore.FileID
        Upload a ligand system with no VDW/electrostatic interactions. 
    ligand_no_vdw_charge_parm_ID: toil.fileStores.FileID 
        Upload a ligand system  with no VDW/electrostatic interactions and charge = 0
    receptor_no_vdw_ID: toil.fileStores.FileID 
         Upload a receptor system with no VDW/electrostatic interactions. 
    complex_ligand_no_charge_ID: toil.fileStores.FileID
        Upload complex_ligand_no_charge_ID to the job store. 
    complex_no_ligand_interaction_ID: toil.fileStores.FileID
        Upload complex_no_ligand_interaction_ID to the job store. 
    '''
    temp_dir = job.fileStore.getLocalTempDir()
    read_complex_prmtop = job.fileStore.readGlobalFile(complex_prmtop, userPath= os.path.join(temp_dir, os.path.basename(complex_prmtop)))
    read_complex_coordiate = job.fileStore.readGlobalFile(complex_coordinate, userPath= os.path.join(temp_dir, os.path.basename(complex_coordinate)))
    complex_traj = pmd.load_file(read_complex_prmtop, read_complex_coordiate)
    ligand_traj = complex_traj[ligand_mask]
    
    
    ligand_no_charge_parm_ID = job.fileStore.writeGlobalFile(alter_topology(ligand_traj, ligand_mask, receptor_mask, no_charge=True))
    complex_ligand_no_charge_ID = job.fileStore.writeGlobalFile(alter_topology(complex_traj, ligand_mask, receptor_mask, no_charge=True))
    complex_no_ligand_interaction_ID = job.fileStore.writeGlobalFile(alter_topology(complex_traj, ligand_mask, receptor_mask, no_charge=True, exculsions=True))
    
    
    #job.fileStore.export_file(complex_no_ligand_interaction_ID, "file://" + os.path.abspath(os.path.join('/home/ayoub/nas0/Impicit-Solvent-DDM/output_directory', os.path.basename(complex_no_ligand_interaction_ID))))
    
    return (ligand_no_charge_parm_ID, complex_ligand_no_charge_ID, complex_no_ligand_interaction_ID)

def alter_topology(solute_amber_parm, ligand_mask, receptor_mask, no_charge=False, exculsions=False)-> str:
    '''
    Altering the ligand charge to zero and non-bonded interactions with receptor w/ligand will not be computed.

    Parameters
    ----------
    solute_parm: parmed.amber._amberparm.AmberParm
        Parmed parameter file  
    ligand_mask: str
        Amber mask which selects all ligand atoms from the complex
    receptor_mask: str
        Amber mask which selects all recetor atoms from the complex 
    no_charge: bool
        If True the atom charges within the ligand will be set to zero.
    exculsions: bool
        If True no non-bonded interactions between the ligand and receptor will be computed 
   
    Returns 
    -------
    solute_altered_filename: str 
        absolute path to ligand topology file containing all modified parameters   
    '''
    solute_parm = copy.deepcopy(solute_amber_parm)
    saved_filename = ""
    if no_charge:
        pmd.tools.actions.change(solute_parm, 'charge', ligand_mask, 0).execute()
        saved_filename += "charges_off_"
    if exculsions:
        pmd.tools.actions.addExclusions(solute_parm, ligand_mask, receptor_mask).execute()
        saved_filename += "exclusions_"
    if solute_parm.name is not None:
        saved_filename += os.path.basename(solute_parm.name)
    else:
        saved_filename += f"{ligand_mask.strip(':')}.parm7"
    solute_parm.save(saved_filename)
    
    return os.path.abspath(saved_filename)

