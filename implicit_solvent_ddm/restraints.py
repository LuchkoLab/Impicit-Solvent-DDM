
#ask luchko about parmed, for dependencies/requirment 
import parmed as pmd 
import pytraj as pt 
import numpy as np
import re 
import glob
import os, os.path
from string import Template

#local imports 
import implicit_solvent_ddm.restraint_finder as findrest



def make_restraints_file(job, complex_file, complex_filename, complex_restrt, complex_restrt_filename, ligand_filename, receptor_filename, ligand_mask, receptor_mask,  restraint_type, work_dir):
    '''
    purpose to create a series of conformational and oritentational restratins template files.

   The orentational restraints will returns 6 atoms best suited for NMR restraints, based on specific restraint type the user specified. 

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        A context manager that represents a Toil workflow
    complex_file: toil.fileStore.FileID
        The jobStoreFileID of the imported file is an parameter file (.parm7) of a complex
    complex_filename: str
        Name of the parameter complex file
    complex_restrt: toil.fileStore.FileID
        The jobStoreFileID of the imported file. The file being an coordinate file (.ncrst, .nc) of a complex
    complex_restrt_filename: srt
        Name of the coordinate complex file
    argSet: dict
        A dictionary of parameters from a .yaml configuation file
    work_dir: str
        The absolute path to user's working directory

    Returns
    -------
    None
    '''
    tempDir = job.fileStore.getLocalTempDir()
    solute = job.fileStore.readGlobalFile(complex_file,  userPath=os.path.join(tempDir, complex_filename))
    rst = job.fileStore.readGlobalFile(complex_restrt[0],  userPath=os.path.join(tempDir, complex_restrt_filename))

    #make conformational restraint folders 
    if not os.path.exists(work_dir + '/mdgb/freeze_restraints_folder/ligand'):
        os.makedirs(work_dir + '/mdgb/freeze_restraints_folder/ligand')
    if not os.path.exists(work_dir + '/mdgb/freeze_restraints_folder/receptor'):
        os.makedirs(work_dir + '/mdgb/freeze_restraints_folder/receptor')
    if not os.path.exists(work_dir + '/mdgb/freeze_restraints_folder/complex'):
         os.makedirs(work_dir + '/mdgb/freeze_restraints_folder/complex')

    if not os.path.exists(work_dir + '/mdgb/orientational_restraints_folder/complex/'):
        os.makedirs(work_dir + '/mdgb/orientational_restraints_folder/complex/')
    
    #restraint_type = argSet["parameters"]["restraint_type"]
    #receptor_mask = argSet["parameters"]["receptor_mask"]
    #ligand_mask = argSet["parameters"]["ligand_mask"]
    
    
    traj = pt.load(rst, solute)
    #if argSet['residue_label']:
     #   pass
    #if argSet['complex_label']: 
      #  pass
 
    com_complex, num_atoms, ligand, com_ligand, receptor, com_receptor = map_molecule_parm(traj, receptor_mask, ligand_mask)
    
    job.log("\n parm7_file: " + complex_file)
    job.log(f"\n rest_ref_file:  + {complex_restrt}")
    job.log("\n num_atoms: "+str(num_atoms))
    job.log("\n ligand: "+str(ligand))
    job.log("\n com_ligand: "+str(com_ligand))
    job.log("\n receptor: "+str(receptor))
    job.log("\n com_receptor:"  + str(com_receptor))

    receptor_atom_neighbor_index = create_atom_neighbor_index(receptor.n_atoms, receptor)
    ligand_atom_neighbor_index = create_atom_neighbor_index(ligand.n_atoms, ligand)
    complex_atom_neighbor_index = create_atom_neighbor_index(traj.n_atoms, traj)
    
    
    complex_name = re.sub(r"\..*","",complex_filename)
    ligand_name =  re.sub(r"\..*","",ligand_filename)
    receptor_name = re.sub(r"\..*","",receptor_filename)

    write_freezing_restraints(receptor_atom_neighbor_index, ligand_atom_neighbor_index, complex_atom_neighbor_index, receptor.n_atoms, complex_name, ligand_name, receptor_name, work_dir)
    
    write_orientational_restraints(solute, complex_name, rst, restraint_type, work_dir)
   
def map_molecule_parm(traj, receptor_mask, ligand_mask):

    '''
    To map out the trajectory of the receptor and ligand separately 

    Also computes the center of mass for complex, receptor and ligand 

    Parameters
    ----------
    traj: pytraj.trajectory.trajectory.Trajectory
        The last trajectory frame from complex simualtion (state 9)
    receptor_mask: str
        An AMBER mask notation to select the receptor atoms only 
    ligand_mask: str
        An AMBER mask notation to select the ligand atoms only

    Returns
    -------
    com_complex: numpy.ndarray 
        compute center of mass for complex 
    num_atoms: int 
        Total number of atoms 
    ligand: pytraj.trajectory.trajectory.Trajectory
        Trajectory frame of the ligand only 
    com_ligand: numpy.ndarray
        center of mass for ligand 
    com_receptor: numpy.ndarray
        center of mass for receptor 
    '''
    #prmtop_df.columns = prmtop_df.columns.get_level_values(0)
    #ligand_label = prmtop_df['RESIDUE_LABEL'][1].rstrip()

    num_atoms = traj.n_atoms
    com_complex = pt.center_of_mass(traj)
    ligand = traj[ligand_mask]
    com_ligand = pt.center_of_mass(traj[ligand_mask])
    receptor = traj[receptor_mask]
    com_receptor = pt.center_of_mass(traj[receptor_mask])

    return com_complex, num_atoms, ligand, com_ligand, receptor, com_receptor

def create_atom_neighbor_index(num_atoms, molecule):
     '''                                                                                                                                   
     Conformational restraints will be applied by creating harmonic distance restraints between every atom and all neighbors within 6 Å that were part of the same molecule
     
     Parameters
     ----------
     num_atoms: int 
         number of atoms in the given molecule 
     molecule: pytraj.trajectory.trajectory.Trajectory
         Trajectory file for the given molecule 
    
    Returns
    -------
    atom_neighbor_array: list 
         A list of index values for nearest atom neighbor 
    '''

    #Radii to use when searching for neighboring atoms to restrain to
     atom_coords = molecule.xyz
     #conformational restraints between every atom and all neighbors within 6 angstrom 
     restraint_distance = 6
     current_atom = 0
     atom_neighbor_array = []
     while current_atom < num_atoms:
         current_neighbor = current_atom + 1
         neighbor_array = []
         while current_neighbor < num_atoms:
             if current_atom != current_neighbor:
                 atom_neighbor_distance = distance_calculator(atom_coords[0][current_atom],atom_coords[0][current_neighbor])
                 if atom_neighbor_distance <= restraint_distance:
                     neighbor_array = [current_atom + 1, current_neighbor + 1, atom_neighbor_distance]# '+1' translates from array index to \fortran/amber atom index, which starts at 1 not 0
                     atom_neighbor_array.append(neighbor_array)
             current_neighbor += 1
         current_atom += 1
     return atom_neighbor_array


def distance_calculator(point_a, point_b):

    '''
    Calcualtes the distance between two atoms

    Parameters
    ----------
    point_a: numpy.ndarray
        current atom coordinates 
    point_b: numpy.ndarray
        current neighbor coordinates 
   
    Returns
    -------
    distance: numpy.float64
        the distance between the two atoms 
    '''
    
    a_array = np.asarray(point_a, dtype = float)
    b_array = np.asarray(point_b, dtype = float)
    distance = np.linalg.norm(a_array - b_array)

    return distance

def write_freezing_restraints(receptor_freezing_restraints, ligand_freezing_restraints, complex_freezing_restraints, num_atoms_receptor, complex_name, ligand_name, receptor_name, work_dir):
    

    #restraint force constants are given as dummy_variable, actual values replace these when specified by mdgb.py
    for i in range(len(ligand_freezing_restraints)):
        lig_restraint_string = ('\n\n&rst iat = '+str(ligand_freezing_restraints[i][0])+', '+str(ligand_freezing_restraints[i][1])+',\n'
                                +'r1=0, r2 = '+str(ligand_freezing_restraints[i][2])+', r3 = '+str(ligand_freezing_restraints[i][2])+', r4 = 1000\n'
                                +'rk2=$frest, rk3=$frest,\n'
                                +'/\n'
                                +'\n'
                                )
        
       # file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_ligand_restraint.FRST", "a+")
        file = open(work_dir + "/mdgb/freeze_restraints_folder/ligand/"+ligand_name+"_restraint.FRST", "a+")
        file.write(lig_restraint_string)
        file.close()
        
        #This string contains the same values as the above, but its indeces are adjusted to match the complex file
        print('ligand_freezing_restraints[i][1+num_atoms_receptor]:', ligand_freezing_restraints[i][1]+num_atoms_receptor)
        print('ligand_freezing_restraints[i][0+num_atoms_receptor]:', ligand_freezing_restraints[i][0]+num_atoms_receptor)
        complex_ligand_residue_string = ('\n\n&rst iat = '+str(ligand_freezing_restraints[i][0]+num_atoms_receptor)+', '+str(ligand_freezing_restraints[i][1]+num_atoms_receptor)+',\n'
                                         +'r1=0, r2 = '+str(ligand_freezing_restraints[i][2])+', r3 = '+str(ligand_freezing_restraints[i][2])+', r4 = 1000\n'
                                         +'rk2=$frest, rk3=$frest,\n'
                                         +'/\n'
                                         +'\n'
                                         )

        #file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_complex_restraint.FRST", "a+")
        file = open(work_dir + "/mdgb/freeze_restraints_folder/complex/"+complex_name+"_restraint.FRST", "a+")
        file.write(complex_ligand_residue_string)
        file.close()

        
    #file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_ligand_restraint.FRST", "a+")
    file = open(work_dir + "/mdgb/freeze_restraints_folder/ligand/"+ ligand_name +"_restraint.FRST", "a+")
    end_string = ('&end\n')
    file.write(end_string)
    file.close()

    for k in range(len(receptor_freezing_restraints)):
        rec_restraint_string = ('\n\n&rst iat = '+str(receptor_freezing_restraints[k][0])+', '+str(receptor_freezing_restraints[k][1])+',\n'
                                +'r1=0, r2 = '+str(receptor_freezing_restraints[k][2])+', r3 = '+str(receptor_freezing_restraints[k][2])+',r4 = 1000\n'
                                +'rk2=$frest, rk3=$frest,\n'
                                +'/\n'
                                +'\n'
                                )


        #file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_receptor_restraint.FRST", "a+")
        file = open(work_dir + "/mdgb/freeze_restraints_folder/receptor/"+ receptor_name + "_restraint.FRST", "a+")
        file.write(rec_restraint_string)
        file.close()

        #file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_complex_restraint.FRST", "a+")
        file = open(work_dir + "/mdgb/freeze_restraints_folder/complex/"+complex_name +"_restraint.FRST", "a+")
        file.write(rec_restraint_string)
        file.close()

    #file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_receptor_restraint.FRST", "a+")
    file = open(work_dir + "/mdgb/freeze_restraints_folder/receptor/"+receptor_name+"_restraint.FRST", "a+")
    end_string = ('&end\n')
    file.write(end_string)
    file.close()

    #file = open(work_dir + "/mdgb/freeze_restraints_folder/"+name_of_system+"_complex_restraint.FRST", "a+")
    file = open(work_dir + "/mdgb/freeze_restraints_folder/complex/"+complex_name+"_restraint.FRST", "a+")
    end_string = ('&end\n')
    file.write(end_string)
    file.close()

def write_orientational_restraints(complex_file, complex_name, complex_coordinate, restraint_type, work_dir):

    '''
    To create an orientational restraint .RST file

    The function is an Toil function which will run the restraint_finder module and returns 6 atoms best suited for NMR restraints.

    Parameters
    ----------
    complex_file: toil.fileStore.FileID
        The jobStoreFileID of the imported file is an parameter file (.parm7) of a complex
    complex_name: str
        Name of the parameter complex file
    complex_coord: toil.fileStore.FileID
        The jobStoreFileID of the imported file. The file being an coordinate file (.ncrst, .nc) of a complex
    restraint_type: int
        The type of orientational restraints chosen.
    work_dir: str
        The absolute path to user's working directory 

    Returns
    -------
    None
    '''
     
    atom_R3, atom_R2, atom_R1, atom_L1, atom_L2, atom_L3, dist_rest, lig_angrest, rec_angrest, lig_torres, rec_torres, central_torres = findrest.remote_run_complex(complex_file, complex_coordinate, restraint_type)
    
    
    restraint_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/templates/restraint.RST")

    with open(restraint_path) as t:
        template = Template(t.read())
        restraint_template = template.substitute(
            atom_R3 = atom_R3,
            atom_R2 = atom_R2,
            atom_R1 = atom_R1,
            atom_L1 = atom_L1,
            atom_L2 = atom_L2,
            atom_L3 = atom_L3,
            dist_rest = dist_rest,
            lig_angrest = lig_angrest,
            rec_angrest = rec_angrest,
            central_torres = central_torres,
            rec_torres = rec_torres,
            lig_torres = lig_torres,
            drest= '$drest',
            arest = '$arest',
            trest = '$trest'
            )

    with open(work_dir + '/mdgb/orientational_restraints_folder/complex/'+ complex_name +'_orientational.RST', "a+") as output:
        output.write(restraint_template)

def write_empty_restraint_file():
    """
    Creates an empty restraint file in the case the no restraints are desired for a current run.

    Parameters
    ----------
    None

    Returns
    -------
    restraint.RST: str(file)
        absolute path of the created restraint file
        """
    #This function creates an empty restraint file in the case the no restraints are desired for a current run.
    #This function added by mbarton
    file = open("restraint.RST","w+")
    file.write("")
    file.close()
    return os.path.abspath("restraint.RST")

def write_restraint_forces(solute_filename, work_dir, conformational_restraint, orientational_restraint):


    #tempDir = job.fileStore.getLocalTempDir()
    current_toil_dir = os.getcwd()
    #solute = job.fileStore.readGlobalFile(solute_file, userPath=os.path.join(tempDir, solute_filename))
    #traj_file = job.fileStore.readGlobalFile(solute_traj_file, userPath=os.path.join(tempDir, solute_traj_filename))
   
    #restraint paths for orientational and conformational restraint templates 
    restraint_paths = []
    if conformational_restraint != None:
        restraint_paths.append(os.path.abspath(work_dir + '/mdgb/freeze_restraints_folder/'))
    if orientational_restraint != None:
        restraint_paths.append(os.path.abspath(work_dir + '/mdgb/orientational_restraints_folder/'))
    
    solu = re.sub(r"\..*", "", solute_filename)

    #os.chdir(restraint_path)
    #restraint template files 
    restraint_template_filenames = []
    for path in restraint_paths:
        os.chdir(path)
        for directory in os.listdir():
            os.chdir(directory)
            for file in glob.glob(solu + "*"):
                restraint_template_filenames.append(os.path.abspath(file))
            os.chdir('../')

    #for directory in os.listdir():
     #   os.chdir(directory)
      #  for file in glob.glob(solu + "*"):
       #     conformational_restraint_path = os.path.abspath(file)
        #os.chdir('../')
    
    #move back to toil temporary directory
    os.chdir(current_toil_dir)
    
    # con_basename = os.path.basename(conformational_restraint_path)
    # con_file = job.fileStore.importFile("file://" +conformational_restraint_path)
    
    #conformational_rest = job.fileStore.readGlobalFile(con_file, userPath=os.path.join(tempDir,con_basename))
    
    with open(restraint_template_filenames[0]) as t:
        template = Template(t.read())
        restraint_temp = template.substitute(
            frest = conformational_restraint
            )
   
    if orientational_restraint != None:
        with open(restraint_template_filenames[1]) as ot:
            orientational_temp = Template(ot.read())
            orientational_restraint_temp = orientational_temp.substitute(
              drest = conformational_restraint,
              arest = orientational_restraint,
              trest = orientational_restraint
            )
        with open('temp_restraint.RST', 'a+') as output:
            output.write(orientational_restraint_temp)

    with open('temp_restraint.RST', 'a+') as output:
        output.write(restraint_temp)
        #return os.path.abspath('restraint.RST')
    
    reading_file = open('temp_restraint.RST')
    new_file_content = ""
    for line in reading_file:
        new_line = line.replace("&end", "")
        new_file_content += new_line 
    reading_file.close()
    #write new restraint file 
    writing_file = open("restraint.RST", "w")
    writing_file.write(new_file_content)
    writing_file.write("&end")
    writing_file.close()
    return os.path.abspath('restraint.RST')