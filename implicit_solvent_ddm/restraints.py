import itertools
import os
import re
import time
from string import Template

import numpy as np
import parmed as pmd
import pytraj as pt

from implicit_solvent_ddm.config import Config


def flat_bottom_restraints_template(host_guest_atoms, guest_atoms, flat_bottom_distances):
   
    restraint_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)) ,"templates/restraints/COM.restraint"))
    string_template = ""
    
    with open(restraint_path) as f:
        template = Template(f.read())
        
        restraint_template = template.substitute(
            host_atom_numbers = host_guest_atoms,
            guest_atom_numbers = guest_atoms, 
            r1 = flat_bottom_distances["r1"],
            r2 = flat_bottom_distances["r2"],
            r3 = flat_bottom_distances["r3"],
            r4 = flat_bottom_distances["r4"],
            rk2 = flat_bottom_distances["rk2"],
            rk3 = flat_bottom_distances["rk3"])
        
        string_template += restraint_template
    return string_template

def conformational_restraints_template(solute_conformational_restraint, num_receptor_atoms=0):
    
    restraint_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)) ,"templates/restraints/conformational_restraints.template"))
    string_template = ""
    for index in range(len(solute_conformational_restraint)):
        with open(restraint_path) as f:
            template = Template(f.read())
            
            restraint_template = template.substitute(
                solute_primary_atom = solute_conformational_restraint[index][0]+num_receptor_atoms,
                solute_sec_atom = solute_conformational_restraint[index][1]+num_receptor_atoms,
                distance = solute_conformational_restraint[index][2],
                frest = '$frest'
                )
            string_template += restraint_template
    return string_template

def orientational_restraints_template(atom_R3, atom_R2, atom_R1, atom_L1, atom_L2, atom_L3, dist_rest, 
                                      lig_angrest, rec_angrest, lig_torres, rec_torres, central_torres):
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
    
    string_template = ""
    restraint_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates/restraints/orientational.template"))
    # restraint_path = "/nas0/ayoub/Impicit-Solvent-DDM/implicit_solvent_ddm/templates/restraint.RST"
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
        string_template += restraint_template
    return string_template 

def write_empty_restraint(job):
    temp_dir = job.fileStore.getLocalTempDir()
    with open('empty.restraint', "w") as fn:
        fn.write("")
    return job.fileStore.writeGlobalFile('empty.restraint')


def fill_in_restraint_forces(job , config_object:Config, conformational_promise, orientational_promise):
    
    for (conformational_force, orientational_force) in zip(config_object.intermidate_args.conformational_restraints_forces, config_object.intermidate_args.orientational_restriant_forces):
        
        config_object.inputs[f"ligand_{conformational_force}_rst"] = job.addChildJobFn(write_restraint_forces, 
                                                                                       conformational_promise[1], 
                                                                                       conformational_force=conformational_force).rv()
          
        config_object.inputs[f"ligand_{conformational_force}_rst"] = job.addChildJobFn(write_restraint_forces,
                                                                                       conformational_promise[1], 
                                                                                       conformational_force=conformational_force).rv()
        config_object.inputs[f"receptor_{conformational_force}_rst"] = job.addChildJobFn(write_restraint_forces,
                                                                                         conformational_promise[2], 
                                                                                         conformational_force=conformational_force).rv()
        config_object.inputs[f"complex_{conformational_force}_{orientational_force}_rst"] = job.addChildJobFn(write_restraint_forces, 
                                                                                                              conformational_promise[0],
                                                                                                              orientational_promise,
                                                                                                              conformational_force=conformational_force,
                                                                                                              orientational_force=orientational_force).rv()
    

def write_restraint_forces(job, conformational_template, orientational_template=None, conformational_force=0.0, orientational_force=0.0):
    
    temp_dir = job.fileStore.getLocalTempDir()
    
    read_conformational_template = job.fileStore.readGlobalFile(conformational_template,  userPath=os.path.join(temp_dir, os.path.basename(conformational_template)))
    string_template = ""
    
    if orientational_template is not None:
        read_orientational_template =  job.fileStore.readGlobalFile(orientational_template,  
                                                                    userPath=os.path.join(temp_dir, os.path.basename(orientational_template)))
        with open(read_orientational_template) as oren_temp:
            template = Template(oren_temp.read())
            orientational_temp = template.substitute(
                drest = conformational_force,
                arest = orientational_force,
                trest = orientational_force)
            string_template += orientational_temp
            
    with open(read_conformational_template) as temp:
        template = Template(temp.read())
        restraint_temp = template.substitute(
            frest = conformational_force)
        string_template += restraint_temp
        
            
    string_template = string_template.replace("&end", "")
    
    with open('restraint.RST', "w") as fn:
        fn.write(string_template)
        fn.write("&end")
        
    return job.fileStore.writeGlobalFile('restraint.RST')
    
def get_flat_bottom_restraints(job, complex_prmtop, complex_coordinate, flat_well_parabola):
    
    tempDir = job.fileStore.getLocalTempDir()
    complex_prmtop_ID = job.fileStore.readGlobalFile(complex_prmtop,  userPath=os.path.join(tempDir, os.path.basename(complex_prmtop)))
    complex_coordinate_ID = job.fileStore.readGlobalFile(complex_coordinate,  userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])))
    
    complex_parmed_traj = pmd.load_file(complex_prmtop_ID, complex_coordinate_ID)
    host_atom_numbers = ','.join(
        [str(atom) for atom in range(1, int(complex_parmed_traj.parm_data["RESIDUE_POINTER"][1]))])
    guest_atom_numbers = ','.join(
        [str(i) for i in range(int(complex_parmed_traj.parm_data['RESIDUE_POINTER'][1]),
                                       int(complex_parmed_traj.parm_data['POINTERS'][0]+1))])
    flat_bottom_string = flat_bottom_restraints_template(host_atom_numbers, guest_atom_numbers, flat_well_parabola)
    
    temp_file = job.fileStore.getLocalTempFile()
    with open(temp_file, "w") as fH:
        fH.write(flat_bottom_string)
        
    return job.fileStore.writeGlobalFile(temp_file)

def get_conformational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask):
    
    '''
    Purpose to create a series of conformational restraint template files.

   The orentational restraints will returns 6 atoms best suited for NMR restraints, based on specific restraint type the user specified. 

    Parameters
    ----------
    job: toil.job.FunctionWrappingJob
        A context manager that represents a Toil workflow
    complex_coordinate: toil.fileStore.FileID
        The jobStoreFileID of the imported file. The file being an coordinate file (.ncrst, .nc) of a complex
    complex_restrt_filename: srt
        Name of the coordinate complex file
    

    Returns
    -------
    restraint_complex_ID: str 
        Conformational restraint template for the complex
    restraint_ligand_ID: str 
        Conformational restraint template for the ligand only 
    restriant_receptor_ID: str 
        Conformational restraint template for the receptor only 
    '''
    tempDir = job.fileStore.getLocalTempDir()
    complex_prmtop_ID = job.fileStore.readGlobalFile(complex_prmtop,  userPath=os.path.join(tempDir, os.path.basename(complex_prmtop)))
    complex_coordinate_ID = job.fileStore.readGlobalFile(complex_coordinate,  userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])))
    
    traj_complex = pt.load(complex_coordinate_ID, complex_prmtop_ID)
    ligand = traj_complex[ligand_mask]
    receptor = traj_complex[receptor_mask]
    
    receptor_atom_neighbor_index = new_create_atom_neighbor_index(receptor.n_atoms, receptor)
    ligand_atom_neighbor_index = new_create_atom_neighbor_index(ligand.n_atoms, ligand)
   
    ligand_template = conformational_restraints_template(ligand_atom_neighbor_index)
    receptor_template = conformational_restraints_template(receptor_atom_neighbor_index)
    complex_template = conformational_restraints_template(ligand_atom_neighbor_index, num_receptor_atoms=receptor.n_atoms)
    
   
    # Create a local temporary file.
    
    ligand_scratchFile = job.fileStore.getLocalTempFile()
    receptor_scratchFile = job.fileStore.getLocalTempFile()
    complex_scratchFile = job.fileStore.getLocalTempFile()
    # job.log(f"ligand_template {ligand_template}")
    with open(complex_scratchFile, "w") as fH:
        fH.write(complex_template)
        fH.write(receptor_template)
        fH.write('&end')  
    with open(ligand_scratchFile, "w") as fH:
        fH.write(ligand_template)
        fH.write('&end')
    with open(receptor_scratchFile, "w") as fH:
        fH.write(receptor_scratchFile)
        fH.write('&end')
        
    restraint_complex_ID = job.fileStore.writeGlobalFile(complex_scratchFile)
    restraint_ligand_ID = job.fileStore.writeGlobalFile(ligand_scratchFile)
    restriant_receptor_ID = job.fileStore.writeGlobalFile(receptor_scratchFile)
    
    #job.fileStore.export_file(restraint_complex_ID, "file://" + os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/output_directory", os.path.basename(restraint_complex_ID))))
    #toil.exportFile(outputFileID, "file://" + os.path.abspath(os.path.join(ioFileDirectory, "out.txt")))

    return (restraint_complex_ID, restraint_ligand_ID, restriant_receptor_ID)

def get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
    """
    Job to create orientational restraint file, based on user specified restraint type chosen within the config. 
    
    The orentational restraints will returns 6 atoms best suited for NMR restraints, based on specific restraint type the user specified. 
    restaint_type = 1 : Find atom closest to ligand's CoM and relevand information
    restaint_type = 2: Distance Restraints will be between CoM Ligand and closest heavy atom in receptor
    restraint_type = 3: Distance restraints will be between the two closest heavy atoms in the ligand and the receptor
    """
    
    tempDir = job.fileStore.getLocalTempDir()
    complex_prmtop_ID = job.fileStore.readGlobalFile(complex_prmtop,  userPath=os.path.join(tempDir, os.path.basename(complex_prmtop)))
    complex_coordinate_ID = job.fileStore.readGlobalFile(complex_coordinate,  userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])))
    
    
    traj_complex = pt.load(complex_coordinate_ID, complex_prmtop_ID)
    ligand = traj_complex[ligand_mask]
    receptor = traj_complex[receptor_mask]
    
    #center of mass 
    ligand_com = pt.center_of_mass(ligand)
    receptor_com = pt.center_of_mass(receptor)
    #Get ParmEd information
    parmed_traj = pmd.load_file(complex_prmtop_ID)
    
    if (restraint_type == 1):
        #find atom closest to ligand's CoM and relevand information
        ligand_suba1, lig_a1_coords, dist_liga1_com  = screen_for_distance_restraints(ligand.n_atoms,  ligand_com, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
        dist_liga1_com = distance_calculator(lig_a1_coords, ligand_com)
        receptor_a1, rec_a1_coords, dist_reca1_com = screen_for_distance_restraints(receptor.n_atoms,  receptor_com, receptor)
        dist_rest = distance_calculator(lig_a1_coords, rec_a1_coords) 
    
    elif (restraint_type == 2):  
        #find atom closest to ligand's CoM and relevand information                                                                                                                                                                                           
        ligand_suba1, lig_a1_coords, dist_liga1_com  = screen_for_distance_restraints(ligand.n_atoms,  ligand_com, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
        receptor_a1, rec_a1_coords, dist_rest = screen_for_distance_restraints(receptor.n_atoms, lig_a1_coords, receptor)
    
    elif (restraint_type == 3):   
        ligand_suba1, lig_a1_coords, receptor_a1, rec_a1_coords, dist_rest = screen_for_shortest_distant_restraint(receptor, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
    #find distance between CoM atoms for distance restraint
    else:
        raise RuntimeError("Invalid --r1 type input, must be 1,2 or 3 to choose type of restraint")

    ligand_suba2, ligand_atom2_name,  lig_a2_coords, dist_liga2_a3, dist_liga1_a2, lig_angle1, lig_angle2, lig_torsion, ligand_suba3, ligand_atom3_name, lig_a3_coords = screen_arrays_for_angle_restraints(lig_a1_coords, rec_a1_coords, ligand, parmed_traj, traj_complex)
    ligand_a2 = receptor.n_atoms + ligand_suba2
    ligand_a3 = receptor.n_atoms + ligand_suba3

    receptor_a2, receptor_atom2_name,  rec_a2_coords, dist_reca2_a3, dist_reca1_a2, rec_angle1, rec_angle2, rec_torsion, receptor_a3, receptor_atom3_name, rec_a3_coords = screen_arrays_for_angle_restraints(rec_a1_coords, lig_a1_coords, receptor, parmed_traj, traj_complex)

    
    #calculate torsion restraint inside receptor                                                                                                                                                                    
    central_torsion = wikicalculate_dihedral_angle(rec_a2_coords, rec_a1_coords, lig_a1_coords, lig_a2_coords)
    orientaional_conformational_template = orientational_restraints_template(receptor_a3, receptor_a2, receptor_a1, ligand_a1, ligand_a2, ligand_a3, dist_rest, lig_angle1, rec_angle1, lig_torsion, rec_torsion, central_torsion)
    
    complex_name = re.sub(r"\..*","",os.path.basename(complex_prmtop))
    
    with open(f"{complex_name}_orientational_template.RST", "w") as restraint_string:
        restraint_string.write(orientaional_conformational_template)
    
    return job.fileStore.writeGlobalFile(f"{complex_name}_orientational_template.RST")

def new_create_atom_neighbor_index(num_atoms, molecule):
    
    atom_coords = molecule.xyz
    atom_neighbor_array = []
    for current_atom in range(num_atoms):
        atom_distance = list(map(distance_calculator, atom_coords[0][current_atom:], itertools.repeat(atom_coords[0][current_atom], len(atom_coords[0][current_atom:]))))
        closest_neighbor = list(map(distance_filter, atom_distance))
        for neigbor_position, neighbors in enumerate(closest_neighbor): 
            if neighbors:
                neighbor_index = len(atom_coords[0]) - len(atom_coords[0][current_atom:]) + 1 
                atom_neighbor_array.append([current_atom + 1, neigbor_position + neighbor_index, atom_distance[neigbor_position]])
    return atom_neighbor_array

def distance_calculator(point_a, point_b):                                                                                                                                                                                   
    '''
    Function calcualtes the distance between two points
    '''                                  
    a_array = np.asarray(point_a, dtype = float)
    b_array = np.asarray(point_b, dtype = float)
    difference = np.subtract(a_array, b_array)
    return np.linalg.norm(difference)
    
def distance_filter(distance)->bool:
    '''
        Returns a boolean whether the atoms pairs are within 6 angstroms 
        
        Returns
        -------
            Returns True: if the distance does not equal 0 and is less than or equal to 6
            Returns False: if the distance equals to 0 or is greater than 6 
    '''
    #atom_neighbor_distance = distance_calculator(current_atom_coordinate, atom_neighbor_coordinate)
    #ignore if the atom is itself 
    return distance != 0 and distance <= 6 
    # if distance == 0 or distance > 6:
    #     return False 
    # else:
    #     return True



def screen_for_distance_restraints(num_atoms, com, mol):
    '''
    This function combs a molecule object (num_atoms) to find the atom closest to the coordinates at (com).  
    This (com) atom is traditionally an atom closest to the center of mass used as a distance restraint in NMR calculations in AMBER.
    It should however be re_named as it sometimes is another atom altogther.

    '''
    atom_coords = mol.xyz
    #calculate the distance betwen indivdual atoms and center of mass of solute 
    distances = list(map(distance_calculator, atom_coords[0], itertools.repeat(com, len(atom_coords[0]))))                                                                    
    #initialize from the initial atom 
    selected_atom_parmindex = 1
    shorest_distance = distances[0]
    selected_atom_name = mol.topology.atom(0).name
    selected_atom_position= atom_coords[0][0] 
    #cycle through all computed distances 
    for atom_index in range(num_atoms):
        atom_name = mol.topology.atom(atom_index).name
        if not atom_name.startswith('H'):
            if distances[atom_index] < shorest_distance:
                shorest_distance = distances[atom_index]
                selected_atom_parmindex = atom_index + 1  #Plus one alters the index intialized at 0 to match with parm id initialized at 1                                                                            
                selected_atom_name = atom_name
                selected_atom_position = atom_coords[0][atom_index] 
    print ('Success!! Selected distance atom: Type:', selected_atom_name, 'At coordinates: ', selected_atom_position)
    return selected_atom_parmindex, selected_atom_position, shorest_distance

def screen_for_shortest_distant_restraint(receptor_mol, ligand_mol):
    start_time = time.time()
    receptor_coords = receptor_mol.xyz
    ligand_coords = ligand_mol.xyz
    distances = []
    for atom in range(ligand_mol.n_atoms):
        distances += list(map(distance_calculator, receptor_coords[0], 
                              itertools.repeat(ligand_coords[0][atom], len(receptor_coords[0]))))
    shortest_distance = distances[0]
    
    #initial distance between primary atom of the ligand and Receptor 
    ligand_selected_atom_parmindex = 1
    receptor_selected_atom_parmindex = 1
    ligand_selected_atom_position = ligand_coords[0][0]
    receptor_selected_atom_position = receptor_coords[0][0]
    
    for index, distance in enumerate(distances):
        
        ligand_atom_index = int(index/receptor_mol.n_atoms) 
        receptor_atom_index = index % receptor_mol.n_atoms
        
        receptor_atom_name = receptor_mol.topology.atom(receptor_atom_index).name
        ligand_atom_name = ligand_mol.topology.atom(ligand_atom_index).name
        
        if not receptor_atom_name.startswith('H') and not ligand_atom_name.startswith('H'):
            if distance < shortest_distance:
                shortest_distance = distance 
                ligand_selected_atom_parmindex = ligand_atom_index+ 1  #Plus one alters the index intialized at 0 to match with parm id initialized at 1
                receptor_selected_atom_parmindex = receptor_atom_index + 1
                ligand_selected_atom_name = ligand_atom_name
                receptor_selected_atom_name = receptor_atom_name
                ligand_selected_atom_position = ligand_coords[0][ligand_atom_index]
                receptor_selected_atom_position = receptor_coords[0][receptor_atom_index]

    print("--- %s seconds ---" % (time.time() - start_time))
    
    return ligand_selected_atom_parmindex, ligand_selected_atom_position, receptor_selected_atom_parmindex, receptor_selected_atom_position, shortest_distance

#barton source code need to refactor 
def screen_arrays_for_angle_restraints(atom1_position, atomx_position, mol, parmed_traj, traj):
    '''
    This function screens the arrays created by create_DistanceAngle_array() to help find good matches.

    Args:
    num_atoms: Number of atoms in mol
    angleValues_array: An array of angle values who's index corresponds to distanceValues_array, 
    atom_id_array, 
    position_tethered_atom, 
    mol: Molecule Object with multipole attributes, coords, atom names, etc.
    

    '''

    print('')
    print("Testing New Function")

    #Initialize values
    i = 0
    j = 0
    min_angle = 80
    max_angle = 100
    num_atoms = mol.n_atoms
    atom_coords = mol.xyz
    atom2_position = []
    atom3_position = []
    current_distance_a1a2 = 0
    current_distance_a2a3 = 0
    #saved_distance_a1a2_value = 0
    #saved_distance_a2a3_value = 0
    saved_average_distance_value = 0
    #current_angle_a1a2 = 0
    #current_angle_a2a3 = 0
    #current_torsion_angle1 = 0
    #current_torsion_angle2 = 0
    #current_torsion_angle3 = 0
    success = 0
    k = 0

    while success == 0:
        #print('1')
        #print ('min_angle:', min_angle)
        #print ('max_angle:', max_angle)
        while i < num_atoms:
            j=0
            atom2_name = mol.topology.atom(i).name
            #print ('atom_name i;', mol.topology.atom(i).name)
            if atom2_name.startswith('H') == False:
                while j < num_atoms:
                    if i != j : 
                        atom3_name = mol.topology.atom(j).name
                        #print ('atom_name j;', mol.topology.atom(j).name)
                        if atom3_name.startswith('H') == False:
                            #print ('H success 2' )
                            atom2_position = atom_coords[0][i] #[position_data[i][1],position_data[i][2],position_data[i][3]]
                            atom3_position = atom_coords[0][j] #[position_data[i][1],position_data[i][2],position_data[i][3]]
                            angle_a1a2 = find_angle(atom2_position, atom1_position, atomx_position)
                            angle_a2a3 = find_angle(atom3_position, atom2_position, atom1_position) 
                            if angle_a1a2 > min_angle and angle_a1a2 < max_angle and angle_a2a3 > min_angle and angle_a2a3 < max_angle:
                                torsion_angle = wikicalculate_dihedral_angle(atomx_position, atom1_position, atom2_position, atom3_position)
                                num_heavy_bonds_on_atom_j = find_heavy_bonds(mol, j,parmed_traj, traj)
                                if num_heavy_bonds_on_atom_j > 1:
                                    new_distance_a1a2 = distance_calculator(atom1_position, atom2_position)
                                    new_distance_a2a3 = distance_calculator(atom2_position, atom3_position)
                                    new_distance_a3_norm_a1a2 = norm_distance(atom1_position, atom2_position, atom3_position)
                                    if (new_distance_a1a2 + new_distance_a3_norm_a1a2)/2 > saved_average_distance_value or success == 0:
                                        success = 1
                                        saved_distance_a1a2_value =  new_distance_a1a2
                                        saved_distance_a2a3_value =  new_distance_a2a3
                                        #The norm distance is the determining parameter for success, distance of p3 to the line connecting p1 and p2.
                                        # saved_distance_a2a3_value is only saved for informational purposes
                                        saved_a3norm_distance_value = (new_distance_a1a2 + new_distance_a3_norm_a1a2)/2
                                        saved_angle_a2a3 = angle_a2a3
                                        saved_angle_a1a2 = angle_a1a2
                                        saved_torsion_angle = torsion_angle
                                        saved_atom2_position = atom2_position
                                        saved_atom3_position = atom3_position
                                        selected_atom2 = i + 1
                                        selected_atom3 = j + 1
                                        selected_atom2_name = atom2_name
                                        selected_atom3_name = atom3_name    
                                    k += 1
                                

                    j += 1
                    #print ('j: ', j)
            i += 1
            #print ('i: ', i)
            
        if success == 1:
            print('Number of accepted cases: ', k)
        elif success == 0 and min_angle > 10:
            min_angle -= 1
            max_angle += 1
            print ('Widening acceptable angle: min_angle: ', min_angle, 'max_angle: ', max_angle)
            i = 0
            j = 0
        elif success == 0 and min_angle <=10:
            sys.exit("no suitable restraint atom found that fit all parameters!!")

    print ('Success!! Selected angle atom2: Type:', selected_atom2_name, 'At coordinates: ', saved_atom2_position)
    print ('Success!! Selected angle atom3: Type:', selected_atom3_name, 'At coordinates: ', saved_atom3_position)
    return selected_atom2, selected_atom2_name, saved_atom2_position, saved_distance_a2a3_value, saved_distance_a1a2_value, saved_angle_a1a2, saved_angle_a2a3, saved_torsion_angle, selected_atom3, selected_atom3_name, saved_atom3_position

def find_angle(point_a, point_b, point_c):
    '''
    Function calculates the angle created by two lines connectin 3 points, where point_b is the vertex.

    '''

    a = np.array(point_a, dtype = float)
    b = np.array(point_b, dtype = float)
    c = np.array(point_c, dtype = float)
    #difference 
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle_degrees =  np.degrees(angle)
    
    return angle_degrees
#barton source code 
def wikicalculate_dihedral_angle(atom1, atom2, atom3, atom4):

    '''                                                
    Function is one of two options to calculate dihedral angles

    Code for torsion restraints from:                                                                                                                                                                                                                                     
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python                                                                                                                                              
    WIKIPEDIA EXAMPLE
    '''
    p0 = np.array(atom1, dtype = float)
    p1 = np.array(atom2, dtype = float)
    p2 = np.array(atom3, dtype = float)
    p3 = np.array(atom4, dtype = float)

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    initial_angle = np.degrees(np.arctan2(y, x))
    if initial_angle < 0:
       final_angle = 360 + initial_angle
    else:
        final_angle = initial_angle


    return final_angle
#barton source code  
def find_heavy_bonds(mol, j, parmed_traj, traj):

   
    parmed_atom_j = parmed_traj.atoms[j]
   
    num_total_bonds = len(parmed_atom_j.bonds)

    num_heavy_bonds = 0
    bond_index = 0
    
    while bond_index < num_total_bonds:
        #print('bonded atom name: ', parmed_atom_j.bonds[bond_index].atom2.name)
        bonded_atom_name = parmed_atom_j.bonds[bond_index].atom2.name
        bond_index += 1
        
        if bonded_atom_name.startswith('H') == False:
            num_heavy_bonds += 1
      
    return num_heavy_bonds
#barton source code 
def norm_distance(point_a, point_b, point_c):
    p1=np.array(point_a)
    p2=np.array(point_b)
    p3=np.array(point_c)
    #x = p1-p2
    #dotproduct = np.dot(p3, x)/np.dot(x,x)
    #norm_distance = np.linalg.norm(dotproduct*(x)+p2-p3)

    #norm_distance = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)

    # normalized tangent vector
    d = np.divide(p2 - p1, np.linalg.norm(p2 - p1))

    # signed parallel distance components
    s = np.dot(p1 - p3, d)
    t = np.dot(p3 - p2, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p3 - p1, d)

    norm_distance = np.hypot(h, np.linalg.norm(c))
    
    #print('norm_distance: ', norm_distance)
    
    return norm_distance

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


#def workflow(job, prmtop, coordinate):
#     #get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
#     #get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
#     #output = job.addChildJobFn(get_orientational_restraints, prmtop, coordinate, ":CB7", ":M01", 1)
#     output = job.addChildJobFn(get_flat_bottom_restraints, prmtop, coordinate, {'r1': 0, "r2": 0, "r3": 10, "r4": 20, "rk2": 0.1, "rk3": 0.1})

if __name__ == "__main__":
    
    #traj = pt.load("/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/split_complex_folder/ligand/split_M01_000.ncrst.1", "/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/M01_000/4/4.0/M01_000.parm7")
    complex_coord = "/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/remd/cb7-mol01/cb7-mol01_300.0K_lastframe.ncrst"
    complex_parm = "/home/ayoub/nas0/Impicit-Solvent-DDM/structs/complex/cb7-mol01.parm7"
    #screen_for_distance_restraints(num_atoms, com, mol)
    traj = pt.load(complex_coord, complex_parm)
    receptor = traj[":CB7"]
    ligand = traj[":M01"]
    ligand_com = pt.center_of_mass(ligand)
    receptor_com = pt.center_of_mass(receptor)
    ligand_suba1, lig_a1_coords, dist_liga1_com  = screen_for_distance_restraints(ligand.n_atoms,  ligand_com, ligand)
    receptor_a1, rec_a1_coords, dist_reca1_com = screen_for_distance_restraints(receptor.n_atoms,  receptor_com, receptor)
    ligand_suba1, lig_a1_coords, receptor_a1, rec_a1_coords, dist_rest = screen_for_shortest_distant_restraint(receptor, ligand)
   
