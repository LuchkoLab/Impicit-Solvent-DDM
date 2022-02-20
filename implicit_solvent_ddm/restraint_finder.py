import re
import math
import sys
import numpy as np
import argparse
import pytraj as pt
import parmed as pmd


'''
restraint_finder.py

This script is meant to identify 6 atoms best suited for NMR restraints in AMBER.

1: Distance restraints are selected within the ligand and receptor by finding the atom closest to their respective CoM's
2,3: Angle Restraints are selected by finding the furthest atom within the respective ligand/recpetor that makes an angle with the distance retraint between 45 and 135 degrees
4,5,6: Torsion Restraints are selected by finding the furthest atom within the respective ligand/recpetor that makes an angle with the angles retraints between 45 and 135 degrees  

Notes:
It will need a version of AMBER loaded that is compatible with Anaconda, have been working mostly with AMBER 18
Anaconda must be mounted

Running:
If run from the command line simply using python restraint_finder.py --p (parm) --c (coord), the script will print to screen all the data regarding the restraints it selects and will also plot what they look like in Mathplotlib 3D.

Calling as a function:
To call this script and simply have it return restraints, simply call the function remote_run(coord, parm).
This also avoids the printing to screen and the plotting.

'''



def distance_calculator(point_a, point_b):                                                                                                                                                                                   
    '''
    Function calcualtes the distance between two points
    '''
                                      
    a_array = np.asarray(point_a, dtype = float)
    b_array = np.asarray(point_b, dtype = float)
    distance = np.linalg.norm(a_array - b_array)
 
    return distance

def screen_for_shortest_distant_restraint(receptor_num_atoms, ligand_num_atoms, receptor_mol, ligand_mol):
    '''
    This function combs through two molecules and finds the two heavy atoms closest to each other
    '''

    i = 0
    j = 0
    receptor_coords = receptor_mol.xyz
    ligand_coords = ligand_mol.xyz
   
    ligand_position = ligand_coords[0][0]
    receptor_position = receptor_coords[0][0]
   
    current_distance = distance_calculator(ligand_position, receptor_position)
    saved_distance_value = current_distance
    selected_ligand_parmindex = 1
    selected_receptor_parmindex = 1
    #atom_name = mol.topology.atom.name                                                                                                                                                                                            
    #atom = mol.topology.atom                                                                                                                                                                                                      
    #print ('atom, atomname: ', atom, atom_name)                                                                                                                                                                                   
    #selected_atom = i                                                                                                                                                                                                             
    ligand_selected_position = ligand_coords[0][0]
    receptor_selected_position = receptor_coords[0][0]
    #selected_atom_name = mol.topology.atom(i).name                                                                                                                                                                                
    ligand_atom_name = ligand_mol.topology.atom(i).name
    ligand_selected_atom_name = ligand_atom_name
    ligand_atom = ligand_mol.topology.atom(i)
    receptor_atom_name = receptor_mol.topology.atom(i).name
    receptor_selected_atom_name = receptor_atom_name
    receptor_atom = receptor_mol.topology.atom(i)
    
    #print ('atom, atomname: ', atom, atom_name)                                                                                                                                                                                   
    #print ('num_atoms: ', num_atoms)                                                                                                                                                                                              
    while i < ligand_num_atoms:
        j = 0
        ligand_position = ligand_coords[0][i]
        ligand_atom_name = ligand_mol.topology.atom(i).name
        while j < receptor_num_atoms:
            receptor_position = receptor_coords[0][j]
            receptor_atom_name = receptor_mol.topology.atom(j).name
        #dist = distance_calculator(com, atom_position)                                                                                                                                                                            
            new_distance = distance_calculator(ligand_position, receptor_position)
        #print ('Evaluating Atom: ', i, ', Position :', atom_position, ', Distance from CoM: ', new_distance)                                                                                                                     
            #print ('Current retainted minimum distance: ', saved_distance_value, ' at atom: ', ligand_selected_atom, receptor_selected_atom)                                                                                
            #print('success 1', i, j, receptor_atom_name, ligand_atom_name, new_distance, saved_distance_value) 
            if new_distance < saved_distance_value: # and atom_name.startswith != 'H':
                #print('success 2')
                if receptor_atom_name.startswith('H') == False and ligand_atom_name.startswith('H') == False:
                    #print('success 3')
                    #print('success 1', i, j, receptor_atom_name, ligand_atom_name, new_distance)
                    saved_distance_value = new_distance
                    ligand_selected_atom_parmindex = i + 1  #Plus one alters the index intialized at 0 to match with parm id initialized at 1                                                                                     
                    receptor_selected_atom_parmindex = j + 1
                    ligand_selected_atom_name = ligand_atom_name
                    receptor_selected_atom_name = receptor_atom_name
                    ligand_selected_atom_position = ligand_position
                    receptor_selected_atom_position = receptor_position
            #print ('Current retainted minimum distance: ', saved_distance_value, ' at index: ', selected_atom)                                                                                                                
            j += 1
            
        i += 1

    print ('saved_distance_value: ', saved_distance_value)
    print ('Success!! Selected L1: Type:', ligand_selected_atom_name, 'At coordinates: ', ligand_selected_atom_position)
    print ('Success!! Selected R1: Type:', receptor_selected_atom_name, 'At coordinates: ', receptor_selected_atom_position)
    return ligand_selected_atom_parmindex, ligand_selected_atom_position, receptor_selected_atom_parmindex, receptor_selected_atom_position, saved_distance_value


def screen_for_angle_restraints(num_atoms, position_tether_atom, angle_tether_atom, mol):    
    '''
    num_atoms
    postion_tether_atom = atom to which this function will tether by distance/position.  THIS IS NOT NECESSARILY THE DISTANCE RESTRAINT ATOM.  It is the atom ONE step removed from the current selected atom. 
    angle_tether_atom = an atom which is already tethered to the poition_tether_atom by distannce.  Will be used to calculate an angled between position_tether, angle_tether, and selected atom. THIS IS NOT NECESSARILY THE ANGLE RESTRAINT ATOM.  It is the atom TWO steps removed by tether from teh current selected atom.
    mol = current molecule object

    '''
    
    #initialize
    i = 0
    atom_coords = mol.xyz
    atom_position = []
    #print ('Molecule.xyz: ', atom_coords)
    current_atom_position = atom_coords[0][0]
    #print ('atom position 1: ', current_atom_position)
    #atom_position = [float(position_data[0][1]),float(position_data[0][2]),float(position_data[0][3])]
    current_distance = 0 #distance_calculator(position_tether_atom, current_atom_position)
    saved_distance_value = 0
    #selected_atom = i
    #selected_atom_position = i
    #selected_atom_name = mol.topology.atom(i).name
    atom_name = mol.topology.atom(i).name
    atom = mol.topology.atom(i)
    angle = 0.0
    min_angle = 55
    max_angle = 125
    #selected_angle = angle
    success = 0
    k=0

    while success == 0:
        #print ('min_angle:', min_angle)
        #print ('max_angle:', max_angle)
        while i < num_atoms:
        #print ('pos data:', position_data[i], position_data[i][2])
            atom_position = atom_coords[0][i] #[position_data[i][1],position_data[i][2],position_data[i][3]]
            new_distance = distance_calculator(position_tether_atom, atom_position)
            atom_name = mol.topology.atom(i).name
        #print ('Evaluating Atom: ', i, ', Position :', atom_position, ', Distance from pos_tether: ', new_distance)
        #print ('Current retainted maximmum distance: ', saved_distance_value, ' at atom: ', selected_atom)                                                                              
            if new_distance > saved_distance_value:
            #print ('SUCCESS! Loop2')
            #print ('startswith: ', atom_name.startswith('H')
                angle = find_angle(atom_position, position_tether_atom, angle_tether_atom)
                #print ("angle: ", angle)
                if angle > min_angle and angle < max_angle:
                    #print ('SUCCESS! Loop 3')
                    #print ('ATOM: ', atom_name)
                    if atom_name.startswith('H') == False:
                        #print ('SUCCESS! Loop 4')
                        success = 1
                        selected_atom_name = atom_name
                        saved_distance_value = new_distance
                        selected_atom = i + 1 #Plus one alters the index intialized at 0 to match with parm id initialized at 1   
                        selected_atom_position = atom_position
                        selected_angle = angle
            i += 1
        if success == 0:
            sys.exit("no suitable restraint atom found!!")
            
        '''
        if success == 0:
            min_angle -= 5
            max_angle += 5
            print("no suitable atom found, widening search angle restrictions to: ", min_angle, '- ', max_angle, 'degrees, rescanning...')
            i = 0
            if min_angle < 0:
                sys.exit("no suitable restraint atom found!!")
        '''

    print ('Success!! Selected angle atom: Type:', selected_atom_name, 'At coordinates: ', selected_atom_position)
    return selected_atom, selected_atom_position, saved_distance_value, selected_angle

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

       


def create_DistanceAngle_array(num_atoms, position_tether_atom, angle_tether_atom, mol):
    i=0
    distanceValues_array = []
    angleValues_array = []
    atom_coords = mol.xyz
    atom_cooridinates_array = []
    atom_position = []
    id_array = []

    while i < num_atoms:
        atom_position = atom_coords[0][i] #[position_data[i][1],position_data[i][2],position_data[i][3]]
        new_distance = distance_calculator(position_tether_atom, atom_position)
        atom_name = mol.topology.atom(i).name
        new_angle = find_angle(atom_position, position_tether_atom, angle_tether_atom)
        distanceValues_array.append(new_distance)
        angleValues_array.append(new_angle)
        #id_array.append(i+1)
        atom_cooridinates_array.append(atom_coords[0][i])

        #print('Size of distance array:', len(distanceValues_array))
        #print('Size of id array:', len(id_array))
        #print('Size of angle array:', len(angleValues_array))
        i += 1

    return angleValues_array, distanceValues_array, atom_cooridinates_array #id_array
                

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

def calculate_dihedral_angle(atom1, atom2, atom3, atom4):

    '''
    Function is one of two options to calculate dihedral angles  

    Code for torsion restraints from:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    '''

    p0 = np.array(atom1, dtype = float)
    p1 = np.array(atom2, dtype = float)
    p2 = np.array(atom3, dtype = float)
    p3 = np.array(atom4, dtype = float)

    #define 3 vectors between 4 points
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def find_angle(point_a, point_b, point_c):
    '''
    Function calculates the angle created by two lines connectin 3 points, where point_b is the vertex.

    '''

    a = np.array(point_a, dtype = float)
    b = np.array(point_b, dtype = float)
    c = np.array(point_c, dtype = float)
    
    #print (a)
    #print (b)
    #print (c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    angle_degrees =  np.degrees(angle)
    return angle_degrees

def screen_for_distance_restraints(num_atoms, com, mol):
    '''
    This function combs a molecule object (num_atoms) to find the atom closest to the coordinates at (com).  
    This (com) atom is traditionally an atom closest to the center of mass used as a distance restraint in NMR calculations in AMBER.
    It should however be re_named as it sometimes is another atom altogther.

    '''
    

    i = 0
    atom_coords = mol.xyz
    #print ('Molecule.xyz: ', atom_coords)
    atom_position = atom_coords[0][0]
    #selected_atom_position = atom_position
    #print ('atom position 1: ', atom_position)
    current_distance = distance_calculator(com, atom_position)                                                                                
    saved_distance_value = current_distance    
    selected_atom_parmindex = 1
    #atom_name = mol.topology.atom.name
    #atom = mol.topology.atom
    #print ('atom, atomname: ', atom, atom_name)
    #selected_atom = i
    selected_atom_position = atom_coords[0][0]
    #selected_atom_name = mol.topology.atom(i).name
    atom_name = mol.topology.atom(i).name
    selected_atom_name = atom_name
    atom = mol.topology.atom(i)
    #print ('atom, atomname: ', atom, atom_name)
    #print ('num_atoms: ', num_atoms)
    while i < num_atoms:
        atom_position = atom_coords[0][i]
        atom_name = mol.topology.atom(i).name
        #dist = distance_calculator(com, atom_position)
        new_distance = distance_calculator(com, atom_position)
        #print ('Evaluating Atom: ', i, ', Position :', atom_position, ', Distance from CoM: ', new_distance)                                                         
        #print ('Current retainted minimum distance: ', saved_distance_value, ' at atom: ', selected_atom)  
        if new_distance < saved_distance_value: # and atom_name.startswith != 'H': 
            if atom_name.startswith('H') == False:
                saved_distance_value = new_distance                                                                                                                     
                selected_atom_parmindex = i + 1  #Plus one alters the index intialized at 0 to match with parm id initialized at 1                                                                            
                selected_atom_name = atom_name
                selected_atom_position = atom_position 
            #print ('Current retainted minimum distance: ', saved_distance_value, ' at index: ', selected_atom)
        i += 1
    
    print ('Success!! Selected distance atom: Type:', selected_atom_name, 'At coordinates: ', selected_atom_position)
    return selected_atom_parmindex, selected_atom_position, saved_distance_value

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

def count_atoms(inputfile, lineflag):
    '''
    This function is not used in the script, it is a hold over from the original version that read in pqr's rather than parm7's and rst7's
    
    This function inistializes num_atoms by counting the number of atoms in the external file to be evaluated.                                                                                                                                                                                                                        
    Args:                                                                                                                                                            
        inputfile: (string) The name and path of the external file for the functino to evaluate.                                                                     
        lineflag: (string) The string at the head of the .pqr files line that determines weather than line is part of the evaluation.                                                                                                                                                                                     
    Returns:                                                                                                                                                         
        atoms: Total number of atoms in the molecule/file to be evaluated by this program.                                                                               '''

    atoms = 0
    with open(inputfile, 'r') as pqr:
       for line in pqr:
           if line.startswith(lineflag):
               atoms += 1
    return atoms


def initialize_x(position_data, inputfile, lineflag):
    '''                                                                                                                                                               
    Function reads in data from .pqr file to charge_data and position_data objects                                                                                    
    '''
    with open(inputfile, 'r') as pqr:
        for line in pqr:
            if line.startswith(lineflag):
                recordname, serial, atomname, residuename, residuenumber, x ,y, z, charge, radius , residue = line.split()
                atom_tag = int(serial)
                #print ('atom_tag: ', atom_tag)
                current_coord = [serial, x, y, z]
                #current_charge = charge
                if recordname == 'TER':
                    print (recordname, serial)
                position_data.append (current_coord)
                #charge_data.append (current_charge)

def find_mol_cm(position_data, num_atoms, inputfile, lineflag):
    '''
    This function is not used in the script, it is a hold over from the original version that read in pqr's rather than parm7's and rst7's   

    Finds the center of mass of a molecule.  Assumes only atoms H, O, C, N, P, S, MG.

    Args:
        position_data: xyz coordinates of every atom in the molecule being evaluated

    Retruns:
        center_of_mass:  The center of geometry as the best approximation of center of mass.

    '''

    #Populate atom mass data set
    mass_data = []
    i = 1
    with open(inputfile, 'r') as pqr:
        for line in pqr:
            if line.startswith(lineflag):
                recordname, serial, atomname, residuename, residuenumber, x ,y, z, charge, radius , residue = line.split()
            if atomname.startswith('H'):
                mass_data.append (1.008)
            elif atomname.startswith('O'):
                mass_data.append (16.00)
            elif atomname.startswith('C'):
                mass_data.append (12.01)
            elif atomname.startswith('N'):
                mass_data.append (14.01)
            elif atomname.startswith('P'):
                mass_data.append (30.97)
            elif atomname.startswith('S'):
                mass_data.append (32.06)
            elif atomname.startswith('MG'):
                mass_data.append (24.30)
            else:
                print ('Invalid atome name ', atomname, ' at line: ', line)

    #Calculate total mass
        total_mass = 0.0
        for index in range(num_atoms):
            total_mass += mass_data[index]
        print ('total mass: ',total_mass)

    #Calculate the center of mass using total_mass and mass_data
    x_tot = 0.0
    y_tot = 0.0
    z_tot = 0.0
    
    #poisition_data[i][0] is atom's serial number in input file, do not use here
    for i in range (num_atoms):
        x_tot += float(position_data[i][1])*(mass_data[i])

    for i in range (num_atoms):
        y_tot += float(position_data[i][2])*(mass_data[i])

    for i in range (num_atoms):
        z_tot += float(position_data[i][3])*(mass_data[i])

    x_cm = x_tot/total_mass
    y_cm = y_tot/total_mass
    z_cm = z_tot/total_mass

    center_of_mass = [x_cm, y_cm, z_cm]
    print ('com in angstroms: ', center_of_mass)

    return center_of_mass

def remote_run_complex(parmfile, coordfile, arguments):
    '''
    This function simply executes the program remotely, assumes a complex as input so finds 6 total restraints.
    It skips the drawing geometry portion and only returns the values needed, it does not print anything to screen.

    '''

    #Get Pytraj information
    traj = pt.load(coordfile, parmfile)
    receptor_mask = arguments["parameters"]["receptor_mask"]
    ligand_mask = arguments["parameters"]["ligand_mask"][0]
    com_complex, num_atoms, ligand, com_ligand, receptor, com_receptor = map_molecule_parm(traj, receptor_mask, ligand_mask)

    #Get ParmEd information
    parmed_traj = pmd.load_file(parmfile)

    #No longer useful for all cases, moved into relevant cases below
    #ligand_suba1, lig_a1_coords, dist_liga1_com  = screen_for_distance_restraints(ligand.n_atoms,  com_ligand, ligand)
    #ligand_a1 = receptor.n_atoms + ligand_suba1
    #dist_liga1_com = distance_calculator(lig_a1_coords, com_ligand)
    
    #print('ligand: ', ligand)
    #print('distance of lig_a1 from lig CoM:', dist_liga1_com)
    #print('ligand_Com: ', com_ligand)
    #print('lig_a1_coords', lig_a1_coords)
    #print('')

    r1_input = arguments["parameters"]["restraint_type"]
    
    if (r1_input == 1):
        print ('Distance Restraints will be between CoM Ligand and CoM Receptor')
        #find atom closest to ligand's CoM and relevand information
        ligand_suba1, lig_a1_coords, dist_liga1_com  = screen_for_distance_restraints(ligand.n_atoms,  com_ligand, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
        dist_liga1_com = distance_calculator(lig_a1_coords, com_ligand)
        receptor_a1, rec_a1_coords, dist_reca1_com = screen_for_distance_restraints(receptor.n_atoms,  com_receptor, receptor)
        dist_rest = distance_calculator(lig_a1_coords, rec_a1_coords) 
    elif (r1_input == 2):  
        print ('Distance Restraints will be between CoM Ligand and closest heavy atom in receptor')
        #find atom closest to ligand's CoM and relevand information                                                                                                                                                                                           
        ligand_suba1, lig_a1_coords, dist_liga1_com  = screen_for_distance_restraints(ligand.n_atoms,  com_ligand, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
        dist_liga1_com = distance_calculator(lig_a1_coords, com_ligand)
        receptor_a1, rec_a1_coords, dist_rest = screen_for_distance_restraints(receptor.n_atoms, lig_a1_coords, receptor)
        dist_reca1_com = distance_calculator(rec_a1_coords, com_receptor)
    elif (r1_input == 3):   
        print('Distance restraints will be between the two closest heavy atoms in the ligand and the receptor')
        ligand_suba1, lig_a1_coords, receptor_a1, rec_a1_coords, dist_rest = screen_for_shortest_distant_restraint(receptor.n_atoms, ligand.n_atoms, receptor, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
        dist_liga1_com = distance_calculator(lig_a1_coords, com_ligand)
        dist_reca1_com = distance_calculator(rec_a1_coords, com_receptor)
    #find distance between CoM atoms for distance restraint
    else:
        print("Invalid --r1 type input, must be 1,2 or 3 to choose type of restraint")
    
     #print('ligand: ', ligand)                                                                                                                                                                                                                         
    print('distance of lig_a1 from lig CoM:', dist_liga1_com)
    print('ligand_Com: ', com_ligand)
    print('lig_a1_coords', lig_a1_coords)
    print('distance of rec_a1 from rec CoM:', dist_reca1_com)
    print('distance between lig_a1 and rec_a1:', dist_rest)

    #Create data Arrays of atoms in ligand, with corresponding ID and angles with distance restraints
    #ligand_angleValues_array, ligand_distanceValues_array, ligand_atom_coordinates_array = create_DistanceAngle_array(ligand.n_atoms, lig_a1_coords, rec_a1_coords, ligand)
    #print(' ligand_distanceValues_array: ', ligand_distanceValues_array)
    #print(' ligand_atom_id_array: ',  ligand_atom_coordinates_array)
    #print(' ligand_angleValues_array: ', ligand_angleValues_array)

    #Create data Arrays of atoms in receptor, with corresponding ID and angles with distance restraints
    #receptor_angleValues_array, receptor_distanceValues_array, receptor_atom_coordinates_array = create_DistanceAngle_array(receptor.n_atoms, rec_a1_coords, lig_a1_coords, receptor)
    #print('receptor_distanceValues_array: ', receptor_distanceValues_array)
    #print(' receptor_atom_id_array: ',  receptor_atom_coordinates_array)
    #print('receptor_angleValues_array: ',receptor_angleValues_array)

    ligand_suba2, ligand_atom2_name,  lig_a2_coords, dist_liga2_a3, dist_liga1_a2, lig_angle1, lig_angle2, lig_torsion, ligand_suba3, ligand_atom3_name, lig_a3_coords = screen_arrays_for_angle_restraints(lig_a1_coords, rec_a1_coords, ligand, parmed_traj, traj)
    ligand_a2 = receptor.n_atoms + ligand_suba2
    ligand_a3 = receptor.n_atoms + ligand_suba3

    receptor_a2, receptor_atom2_name,  rec_a2_coords, dist_reca2_a3, dist_reca1_a2, rec_angle1, rec_angle2, rec_torsion, receptor_a3, receptor_atom3_name, rec_a3_coords = screen_arrays_for_angle_restraints(rec_a1_coords, lig_a1_coords, receptor, parmed_traj, traj)

    #screen_arrays_for_angle_restraints(rec_a1_coords, lig_a1_coords, receptor)

    #find atom 2 in ligand     
    #ligand_suba2, lig_a2_coords, dist_liga1_a2, lig_angle1 = screen_for_angle_restraints(ligand.n_atoms, lig_a1_coords, rec_a1_coords, ligand)
    #ligand_a2 = receptor.n_atoms + ligand_suba2

    #find atom 2 in receptor     
    #receptor_a2, rec_a2_coords, dist_reca1_a2, rec_angle1 = screen_for_angle_restraints(receptor.n_atoms, rec_a1_coords, lig_a1_coords, receptor)

    #find atom 3 in ligand 
    #ligand_suba3, lig_a3_coords, dist_liga2_a3, lig_angle2 = screen_for_angle_restraints(ligand.n_atoms, lig_a2_coords, lig_a1_coords, ligand)
    #ligand_a3 = receptor.n_atoms + ligand_suba3

    #find atom 3 in receptor
    #receptor_a3, rec_a3_coords, dist_reca2_a3, rec_angle2 = screen_for_angle_restraints(receptor.n_atoms, rec_a2_coords, rec_a1_coords, receptor)

    #calculate torsion restraint inside ligand
    #lig_torsion = wikicalculate_dihedral_angle(rec_a1_coords, lig_a1_coords, lig_a2_coords, lig_a3_coords)

    #calculate torsion restraint inside receptor                                                                                                                                                                    
    #rec_torsion = wikicalculate_dihedral_angle(lig_a1_coords, rec_a1_coords, rec_a2_coords, rec_a3_coords)
 
    #calculate torsion restraint inside receptor                                                                                                                                                                    
    central_torsion = wikicalculate_dihedral_angle(rec_a2_coords, rec_a1_coords, lig_a1_coords, lig_a2_coords)

    if __name__ != '__main__':
        return receptor_a3, receptor_a2, receptor_a1, ligand_a1, ligand_a2, ligand_a3, dist_rest, lig_angle1, rec_angle1, lig_torsion, rec_torsion, central_torsion 

    elif __name__ == '__main__':
        return receptor_a3, receptor_a2, receptor_a1, ligand_a1, ligand_a2, ligand_a3, dist_rest, lig_angle1, rec_angle1,lig_angle2, rec_angle2, lig_torsion, rec_torsion, central_torsion, lig_a1_coords, dist_liga1_com, rec_a1_coords, dist_reca1_com, lig_a2_coords, dist_liga1_a2, rec_a2_coords, dist_reca1_a2,  lig_a3_coords, dist_liga2_a3, rec_a3_coords, dist_reca2_a3

if __name__ == '__main__':

    #The below script it for use with error checking.  restraint_finder can be run direcctly and the belwo script will print to screen a large amoutn of information about the restraints it finds using the given input files.
    parser = argparse.ArgumentParser()
    #parser.add_argument("--i", help="inputfile to be analyzed")
    parser.add_argument("--p", help="input .parm file")
    parser.add_argument("--c", help="input coordinate file") #do not use nc files for test runs
    parser.add_argument("--r1", help="Must be 1, 2 or 3.  1: Distance Restraints will be between CoM Ligand and CoM Receptor, 2: Distance Restraints will be between CoM Ligand and closest heavy atom in receptor, 3: Distance restraints will be between the two closest heavy atoms in the ligand and the receptor ")

    args = parser.parse_args()
    #inputfile = args.i
    parm = args.p
    coord = args.c
    r1_input = int(args.r1)
    #comp = args.complex
    #lineflag = 'ATOM'
    traj = pt.load(coord, parm)

    print ("============================================================")

    #The following code assumes input of a complex, for which the ligand and protein must be identified separately 
    com_complex, num_atoms, ligand, com_ligand, receptor, com_receptor = map_molecule_parm(traj)
    print('TESTING INITIAL VALUES!')
    print('parm: ', traj)
    print('parm_com: ', com_complex)
    print('parm_num_atoms: ', num_atoms)
    print('')
    print('parm_ligand: ', ligand)
    print('com_ligand: ', com_ligand)
    print('')
    print('parm_receptor: ', receptor)
    print('com_receptor: ', com_receptor)
    print('')
    print('CHECKING VALUES FROM REMOTE RUN FUNCTION!')
    #receptor_a3, receptor_a2, receptor_a1, ligand_a1, ligand_a2, ligand_a3, dist_rest, lig_angle1, rec_angle1, lig_torsion, rec_torsion, central_torsion = remote_run_complex(parm, coord)
    receptor_a3, receptor_a2, receptor_a1, ligand_a1, ligand_a2, ligand_a3, dist_rest, lig_angle1, rec_angle1, lig_angle2, rec_angle2, lig_torsion, rec_torsion, central_torsion, lig_a1_coords, dist_liga1_com, rec_a1_coords, dist_reca1_com, lig_a2_coords, dist_liga1_a2, rec_a2_coords, dist_reca1_a2,  lig_a3_coords, dist_liga2_a3, rec_a3_coords, dist_reca2_a3 = remote_run_complex(parm, coord, r1_input)
  
    #draw_geometry(rec_a1_coords, rec_a2_coords, rec_a3_coords, lig_a1_coords, lig_a2_coords, lig_a3_coords, ligand, receptor)

    #print('closest: ', dist)

    #position_data, num_atoms, com = map_molecule(inputfile, lineflag)
    #distance_from_com, distance_atom, d_atom_position = screen_for_distance_restraints(num_atoms, com, position_data)
    print ('')
    print ('OUTPUTTING ADDITIONAL VALUES OBTAINED FROM REMOTE RUN FUNCTION!')
    print ("Atom ID's for restraints: [", receptor_a3, receptor_a2, receptor_a1, ligand_a1, ligand_a2, ligand_a3 ,"]")
    print ("Atom ID for Lig_atom1 (Ligand Dist restraint): ", ligand_a1, ", Coords: ", lig_a1_coords, ", Distance from CoM: ", dist_liga1_com)
    print ("Atom ID for Rec_atom1 (Receptor Dist restraint): ", receptor_a1, ", Coords: ", rec_a1_coords, ", Distance from CoM: ", dist_reca1_com)

    print ("Distance between both centrial atoms L1 and R1: ", distance_calculator(lig_a1_coords, rec_a1_coords))

    #angle_atom_distance, angle_atom, a_atom_position = screen_for_angle_restraints(num_atoms, d_atom_position, position_data)
    print ("Atom ID for Lig_atom2: ", ligand_a2, ", Coords: ", lig_a2_coords, ", Distance from Lig_atom1 (Ligand Central Atom): ", dist_liga1_a2, "\nAngle between Lig_atom2, Lig_atom1 and Rec_atom1: ", lig_angle1)
    print ("Atom ID for Rec_Atom2: ", receptor_a2, ", Coords: ", rec_a2_coords, ", Distance from Rec_atom1 (Receptor Central Atom): ", dist_reca1_a2, "\nAngle betwen Rec_atom2, Rec_atom1, and Lig_atom1: ", rec_angle1)
    #torsion_atom_distance, torsion_atom, torsion_coords, angle =  screen_for_torsion_restraints(num_atoms, d_atom_position, a_atom_position, position_data)
    print ("Atom ID for Lig_atom3: ", ligand_a3, ", Coords: ", lig_a3_coords, ", Distance from Lig_atom2: ", dist_liga2_a3, "\nAngle between  Lig_atom3,  Lig_atom2 and  Lig_atom1: ", lig_angle2)
    print ("Atom ID for Rec_atom3: ", receptor_a3, ", Coords: ", rec_a3_coords, ", Distance from Recatom2: ", dist_reca2_a3, "\nAngle between Rec_atom3, Rec_atom2 and Rec_atom1: ", rec_angle2)
    print ("Dihedral Angle for R1:L1-L2:L3, Torsion Restraint within Ligand: ", lig_torsion)
    print ("Dihedral Angle for L1:R1-R2:R3, Torsion Restraint within Receptor: ", rec_torsion)
    print ("Dihedral Angle for R2:R1-L1:L2, Central Torsion Restraint: ", central_torsion)
    print ("============================================================")
    #draw_geometry(rec_a1_coords, rec_a2_coords, rec_a3_coords, lig_a1_coords, lig_a2_coords, lig_a3_coords, ligand, receptor)
