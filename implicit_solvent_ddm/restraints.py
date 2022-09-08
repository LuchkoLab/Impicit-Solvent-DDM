import itertools
import logging
import math
import os
import re
import time
from string import Template
from turtle import distance, pos
from unittest.mock import NonCallableMagicMock

import numpy as np
import pandas as pd
import parmed as pmd
import pytraj as pt
from toil.common import FileID
from toil.job import Job

from implicit_solvent_ddm.config import Config


class RestraintMaker(Job):
    def __init__(self,  config: Config, conformational_template=None, orientational_template=None) -> None:
        super().__init__()
        self.complex_restraints_file = config.intermidate_args.complex_restraint_files
        self.ligand_restraint_file = config.intermidate_args.guest_restraint_files
        self.receptor_restraint_file = config.intermidate_args.receptor_restraint_files
        self.conformational_forces = config.intermidate_args.conformational_restraints_forces
        self.orientational_forces = config.intermidate_args.orientational_restriant_forces
        self.config = config 
        self.restraints = {}
    
    
    def run(self,fileStore):
        
        for index, (conformational_force, orientational_force) in enumerate(zip(
                self.config.intermidate_args.conformational_restraints_forces,
                self.config.intermidate_args.orientational_restriant_forces)):
            
            if self.ligand_restraint_file == None:
                
               
                    
                conformational_restraints = self.addChildJobFn(get_conformational_restraints, 
                            self.config.endstate_files.complex_parameter_filename,
                            self.config.inputs["endstate_complex_lastframe"],
                            self.config.amber_masks.receptor_mask,
                            self.config.amber_masks.ligand_mask)
                
                self.conformational_restraints = conformational_restraints
                
                orientational_restraints = self.addChildJobFn(get_orientational_restraints,
                self.config.endstate_files.complex_parameter_filename,
                self.config.inputs["endstate_complex_lastframe"],
                self.config.amber_masks.receptor_mask,
                self.config.amber_masks.ligand_mask,
                self.config.intermidate_args.restraint_type,
                self.config.intermidate_args.max_orientational_restraint,
                self.config.intermidate_args.max_conformational_restraint)
                
                self.boresch_deltaG = orientational_restraints.rv(1)
                
                self.restraints[
                f"ligand_{conformational_force}_rst"] = self.addFollowOnJobFn(
                write_restraint_forces,
                conformational_restraints.rv(1),
                conformational_force=conformational_force).rv()
                    
                self.restraints[
                    f"receptor_{conformational_force}_rst"] =  self.addFollowOnJobFn(
                    write_restraint_forces,
                    conformational_restraints.rv(2),
                    conformational_force=conformational_force).rv()
                
                self.restraints[
                    f"complex_{conformational_force}_{orientational_force}_rst"] = self.addFollowOnJobFn(
                    write_restraint_forces,
                    conformational_restraints.rv(0),
                    orientational_restraints.rv(0),
                    conformational_force=conformational_force,
                    orientational_force=orientational_force).rv()
            else:
                self.restraints[f"ligand_{conformational_force}_rst"] = self.ligand_restraint_file[index]
                    
                self.restraints[
                    f"receptor_{conformational_force}_rst"] = self.receptor_restraint_file[index]
                
                self.restraints[
                    f"complex_{conformational_force}_{orientational_force}_rst"] = self.complex_restraints_file[index]

                self.boresch_deltaG = compute_boresch_restraints(dist_restraint_r=self.config.boresch_parameters.dist_restraint_r,
                                                                 angle1_rest_val=self.config.boresch_parameters.angle2_rest_val,
                                                                 angle2_rest_val=self.config.boresch_parameters.angle2_rest_val,
                                                                 dist_rest_Kr=self.config.boresch_parameters.dist_rest_Kr,
                                                                 angle1_rest_Ktheta1=self.config.boresch_parameters.max_conformational_force,
                                                                 angle2_rest_Ktheta2=self.config.boresch_parameters.max_conformational_force,
                                                                 torsion1_rest_Kphi1=self.config.boresch_parameters.max_orientational_force,
                                                                 torsion2_rest_Kphi2=self.config.boresch_parameters.max_orientational_force,
                                                                 torsion3_rest_Kphi3=self.config.boresch_parameters.max_orientational_force,)
                
        return self
    
    
    def add_complex_window(self, conformational_force, orientational_force):
        
        return self.addChildJobFn(
                write_restraint_forces,
                self.conformational_restraints.rv(0),
                self.orientational_restraints.rv(0),
                conformational_force=conformational_force,
                orientational_force=orientational_force).rv()
    
    def add_ligand_window(self, system,  conformational_force):
        
        return self.addChildJobFn(
            write_restraint_forces,
            self.conformational_restraints.rv(1),
            conformational_force=conformational_force).rv()
        
    def add_receptor_window(self, conformational_force):
            
        return self.addFollowOnJobFn(
            write_restraint_forces,
            self.conformational_restraints.rv(2),
            conformational_force=conformational_force).rv()
            
    @staticmethod
    def get_restraint_file(restraint_obj, system, conformational_force, orientational_force=None):
        
        if system == 'ligand':
            return restraint_obj[f"ligand_{conformational_force}_rst"]
        
        elif system == "receptor":
            return restraint_obj[f"receptor_{conformational_force}_rst"]
        else:
            return restraint_obj[f"complex_{conformational_force}_{orientational_force}_rst"]

def get_conformational_restraints(
    job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask
):

    """
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
    """
    tempDir = job.fileStore.getLocalTempDir()
    complex_prmtop_ID = job.fileStore.readGlobalFile(
        complex_prmtop, userPath=os.path.join(tempDir, os.path.basename(complex_prmtop))
    )
    complex_coordinate_ID = job.fileStore.readGlobalFile(
        complex_coordinate,
        userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])),
    )

    traj_complex = pt.load(complex_coordinate_ID, complex_prmtop_ID)
    ligand = traj_complex[ligand_mask]
    receptor = traj_complex[receptor_mask]

    receptor_atom_neighbor_index = create_atom_neighbor_array(receptor.xyz[0])
    ligand_atom_neighbor_index = create_atom_neighbor_array(ligand.xyz[0])

    ligand_template = conformational_restraints_template(ligand_atom_neighbor_index)
    receptor_template = conformational_restraints_template(receptor_atom_neighbor_index)
    complex_template = conformational_restraints_template(
        ligand_atom_neighbor_index, num_receptor_atoms=receptor.n_atoms
    )

    # Create a local temporary file.

    ligand_scratchFile = job.fileStore.getLocalTempFile()
    receptor_scratchFile = job.fileStore.getLocalTempFile()
    complex_scratchFile = job.fileStore.getLocalTempFile()
    # job.log(f"ligand_template {ligand_template}")
    with open(complex_scratchFile, "w") as fH:
        fH.write(complex_template)
        fH.write(receptor_template)
        fH.write("&end")
    with open(ligand_scratchFile, "w") as fH:
        fH.write(ligand_template)
        fH.write("&end")
    with open(receptor_scratchFile, "w") as fH:
        fH.write(receptor_template)
        fH.write("&end")

    restraint_complex_ID = job.fileStore.writeGlobalFile(complex_scratchFile)
    restraint_ligand_ID = job.fileStore.writeGlobalFile(ligand_scratchFile)
    restriant_receptor_ID = job.fileStore.writeGlobalFile(receptor_scratchFile)

    # job.fileStore.export_file(restraint_complex_ID, "file://" + os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/output_directory", os.path.basename(restraint_complex_ID))))
    # toil.exportFile(outputFileID, "file://" + os.path.abspath(os.path.join(ioFileDirectory, "out.txt")))

    return (restraint_complex_ID, restraint_ligand_ID, restriant_receptor_ID)


def get_orientational_restraints(
    job,
    complex_prmtop,
    complex_coordinate,
    receptor_mask,
    ligand_mask,
    restraint_type,
    max_torisonal_rest,
    max_distance_rest,
):
    """
    Job to create orientational restraint file, based on user specified restraint type chosen within the config.

    The orentational restraints will returns 6 atoms best suited for NMR restraints, based on specific restraint type the user specified.
    restaint_type = 1 : Find atom closest to ligand's CoM and relevand information
    restaint_type = 2: Distance Restraints will be between CoM Ligand and closest heavy atom in receptor
    restraint_type = 3: Distance restraints will be between the two closest heavy atoms in the ligand and the receptor
    """

    tempDir = job.fileStore.getLocalTempDir()
    complex_prmtop_ID = job.fileStore.readGlobalFile(
        complex_prmtop, userPath=os.path.join(tempDir, os.path.basename(complex_prmtop))
    )
    complex_coordinate_ID = job.fileStore.readGlobalFile(
        complex_coordinate,
        userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])),
    )

    traj_complex = pt.load(complex_coordinate_ID, complex_prmtop_ID)
    ligand = traj_complex[ligand_mask]
    receptor = traj_complex[receptor_mask]

    # center of mass
    ligand_com = pt.center_of_mass(ligand)
    receptor_com = pt.center_of_mass(receptor)
    # Get ParmEd information
    parmed_traj = pmd.load_file(complex_prmtop_ID)

    if restraint_type == 1:
        # find atom closest to ligand's CoM and relevand information
        ligand_suba1, lig_a1_coords, dist_liga1_com = screen_for_distance_restraints(ligand.n_atoms, 
            ligand_com, ligand
        )
        ligand_a1 = receptor.n_atoms + ligand_suba1
        dist_liga1_com = distance_calculator(lig_a1_coords, ligand_com)
        receptor_a1, rec_a1_coords, dist_reca1_com = screen_for_distance_restraints(receptor.n_atoms,
             receptor_com, receptor
        )
        dist_rest = distance_calculator(lig_a1_coords, rec_a1_coords)

    elif restraint_type == 2:
        # find atom closest to ligand's CoM and relevand information
        ligand_suba1, lig_a1_coords, dist_liga1_com = screen_for_distance_restraints(ligand.n_atoms,
            ligand_com, ligand
        )
        ligand_a1 = receptor.n_atoms + ligand_suba1
        receptor_a1, rec_a1_coords, dist_rest = screen_for_distance_restraints(receptor.n_atoms,
            lig_a1_coords, receptor
        )

    elif restraint_type == 3:
        (
            ligand_suba1,
            lig_a1_coords,
            receptor_a1,
            rec_a1_coords,
            dist_rest,
        ) = shortest_distance_between_molecules(receptor, ligand)
        ligand_a1 = receptor.n_atoms + ligand_suba1
    # find distance between CoM atoms for distance restraint
    else:
        raise RuntimeError(
            "Invalid --r1 type input, must be 1,2 or 3 to choose type of restraint"
        )
    
    '''
    return (
        selected_atom2,
        saved_atom2_position,
        saved_distance_a2a3_value,
        saved_distance_a1a2_value,
        saved_angle_a1a2,
        saved_angle_a2a3,
        saved_torsion_angle,
        selected_atom3,
        saved_atom3_position,
    )
    )
    '''
    (
        ligand_suba2,
        lig_a2_coords,
        dist_liga2_a3,
        dist_liga1_a2,
        lig_angle1,
        lig_angle2,
        lig_torsion,
        ligand_suba3,
        lig_a3_coords,
    ) = refactor_screen_arrays_for_angle_restraints(
        lig_a1_coords, rec_a1_coords, ligand, parmed_traj
    )
    ''' (
        ligand_suba2,
        ligand_atom2_name,
        lig_a2_coords,
        dist_liga2_a3,
        dist_liga1_a2,
        lig_angle1,
        lig_angle2,
        lig_torsion,
        ligand_suba3,
        ligand_atom3_name,
        lig_a3_coords,
    ) = screen_arrays_for_angle_restraints(
        lig_a1_coords, rec_a1_coords, ligand, parmed_traj, traj_complex
    )
    '''
    ligand_a2 = receptor.n_atoms + ligand_suba2
    ligand_a3 = receptor.n_atoms + ligand_suba3

    (
        receptor_a2,
        rec_a2_coords,
        dist_reca2_a3,
        dist_reca1_a2,
        rec_angle1,
        rec_angle2,
        rec_torsion,
        receptor_a3,
        rec_a3_coords,
    ) = refactor_screen_arrays_for_angle_restraints(
        rec_a1_coords, lig_a1_coords, receptor, parmed_traj
    )
   
    central_torsion = wikicalculate_dihedral_angle(
        rec_a2_coords, rec_a1_coords, lig_a1_coords, lig_a2_coords)
    
    orientaional_conformational_template = orientational_restraints_template(
        receptor_a3,
        receptor_a2,
        receptor_a1,
        ligand_a1,
        ligand_a2,
        ligand_a3,
        dist_rest,
        lig_angle1,
        rec_angle1,
        lig_torsion,
        rec_torsion,
        central_torsion,
    )

    complex_name = re.sub(r"\..*", "", os.path.basename(complex_prmtop))

    with open(f"{complex_name}_orientational_template.RST", "w") as restraint_string:
        restraint_string.write(orientaional_conformational_template)

    return (
        job.fileStore.writeGlobalFile(f"{complex_name}_orientational_template.RST"),
        compute_boresch_restraints(
            dist_rest,
            rec_angle1,
            lig_angle1,
            max_distance_rest,
            max_torisonal_rest,
            max_torisonal_rest,
            max_torisonal_rest,
            max_torisonal_rest,
            max_torisonal_rest,
        ),
    )
    
def write_restraint_forces(
    job,
    conformational_template,
    orientational_template=None,
    conformational_force=0.0,
    orientational_force=0.0,
):

    temp_dir = job.fileStore.getLocalTempDir()

    read_conformational_template = job.fileStore.readGlobalFile(
        conformational_template,
        userPath=os.path.join(temp_dir, os.path.basename(conformational_template)),
    )
    string_template = ""

    if orientational_template is not None:
        read_orientational_template = job.fileStore.readGlobalFile(
            orientational_template,
            userPath=os.path.join(temp_dir, os.path.basename(orientational_template)),
        )
        with open(read_orientational_template) as oren_temp:
            template = Template(oren_temp.read())
            orientational_temp = template.substitute(
                drest=conformational_force,
                arest=orientational_force,
                trest=orientational_force,
            )
            string_template += orientational_temp

    with open(read_conformational_template) as temp:
        template = Template(temp.read())
        restraint_temp = template.substitute(frest=conformational_force)
        string_template += restraint_temp

    string_template = string_template.replace("&end", "")

    with open("restraint.RST", "w") as fn:
        fn.write(string_template)
        fn.write("&end")

    return job.fileStore.writeGlobalFile("restraint.RST")

  
def compute_boresch_restraints(
    dist_restraint_r:float,
    angle1_rest_val:float,
    angle2_rest_val:float,
    dist_rest_Kr:float,
    angle1_rest_Ktheta1:float,
    angle2_rest_Ktheta2:float,
    torsion1_rest_Kphi1:float,
    torsion2_rest_Kphi2:float,
    torsion3_rest_Kphi3:float,
):
    """
    Analytically calculate the DeltaG of Boresch restraints contribution

    Parameters
    ----------

    """

    Rgas = 8.31446261815324 #Ideal Gas constant (J)/(mol*K)
    kB = (
        Rgas / 4184
    )  # Converting from Joules to kcal units (kB = Rgas when using molar units)
    T = 298  # Value read from mdin file, assume this is natural units for the similatuion(Kelvin)
    V = 1660  # Angstrom Cubed
    # k = 1.38*(10**(-23))

    theta_1_rad = math.radians(angle1_rest_val)
    theta_2_rad = math.radians(angle2_rest_val)
    K_numerator = math.sqrt(
        dist_rest_Kr
        * angle1_rest_Ktheta1
        * angle2_rest_Ktheta2
        * torsion1_rest_Kphi1
        * torsion2_rest_Kphi2
        * torsion3_rest_Kphi3
    )
    K_denom = (2 * (math.pi) * kB * T) ** 3
    left_numerator = 8 * ((math.pi) ** 2) * V
    left_denom = (
        (dist_restraint_r**2) * (math.sin(theta_1_rad)) * (math.sin(theta_2_rad))
    )
    log_argument = (left_numerator / left_denom) * (K_numerator / K_denom)
    result = kB * T * np.log(log_argument)

    df = pd.DataFrame()
    df["r"] = [dist_restraint_r]
    df["theta_1"] = [angle1_rest_val]
    df["theta_2"] = [angle2_rest_val]
    df["Kr"] = [dist_rest_Kr]
    df["Ktheta_1"] = [angle1_rest_Ktheta1]
    df["Ktheta_2"] = [angle2_rest_Ktheta2]
    df["Kphi1"] = [torsion1_rest_Kphi1]
    df["Kphi2"] = [torsion2_rest_Kphi2]
    df["Kphi3"] = [torsion3_rest_Kphi3]
    df["DeltaG"] = [result]
    
    return df


def flat_bottom_restraints_template(
    host_guest_atoms, guest_atoms, flat_bottom_distances
):

    restraint_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "templates/restraints/COM.restraint",
        )
    )
    string_template = ""

    with open(restraint_path) as f:
        template = Template(f.read())

        restraint_template = template.substitute(
            host_atom_numbers=host_guest_atoms,
            guest_atom_numbers=guest_atoms,
            r1=flat_bottom_distances["r1"],
            r2=flat_bottom_distances["r2"],
            r3=flat_bottom_distances["r3"],
            r4=flat_bottom_distances["r4"],
            rk2=flat_bottom_distances["rk2"],
            rk3=flat_bottom_distances["rk3"],
        )

        string_template += restraint_template
    return string_template


def conformational_restraints_template(
    solute_conformational_restraint, num_receptor_atoms=0
):

    restraint_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "templates/restraints/conformational_restraints.template",
        )
    )
    string_template = ""
    for index in range(len(solute_conformational_restraint)):
        with open(restraint_path) as f:
            template = Template(f.read())

            restraint_template = template.substitute(
                solute_primary_atom=solute_conformational_restraint[index][0]
                + num_receptor_atoms,
                solute_sec_atom=solute_conformational_restraint[index][1]
                + num_receptor_atoms,
                distance=solute_conformational_restraint[index][2],
                frest="$frest",
            )
            string_template += restraint_template
    return string_template


def orientational_restraints_template(
    atom_R3,
    atom_R2,
    atom_R1,
    atom_L1,
    atom_L2,
    atom_L3,
    dist_rest,
    lig_angrest,
    rec_angrest,
    lig_torres,
    rec_torres,
    central_torres,
):
    """
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
    """

    string_template = ""
    restraint_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "templates/restraints/orientational.template",
        )
    )
    # restraint_path = "/nas0/ayoub/Impicit-Solvent-DDM/implicit_solvent_ddm/templates/restraint.RST"
    with open(restraint_path) as t:
        template = Template(t.read())
        restraint_template = template.substitute(
            atom_R3=atom_R3,
            atom_R2=atom_R2,
            atom_R1=atom_R1,
            atom_L1=atom_L1,
            atom_L2=atom_L2,
            atom_L3=atom_L3,
            dist_rest=dist_rest,
            lig_angrest=lig_angrest,
            rec_angrest=rec_angrest,
            central_torres=central_torres,
            rec_torres=rec_torres,
            lig_torres=lig_torres,
            drest="$drest",
            arest="$arest",
            trest="$trest",
        )
        string_template += restraint_template
    return string_template


def write_empty_restraint(job):
    temp_dir = job.fileStore.getLocalTempDir()
    with open("empty.restraint", "w") as fn:
        fn.write("")
    return job.fileStore.writeGlobalFile("empty.restraint")


def get_flat_bottom_restraints(
    job, complex_prmtop, complex_coordinate, flat_well_parabola
):

    tempDir = job.fileStore.getLocalTempDir()
    complex_prmtop_ID = job.fileStore.readGlobalFile(
        complex_prmtop, userPath=os.path.join(tempDir, os.path.basename(complex_prmtop))
    )
    complex_coordinate_ID = job.fileStore.readGlobalFile(
        complex_coordinate,
        userPath=os.path.join(tempDir, os.path.basename(complex_coordinate[0])),
    )

    complex_parmed_traj = pmd.load_file(complex_prmtop_ID, complex_coordinate_ID)
    host_atom_numbers = ",".join(
        [
            str(atom)
            for atom in range(
                1, int(complex_parmed_traj.parm_data["RESIDUE_POINTER"][1])
            )
        ]
    )
    guest_atom_numbers = ",".join(
        [
            str(i)
            for i in range(
                int(complex_parmed_traj.parm_data["RESIDUE_POINTER"][1]),
                int(complex_parmed_traj.parm_data["POINTERS"][0] + 1),
            )
        ]
    )
    flat_bottom_string = flat_bottom_restraints_template(
        host_atom_numbers, guest_atom_numbers, flat_well_parabola
    )

    temp_file = job.fileStore.getLocalTempFile()
    with open(temp_file, "w") as fH:
        fH.write(flat_bottom_string)

    return job.fileStore.writeGlobalFile(temp_file)



def create_atom_neighbor_array(atom_coordinates):
    """
     Conformational restraints will be applied by creating harmonic distance restraints between every atom and all neighbors within 6 Å that are part of the same molecule

     Parameters
     ----------
     atom_coordinates: numpy.ndarray
        An array of atomic coordiantes
    Returns
    -------
    atom_neighbor_array: list
         A list containing the nearest neighbor atoms within 6 angstroms
    """

    start_time = time.time()

    atom_neighbor_array = []
    atoms = [_ for _ in range(1, len(atom_coordinates) + 1)]
    atom_pairs = list(itertools.combinations(atoms, r=2))

    coordinate_pairs = list(itertools.combinations(atom_coordinates, r=2))
    atom_distances = list(itertools.starmap(distance_calculator, coordinate_pairs))
    closest_neighbor = list(map(distance_filter, atom_distances))

    for atom_pair_index, neighbors in enumerate(closest_neighbor):
        if neighbors:
            atom_neighbor_array.append(
                [
                    atom_pairs[atom_pair_index][0],
                    atom_pairs[atom_pair_index][1],
                    atom_distances[atom_pair_index],
                ]
            )
    print("--- %s seconds ---" % (time.time() - start_time))
    return atom_neighbor_array


def distance_calculator(point_a, point_b):
    """
    Function calcualtes the distance between two points
    """

    a_array = np.asarray(point_a, dtype=float)
    b_array = np.asarray(point_b, dtype=float)
    difference = np.subtract(a_array, b_array)
    return np.linalg.norm(difference)


def distance_filter(distance) -> bool:
    """
    Returns a boolean whether the atoms pairs are within 6 angstroms

    Returns
    -------
        Returns True: if the distance is less than or equal to 6
        Returns False: if the distance is greater than 6
    """
    return distance <= 6


def distance_btw_center_of_mass(com, mol):
    """
    This function combs a molecule object (num_atoms) to find the atom closest to the coordinates at (com).
    This (com) atom is traditionally an atom closest to the center of mass used as a distance restraint in NMR calculations in AMBER.
    It should however be re_named as it sometimes is another atom altogther.

    Parameters
    ----------
    num_atoms: int
        Total number of atoms in specified system
    com: numpy.ndarry
        Center of mass of Host or Guest system 
    mol: pytraj.trajectory.trajectory.Trajectory
        Pytraj trajectory file of HOST or guest system 

    Returns
    -------
    ligand_selected_atom_parmindex: int
        Index of the ligand atom
    selected_atom_position: numpy.ndarry
        Array of coordiante of selected atom position
    shorest_distance: float
        distance of shortest
    """
    start_time = time.time()

    all_atom_coords = mol.xyz[0]

    ignore_protons = list(
        itertools.filterfalse(
            lambda atom: atom.name.startswith("H"), mol.topology.atoms
        )
    )
    no_proton_coordinates = [all_atom_coords[atom.index] for atom in ignore_protons]

    # dot product
    atom_combinations = list(itertools.product(no_proton_coordinates, com))

    # calculate the distances
    distances = list(itertools.starmap(distance_calculator, atom_combinations))
    shortest_distance = np.min(distances)
    atom_chosen = ignore_protons[np.argmin(distances)]
    selected_atom_parmindex = atom_chosen.index + 1
    selected_atom_position = all_atom_coords[atom_chosen.index]
    selected_atom_position = np.array([selected_atom_position])
    
    print("--- %s seconds ---" % (time.time() - start_time))
    return selected_atom_parmindex, selected_atom_position, shortest_distance


def screen_for_distance_restraints(num_atoms, com, mol):
    """
    This function combs a molecule object (num_atoms) to find the atom closest to the coordinates at (com).
    This (com) atom is traditionally an atom closest to the center of mass used as a distance restraint in NMR calculations in AMBER.
    It should however be re_named as it sometimes is another atom altogther.

    Parameters
    ----------
    num_atoms: int
        Total number of atoms in specified system
    com: numpy.ndarry
        Center of mass of complex
    mol: pytraj.trajectory.trajectory.Trajectory
        Pytraj trajectory file of HOST or guest system 

    Returns
    -------
    ligand_selected_atom_parmindex: int
        Index of the ligand atom
    selected_atom_position: numpy.ndarry
        Array of coordiante of selected atom position
    shorest_distance: float
        distance of shortest
    """
    start_time = time.time()
    atom_coords = mol.xyz
    # calculate the distance betwen indivdual atoms and center of mass of solute
    distances = list(
        map(
            distance_calculator,
            atom_coords[0],
            itertools.repeat(com, len(atom_coords[0])),
        )
    )
    # initialize from the initial atom
    selected_atom_parmindex = 1
    shorest_distance = distances[0]
    selected_atom_name = mol.topology.atom(0).name
    selected_atom_position = atom_coords[0][0]
    # cycle through all computed distances
    for atom_index in range(num_atoms):
        atom_name = mol.topology.atom(atom_index).name
        if not atom_name.startswith("H"):
            if distances[atom_index] < shorest_distance:
                shorest_distance = distances[atom_index]
                selected_atom_parmindex = (
                    atom_index + 1
                )  # Plus one alters the index intialized at 0 to match with parm id initialized at 1
                selected_atom_name = atom_name
                selected_atom_position = atom_coords[0][atom_index]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(
        "Success!! Selected distance atom: Type:",
        selected_atom_name,
        "At coordinates: ",
        selected_atom_position,
    )
    return selected_atom_parmindex, selected_atom_position, shorest_distance


def shortest_distance_between_molecules(receptor_mol, ligand_mol):
    """
        Finds the shortest distance between host and guest atoms.

        Takes in array of coordinates of host and guest atoms and computes the distance only taking into account heavy atoms, ignoring protons
    Parameters
    ----------
        receptor_mol: numpy.ndarry
            An array of x,y,z positions of each atom within the host
        ligand_mol: numpy.ndarry
            An array of x,y,z positions of each atom within the guest
    Returns
    -------
        ligand_selected_atom_parmindex: int
            The selected atom index value (ligand system)
        ligand_selected_atom_position: numpy.ndarry
            The selected atom position (x,y,z) coordinate for the guest
        receptor_selected_atom_parmindex: int
            The selected atom index value (receptor system)
        receptor_selected_atom_position: numpy.ndarry
            The selected atom position (x,y,z) coordinate for the host
        shortest_distance: float
            The distance between the atoms chosen
    """


    start_time = time.time()

    mole_a_all_atoms = ligand_mol.xyz[0]
    mole_a_no_protons = list(
        itertools.filterfalse(
            lambda atom: atom.name.startswith("H"), ligand_mol.topology.atoms
        )
    )
    mole_a_coordinates = [mole_a_all_atoms[atom.index] for atom in mole_a_no_protons]

    # access molecule coordiantes array
    receptor_mol_all_atoms = receptor_mol.xyz[0]
    # ignore any protons
    mole_b_no_protons = list(
        itertools.filterfalse(
            lambda atom: atom.name.startswith("H"), receptor_mol.topology.atoms
        )
    )
    # coordinates with only heavy atoms
    mole_b_coordinates = [
        receptor_mol_all_atoms[atom.index] for atom in mole_b_no_protons
    ]

    # all possible coordinate combinations between molecule a and b
    atom_coordinate_combindations = list(
        itertools.product(mole_a_coordinates, mole_b_coordinates)
    )
    # all possible atom combinations between molecule a and b
    atom_combindations = list(itertools.product(mole_a_no_protons, mole_b_no_protons))

    # calculate distances
    distances = list(
        itertools.starmap(distance_calculator, atom_coordinate_combindations)
    )

    # shortest distance between systems
    shortest_distance = np.min(distances)
    index_position = np.argmin(distances)

    # get the atom chosen for the ligand and receptor
    ligand_atom_chosen = atom_combindations[index_position][0]
    receptor_atom_chosen = atom_combindations[index_position][1]

    ligand_selected_atom_parmindex = ligand_atom_chosen.index + 1
    ligand_selected_atom_position = atom_coordinate_combindations[index_position][0]
    receptor_selected_atom_parmindex = receptor_atom_chosen.index + 1
    receptor_selected_atom_position = atom_coordinate_combindations[index_position][1]
    print("--- %s seconds ---" % (time.time() - start_time))
    print("refactor_screen_between_molecules\n")
    print(f"ligand_selected_atom_parmindex: {ligand_selected_atom_parmindex}")
    print(f"ligand_selected_atom_position: {ligand_selected_atom_position}")
    print(f"receptor_selected_atom_parmindex: {receptor_selected_atom_parmindex}")
    print(f"receptor_selected_atom_position: {receptor_selected_atom_position}")
    print(f"shortest_distance: {shortest_distance}")
    return (
        ligand_selected_atom_parmindex,
        ligand_selected_atom_position,
        receptor_selected_atom_parmindex,
        receptor_selected_atom_position,
        shortest_distance,
    )

def screen_for_shortest_distant_restraint(receptor_mol, ligand_mol):
    """
        Finds the shortest distance between host and guest atoms.

        Takes in array of coordinates of host and guest atoms and computes the distance only taking into account heavy atoms, ignoring protons
    Parameters
    ----------
        receptor_mol: numpy.ndarry
            An array of x,y,z positions of each atom within the host
        ligand_mol: numpy.ndarry
            An array of x,y,z positions of each atom within the guest
    Returns
    -------
        ligand_selected_atom_parmindex: int
            The selected atom index value (ligand system)
        ligand_selected_atom_position: numpy.ndarry
            The selected atom position (x,y,z) coordinate for the guest
        receptor_selected_atom_parmindex:–– int
            The selected atom index value (receptor system)
        receptor_selected_atom_position: numpy.ndarry
            The selected atom position (x,y,z) coordinate for the host
        shortest_distance: float
            The distance between the atoms chosen
    """

    start_time = time.time()
    receptor_coords = receptor_mol.xyz
    ligand_coords = ligand_mol.xyz

    coorindate_pairs = list(itertools.product(ligand_coords[0], receptor_coords[0]))

    distances = list(itertools.starmap(distance_calculator, coorindate_pairs))
    shortest_distance = distances[0]

    # initial distance between primary atom of the ligand and Receptor
    ligand_selected_atom_parmindex = 1
    receptor_selected_atom_parmindex = 1
    ligand_selected_atom_position = ligand_coords[0][0]
    receptor_selected_atom_position = receptor_coords[0][0]

    for index, distance in enumerate(distances):

        ligand_atom_index = int(index / receptor_mol.n_atoms)
        receptor_atom_index = index % receptor_mol.n_atoms

        receptor_atom_name = receptor_mol.topology.atom(receptor_atom_index).name
        ligand_atom_name = ligand_mol.topology.atom(ligand_atom_index).name

        if not receptor_atom_name.startswith("H") and not ligand_atom_name.startswith(
            "H"
        ):
            if distance < shortest_distance:
                shortest_distance = distance
                ligand_selected_atom_parmindex = (
                    ligand_atom_index + 1
                )  # Plus one alters the index intialized at 0 to match with parm id initialized at 1
                receptor_selected_atom_parmindex = receptor_atom_index + 1
                ligand_selected_atom_name = ligand_atom_name
                receptor_selected_atom_name = receptor_atom_name
                ligand_selected_atom_position = ligand_coords[0][ligand_atom_index]
                receptor_selected_atom_position = receptor_coords[0][
                    receptor_atom_index
                ]

    print("--- %s seconds ---" % (time.time() - start_time))
    print("refactor_screen_between_molecules\n")
    print(f"ligand_selected_atom_parmindex: {ligand_selected_atom_parmindex}")
    print(f"ligand_selected_atom_position: {ligand_selected_atom_position}")
    print(f"receptor_selected_atom_parmindex: {receptor_selected_atom_parmindex}")
    print(f"receptor_selected_atom_position: {receptor_selected_atom_position}")
    print(f"shortest_distance: {shortest_distance}")
    return (
        ligand_selected_atom_parmindex,
        ligand_selected_atom_position,
        receptor_selected_atom_parmindex,
        receptor_selected_atom_position,
        shortest_distance,
    )


# barton source code need to refactor
def screen_arrays_for_angle_restraints(
    atom1_position, atomx_position, mol, parmed_traj, traj
):
    """
    This function screens the arrays created by create_DistanceAngle_array() to help find good matches.

    Args:
    num_atoms: Number of atoms in mol
    angleValues_array: An array of angle values who's index corresponds to distanceValues_array,
    atom_id_array,
    position_tethered_atom,
    mol: Molecule Object with multipole attributes, coords, atom names, etc.


    """

    print("")
    print("Testing New Function")

    # Initialize values
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
    # saved_distance_a1a2_value = 0
    # saved_distance_a2a3_value = 0
    saved_average_distance_value = 0
    # current_angle_a1a2 = 0
    # current_angle_a2a3 = 0
    # current_torsion_angle1 = 0
    # current_torsion_angle2 = 0
    # current_torsion_angle3 = 0
    success = 0
    k = 0

    while success == 0:
        # print('1')
        # print ('min_angle:', min_angle)
        # print ('max_angle:', max_angle)
        while i < num_atoms:
            j = 0
            atom2_name = mol.topology.atom(i).name
            # print ('atom_name i;', mol.topology.atom(i).name)
            if atom2_name.startswith("H") == False:
                while j < num_atoms:
                    if i != j:
                        atom3_name = mol.topology.atom(j).name
                        # print ('atom_name j;', mol.topology.atom(j).name)
                        if atom3_name.startswith("H") == False:
                            # print ('H success 2' )
                            atom2_position = atom_coords[0][
                                i
                            ]  # [position_data[i][1],position_data[i][2],position_data[i][3]]
                            atom3_position = atom_coords[0][
                                j
                            ]  # [position_data[i][1],position_data[i][2],position_data[i][3]]
                            angle_a1a2 = find_angle(
                                atom2_position, atom1_position, atomx_position
                            )
                            angle_a2a3 = find_angle(
                                atom3_position, atom2_position, atom1_position
                            )
                            if (
                                angle_a1a2 > min_angle
                                and angle_a1a2 < max_angle
                                and angle_a2a3 > min_angle
                                and angle_a2a3 < max_angle
                            ):
                                torsion_angle = wikicalculate_dihedral_angle(
                                    atomx_position,
                                    atom1_position,
                                    atom2_position,
                                    atom3_position,
                                )
                                num_heavy_bonds_on_atom_j = find_heavy_bonds(
                                    mol, j, parmed_traj, traj
                                )
                                if num_heavy_bonds_on_atom_j > 1:
                                    new_distance_a1a2 = distance_calculator(
                                        atom1_position, atom2_position
                                    )
                                    new_distance_a2a3 = distance_calculator(
                                        atom2_position, atom3_position
                                    )
                                    new_distance_a3_norm_a1a2 = norm_distance(
                                        atom1_position, atom2_position, atom3_position
                                    )
                                    if (
                                        (new_distance_a1a2 + new_distance_a3_norm_a1a2)
                                        / 2
                                        > saved_average_distance_value
                                        or success == 0
                                    ):
                                        success = 1
                                        saved_distance_a1a2_value = new_distance_a1a2
                                        saved_distance_a2a3_value = new_distance_a2a3
                                        # The norm distance is the determining parameter for success, distance of p3 to the line connecting p1 and p2.
                                        # saved_distance_a2a3_value is only saved for informational purposes
                                        saved_a3norm_distance_value = (
                                            new_distance_a1a2
                                            + new_distance_a3_norm_a1a2
                                        ) / 2
                                        saved_average_distance_value=saved_a3norm_distance_value #update distance 
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
                    # print ('j: ', j)
            i += 1
            # print ('i: ', i)

        if success == 1:
            print("Number of accepted cases: ", k)
        elif success == 0 and min_angle > 10:
            min_angle -= 1
            max_angle += 1
            print(
                "Widening acceptable angle: min_angle: ",
                min_angle,
                "max_angle: ",
                max_angle,
            )
            i = 0
            j = 0
        elif success == 0 and min_angle <= 10:
            sys.exit("no suitable restraint atom found that fit all parameters!!")

    print(
        "Success!! Selected angle atom2: Type:",
        selected_atom2_name,
        "At coordinates: ",
        saved_atom2_position,
    )
    print(
        "Success!! Selected angle atom3: Type:",
        selected_atom3_name,
        "At coordinates: ",
        saved_atom3_position,
    )
    print(f"not refactor {saved_a3norm_distance_value}")
    return (
        selected_atom2,
        selected_atom2_name,
        saved_atom2_position,
        saved_distance_a2a3_value,
        saved_distance_a1a2_value,
        saved_angle_a1a2,
        saved_angle_a2a3,
        saved_torsion_angle,
        selected_atom3,
        selected_atom3_name,
        saved_atom3_position,
    )


def refactor_screen_arrays_for_angle_restraints(
    atom1_position, atomx_position, mol, parmed_traj
):
    """
    This function screens the arrays of atom coordinate between Host and Guest systems. 
    
    Purpose is to find 3 atoms of ligand/Host system which give the best match for Boresch restraints 

    Parameters:
    -----------
    atom1_position: numpy.ndarray 
        coordinates of selected atom 
    atomx_position: numpy.ndarray 
        coordinate of selected atom  
    mol: pytraj.Trajectory 
        molecule object with multipole attributes, coords, atom names, etc.
    parmed_traj: parmed.amber._amberparm.AmberParm
        parmed molecule object with atom name attributes 

    """
   
    min_angle = 80
    max_angle = 100
    num_atoms = mol.n_atoms
    atom_coords = mol.xyz[0]
    atom2_position = []
    atom3_position = []
    # ignore protons return a list of heavy atoms
    no_protons = list(
        itertools.filterfalse(
            lambda atom: atom.name.startswith("H"), mol.topology.atoms
        )
    )

    # now pair heavy atoms
    no_proton_pairs = list(itertools.permutations(no_protons, r=2))

    # get parmed infomation of the second atom
    parmed_atoms = [parmed_traj.atoms[atom[1].index] for atom in no_proton_pairs]

    # if the atom have more than 1 heavy atom bond
    heavy_atoms = list(map(refactor_find_heavy_bonds, parmed_atoms))

    only_heavy_pairs = [x for x, y in zip(no_proton_pairs, heavy_atoms) if y]

    saved_average_distance_value = 0
    suitable_restraint = False
    
    while not suitable_restraint:
        for atom_index, atom in enumerate(only_heavy_pairs):
            atom2_position = atom_coords[
                atom[0].index
            ]  # [position_data[i][1],position_data[i][2],position_data[i][3]]
            atom3_position = atom_coords[
                atom[1].index
            ]  # [position_data[i][1],position_data[i][2],position_data[i][3]]
            angle_a1a2 = find_angle(atom2_position, atom1_position, atomx_position)
            angle_a2a3 = find_angle(atom3_position, atom2_position, atom1_position)

            if (
                angle_a1a2 > min_angle
                and angle_a1a2 < max_angle
                and angle_a2a3 > min_angle
                and angle_a2a3 < max_angle
            ):
                torsion_angle = wikicalculate_dihedral_angle(
                    atomx_position, atom1_position, atom2_position, atom3_position
                )
                new_distance_a1a2 = distance_calculator(atom1_position, atom2_position)
                new_distance_a2a3 = distance_calculator(atom2_position, atom3_position)
                new_distance_a3_norm_a1a2 = norm_distance(
                    atom1_position, atom2_position, atom3_position
                )

                if (
                    new_distance_a1a2 + new_distance_a3_norm_a1a2
                ) / 2 > saved_average_distance_value:
                    saved_distance_a1a2_value = new_distance_a1a2
                    saved_distance_a2a3_value = new_distance_a2a3

                    saved_average_distance_value = (
                        new_distance_a1a2 + new_distance_a3_norm_a1a2
                    ) / 2
                    saved_angle_a2a3 = angle_a2a3
                    saved_angle_a1a2 = angle_a1a2
                    saved_torsion_angle = torsion_angle
                    saved_atom2_position = atom2_position
                    saved_atom3_position = atom3_position
                    selected_atom2 = atom[0].index + 1
                    selected_atom3 = atom[1].index + 1
                    selected_atom2_name = atom[0]
                    selected_atom3_name = atom[1]
                    suitable_restraint = True
                    print(f"print(saved_average_distance_value) refactor {saved_average_distance_value}")
        if not suitable_restraint:
            
            if min_angle > 10:
                print(min_angle)
                min_angle -= 1
                max_angle += 1
            else:
                import sys

                sys.exit("no suitable restraint atom found that fit all parameters!!")

    # logger.info(
    #     f"Widen the acceptable angle for min_angle: {min_angle} and max_angle: {max_angle} "
    # )
    # logger.info(
    #     f"Success!! Selected angle atom2: Type: {selected_atom2_name}, 'At coordinates: {saved_atom2_position} "
    # )
    # logger.info(
    #     f"Success!! Selected angle atom3: Type: {selected_atom3_name}, At coordinates: {saved_atom3_position} \n"
    # )

    return (
        selected_atom2,
        saved_atom2_position,
        saved_distance_a2a3_value,
        saved_distance_a1a2_value,
        saved_angle_a1a2,
        saved_angle_a2a3,
        saved_torsion_angle,
        selected_atom3,
        saved_atom3_position,
    )


def find_angle(point_a, point_b, point_c):
    """
    Function calculates the angle created by two lines connectin 3 points, where point_b is the vertex.

    """

    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)
    # difference
    # ba = a - b
    # bc = c - b
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees


# barton source code
def wikicalculate_dihedral_angle(atom1, atom2, atom3, atom4):

    """
    Function is one of two options to calculate dihedral angles

    Code for torsion restraints from:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    WIKIPEDIA EXAMPLE
    """
    p0 = np.array(atom1, dtype=float)
    p1 = np.array(atom2, dtype=float)
    p2 = np.array(atom3, dtype=float)
    p3 = np.array(atom4, dtype=float)

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    initial_angle = np.degrees(np.arctan2(y, x))
    if initial_angle < 0:
        final_angle = 360 + initial_angle
    else:
        final_angle = initial_angle

    return final_angle


def refactor_find_heavy_bonds(parmed_traj_bonds):

    num_total_bonds = len(parmed_traj_bonds.bonds)

    num_heavy_bonds = 0

    for bond_index in range(num_total_bonds):
        bonded_atom_name = parmed_traj_bonds.bonds[bond_index].atom2.name

        if not bonded_atom_name.startswith("H"):
            num_heavy_bonds += 1

    return num_heavy_bonds > 1


# barton source code
def find_heavy_bonds(mol, j, parmed_traj, traj):

    parmed_atom_j = parmed_traj.atoms[j]

    num_total_bonds = len(parmed_atom_j.bonds)

    num_heavy_bonds = 0
    bond_index = 0

    while bond_index < num_total_bonds:
        # print('bonded atom name: ', parmed_atom_j.bonds[bond_index].atom2.name)
        bonded_atom_name = parmed_atom_j.bonds[bond_index].atom2.name
        bond_index += 1

        if bonded_atom_name.startswith("H") == False:
            num_heavy_bonds += 1

    return num_heavy_bonds


# barton source code
def norm_distance(point_a, point_b, point_c):
    p1 = np.array(point_a)
    p2 = np.array(point_b)
    p3 = np.array(point_c)
    # x = p1-p2
    # dotproduct = np.dot(p3, x)/np.dot(x,x)
    # norm_distance = np.linalg.norm(dotproduct*(x)+p2-p3)

    # norm_distance = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)

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

    # print('norm_distance: ', norm_distance)

    return norm_distance


# def workflow(job, prmtop, coordinate):
#     #get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
#     #get_orientational_restraints(job, complex_prmtop, complex_coordinate, receptor_mask, ligand_mask, restraint_type):
#     #output = job.addChildJobFn(get_orientational_restraints, prmtop, coordinate, ":CB7", ":M01", 1)
#     output = job.addChildJobFn(get_flat_bottom_restraints, prmtop, coordinate, {'r1': 0, "r2": 0, "r3": 10, "r4": 20, "rk2": 0.1, "rk3": 0.1})

if __name__ == "__main__":

    # traj = pt.load("/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/split_complex_folder/ligand/split_M01_000.ncrst.1", "/home/ayoub/nas0/Impicit-Solvent-DDM/success_postprocess/mdgb/M01_000/4/4.0/M01_000.parm7")
    complex_coord = (
        "/home/ayoub/nas0/Impicit-Solvent-DDM/barton_source/cb7-mol02_hmass_298K_lastframe.ncrst"
    )
    complex_parm = (
        "/home/ayoub/nas0/Impicit-Solvent-DDM/barton_source/cb7-mol02_hmass.parm7"
    )
    # screen_for_distance_restraints(num_atoms, com, mol)
    traj = pt.load(complex_coord, complex_parm)
    parmed_traj = pmd.load_file(complex_parm)
    receptor = traj[":CB7"]
    ligand = traj[":M02"]
    ligand_com = pt.center_of_mass(ligand)
    receptor_com = pt.center_of_mass(receptor)

    
    # find atom closest to ligand's CoM and relevand information
    # ligand_suba1, lig_a1_coords, dist_liga1_com = distance_btw_center_of_mass(
    
    receptor_atom_neighbor_index = create_atom_neighbor_array(receptor.xyz[0])
    ligand_atom_neighbor_index = create_atom_neighbor_array(ligand.xyz[0])

    print("receptor_atom_neighbor_index")
    print(receptor_atom_neighbor_index)
    print("-"*80)
    print("ligand_atom_neighbor_index")
    print(ligand_atom_neighbor_index)
    ligand_template = conformational_restraints_template(ligand_atom_neighbor_index)
    receptor_template = conformational_restraints_template(receptor_atom_neighbor_index)
    complex_template = conformational_restraints_template(
        ligand_atom_neighbor_index, num_receptor_atoms=receptor.n_atoms
    )

    # Create a local temporary file.

    # ligand_scratchFile = job.fileStore.getLocalTempFile()
    # receptor_scratchFile = job.fileStore.getLocalTempFile()
    # complex_scratchFile = job.fileStore.getLocalTempFile()
    # # job.log(f"ligand_template {ligand_template}")
    with open("complex_conformational.RST", "w") as fH:
        fH.write(complex_template)
        fH.write(receptor_template)
        fH.write("&end")
    with open("ligand_conformational.RST", "w") as fH:
        fH.write(ligand_template)
        fH.write("&end")
    with open("receptor_conformational", "w") as fH:
        fH.write(receptor_template)
        fH.write("&end")

    # restraint_complex_ID = job.fileStore.writeGlobalFile(complex_scratchFile)
    # restraint_ligand_ID = job.fileStore.writeGlobalFile(ligand_scratchFile)
    # restriant_receptor_ID = job.fileStore.writeGlobalFile(receptor_scratchFile)

    # # job.fileStore.export_file(restraint_complex_ID, "file://" + os.path.abspath(os.path.join("/home/ayoub/nas0/Impicit-Solvent-DDM/output_directory", os.path.basename(restraint_complex_ID))))
    # # toil.exportFile(outputFileID, "file://" + os.path.abspath(os.path.join(ioFileDirectory, "out.txt")))

    # return (restraint_complex_ID, restraint_ligand_ID, restriant_receptor_ID)
