
import implicit_solvent_ddm.restraint_finder as findrest
import os, os.path
from string import Template

def find_orientational_restraints(job, complex_file, complex_filename, complex_coord, complex_coord_filename, restraint_type, output_dir):

    '''
    To create an orientational restraint .RST file

    The function is an Toil function which will run the restraint_finder module and returns 6 atoms best suited for NMR restraints.

    Parameters
    ----------
    job: Toil Job object
        Units of work in a Toil worflow is a Job
    complex_file: toil.fileStore.FileID
        The jobStoreFileID of the imported file is an parameter file (.parm7) of a complex
    complex_filename: str
        Name of the parameter complex file
    complex_coord: toil.fileStore.FileID
        The jobStoreFileID of the imported file. The file being an coordinate file (.ncrst, .nc) of a complex
    complex_coord_filename: str
        Name of the coordinate complex file
    restraint_type: int
        The type of orientational restraints chosen.
    output_dir: str
        The absolute path where restraint file will be exported to.

    Returns
    -------
    None
    '''
    tempDir = job.fileStore.getLocalTempDir()
    solute = job.fileStore.readGlobalFile(complex_file , userPath=os.path.join(tempDir, complex_filename))
    solute_coordinate = job.fileStore.readGlobalFile(complex_coord, userPath=os.path.join(tempDir, complex_coord_filename))
    atom_R3, atom_R2, atom_R1, atom_L1, atom_L2, atom_L3, dist_rest, lig_angrest, rec_angrest, lig_torres, rec_torres, central_torres = findrest.remote_run_complex(solute, solute_coordinate, 1)

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
            lig_torres = lig_torres
            )

    with open('Restraint.RST', "w") as output:
        output.write(restraint_template)
        
    restraint_file = job.fileStore.writeGlobalFile('Restraint.RST')
    job.fileStore.exportFile(restraint_file, "file://" + os.path.abspath(os.path.join(output_dir, "Restraint.RST")))

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

