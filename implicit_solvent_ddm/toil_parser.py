
import os, os.path
import re 
import pandas as pd 
import pytraj as pt
#will not need these imports 
#from toil.common import Toil
#from toil.job import Job

def input_parser(argSet, toil):
    '''
    Parses in all user's config file parameters into a pandas data frame
    
    Dataframe will also contain Toil FileID's for imported files.

    Parameters
    ----------
    argSet: dict 
        Dictionary containing user's config parameters
    toil: class toil.common.Toil
        A context manager that represents a Toil workflow
   
    Returns
    -------
    df: pandas.DataFrame 
       A dataframe containing user's config parameters and imported Toil fileID's
    '''
   
    print('working directory', os.getcwd())

    if argSet['parameters']['complex_parameter_filename']:
        parmtop_key = 'complex_parameter_filename'
        coordinate_key = 'complex_coordinate_filename'
        complex_parameter_filename, complex_parameter_basename, complex_coordinate_filename, complex_coordinate_basename = getfiles(toil, argSet, parmtop_key, coordinate_key)
        
    if argSet['parameters']['ligand_parameter_filename']:
        parmtop_key = 'ligand_parameter_filename'
        coordinate_key = 'ligand_coordinate_filename'
        ligand_parameter_filename, ligand_parameter_basename, ligand_coordinate_filename, ligand_coordinate_basename = getfiles(toil, argSet, parmtop_key, coordinate_key)

    if argSet['parameters']['receptor_parameter_filename']:
        parmtop_key = 'receptor_parameter_filename'
        coordinate_key = 'receptor_coordinate_filename'
        receptor_parameter_filename, receptor_parameter_basename, receptor_coordinate_filename, receptor_coordinate_basename = getfiles(toil, argSet, parmtop_key, coordinate_key)

   
    data_inputs = {
        'ligand_parameter_filename': ligand_parameter_filename, 
        'ligand_parameter_basename': ligand_parameter_basename, 
        'ligand_coordinate_filename': ligand_coordinate_filename, 
        'ligand_coordinate_basename': ligand_coordinate_basename,
        'receptor_parameter_filename' : receptor_parameter_filename,
        'receptor_parameter_basename': receptor_parameter_basename,
        'receptor_coordinate_filename': receptor_coordinate_filename,
        'receptor_coordinate_basename': receptor_coordinate_basename,
        'complex_parameter_filename': complex_parameter_filename,
        'complex_parameter_basename': complex_parameter_basename,
        'complex_coordinate_filename': complex_coordinate_filename,
        'complex_coordinate_basename': complex_coordinate_basename
        }
    
    df = pd.DataFrame(data=data_inputs)

    return df
 

def get_receptor_ligand_topologies(argSet):
    '''
    Create a receptor and ligand topology/coordinate files from a complex topology/coordinate files.

    Method utilizes pytraj library to strip ligand or receptor from complex.

    Parameters
    ----------
    argSet: dict 
        Dictionary containing inital user configuration parameters 

    Returns
    -------
    argSet: dict 
       An updated dictionary containing absolute file path from created ligand and receptor topology/coordinate files 
    '''

    receptor_ligand_path = []
    if not os.path.exists(os.getcwd() + '/mdgb/structs/ligand'):
        os.makedirs(os.getcwd() +'/mdgb/structs/ligand')

    receptor_ligand_path.append(os.getcwd() +'/mdgb/structs/ligand')

    if not os.path.exists(os.getcwd() + '/mdgb/structs/receptor'):
        os.makedirs(os.getcwd() +'/mdgb/structs/receptor')

    receptor_ligand_path.append(os.getcwd() +'/mdgb/structs/receptor')


    ligand_parameter_filename = []
    ligand_coordinate_filename = []
    receptor_parameter_filename = []
    receptor_coordinate_filename = []
    number_complexes = len(argSet["parameters"]["complex_parameter_filename"])
 
    

    for complexes in argSet["parameters"]["complex_parameter_filename"]:

        complex_coodinates = pt.load(argSet["parameters"]["complex_coordinate_filename"][-number_complexes], complexes)
        receptor = pt.strip(complex_coodinates, argSet["parameters"]["ligand_mask"][-number_complexes])
        ligand = pt.strip(complex_coodinates, argSet["parameters"]["receptor_mask"])
        
        file_number = 0
        while os.path.exists(receptor_ligand_path[0] + '/' + f"topology_ligand_{file_number}.parm7"):
            file_number +=1 

        pt.write_parm(receptor_ligand_path[0] + '/' + f"topology_ligand_{file_number}.parm7", ligand.top)
        pt.write_traj(receptor_ligand_path[0]+ '/'+ f"coordinate_ligand_{file_number}.ncrst", ligand)
        ligand_parameter_filename.append(receptor_ligand_path[0] + '/' + f"topology_ligand_{file_number}.parm7")
        ligand_coordinate_filename.append(receptor_ligand_path[0] + '/' + f"coordinate_ligand_{file_number}.ncrst.1")
         
        file_number = 0
        
        while os.path.exists(receptor_ligand_path[1] + '/' + f"topology_receptor_{file_number}.parm7"):
            file_number += 1

        pt.write_parm( receptor_ligand_path[1] + '/' + f"topology_receptor_{file_number}.parm7", receptor.top)
        pt.write_traj(receptor_ligand_path[1] + '/' + f"coordinate_receptor_{file_number}.ncrst", receptor)
        receptor_parameter_filename.append(receptor_ligand_path[1] + '/' + f"topology_receptor_{file_number}.parm7")
        receptor_coordinate_filename.append(receptor_ligand_path[1] + '/' + f"coordinate_receptor_{file_number}.ncrst.1")
        
        number_complexes = number_complexes - 1 

    argSet["ligand_parameter_filename"] = ligand_parameter_filename
    argSet["ligand_coordinate_filename"] = ligand_coordinate_filename
    argSet["receptor_parameter_filename"] = receptor_parameter_filename
    argSet["receptor_coordinate_filename"] = receptor_coordinate_filename

    return argSet

def getfiles(toil, argSet, parm_key, coord_key):
     '''
    Imports all required MD files within a Toil workflow

    Parameters
    ----------
    toil: class toil.common.Toil
        A context manager that represents a Toil workflow
    argSet: dict 
        A dictionary containing user's config parameters 
    parm_key: str 
        A dictionary key name for solute parameter file
    coord_key: str 
        Dictionary key name for solute coordinate file 

    Returns
    ------
    solute_filename: list
        The list will contain toil.fileStore.FileID of the imported file (parameter file)
    solute_basename: list
         List of solute file names (ex: solute.parm7)
    solute_coordinate_filename: list toil.fileStore.FileID
         List of jobStoreFileID of the imported coordinate file
    solute_coordinate_basename: list 
         List of solute coordinate file names (ex: solute.ncrst)
     '''
     solute_filename = []
     solute_basename = []
     solute_coordinate_filename = []
     solute_coordinate_basename = []
     #output_dir = []
     num_of_solutes = len(argSet['parameters'][parm_key])
     for solute in argSet['parameters'][parm_key]:

          solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute)
          solute_filename.append(toil.importFile("file://" + os.path.abspath(os.path.join(solute))))
          solute_basename.append(re.sub(r".*/([^/.]*)",r"\1",solute))
          solute_coordinate_filename.append(toil.importFile("file://" + os.path.abspath(os.path.join(argSet['parameters'][coord_key][-num_of_solutes]))))
          solute_coordinate_basename.append(re.sub(r".*/([^/.]*)",r"\1",argSet['parameters'][coord_key][-num_of_solutes]))
          #output_dir.append(os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(state)))
          num_of_solutes = num_of_solutes -1 

     return solute_filename, solute_basename, solute_coordinate_filename, solute_coordinate_basename
          
def get_output_dir(solute_filename, state):
     '''
     A designated directory path to export output data

     Parameters
     ----------
     solute_filename: str
         The solute file name 
     state: int 
         To designate a unique step/state 

     Returns
     -------
     output_dir: str 
         A path to a specific directory where the output data will be exported. 
     '''
     solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute_filename)
     output_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(state))
     
     return output_dir 
