
import os, os.path
import re 
import pandas as pd
import itertools
import pytraj as pt
import sys
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
    data_inputs = {}
    if argSet['parameters']['complex_parameter_filename']:
        parmtop_key = 'complex_parameter_filename'
        coordinate_key = 'complex_coordinate_filename'
        complex_parameter_filename, complex_parameter_basename, complex_coordinate_filename, complex_coordinate_basename = getfiles(toil, argSet, parmtop_key, coordinate_key)
        complex_inputs = {
            'complex_parameter_filename': complex_parameter_filename,
            'complex_parameter_basename': complex_parameter_basename,
            'complex_coordinate_filename': complex_coordinate_filename,
            'complex_coordinate_basename': complex_coordinate_basename
        }
        data_inputs.update(complex_inputs)
    if argSet['parameters']['ligand_parameter_filename']:
        parmtop_key = 'ligand_parameter_filename'
        coordinate_key = 'ligand_coordinate_filename'
        ligand_parameter_filename, ligand_parameter_basename, ligand_coordinate_filename, ligand_coordinate_basename = getfiles(toil, argSet, parmtop_key, coordinate_key)
        ligand_inputs = {
        'ligand_parameter_filename': ligand_parameter_filename, 
        'ligand_parameter_basename': ligand_parameter_basename, 
        'ligand_coordinate_filename': ligand_coordinate_filename, 
        'ligand_coordinate_basename': ligand_coordinate_basename,
        }
        data_inputs.update(ligand_inputs)
        
    if not argSet["ignore_receptor"]:
        parmtop_key = 'receptor_parameter_filename'
        coordinate_key = 'receptor_coordinate_filename'
        receptor_parameter_filename, receptor_parameter_basename, receptor_coordinate_filename, receptor_coordinate_basename = getfiles(toil, argSet, parmtop_key, coordinate_key)
        receptor_inputs = {
            'receptor_parameter_filename' : receptor_parameter_filename,
            'receptor_parameter_basename': receptor_parameter_basename,
            'receptor_coordinate_filename': receptor_coordinate_filename,
            'receptor_coordinate_basename': receptor_coordinate_basename,     
            }
        data_inputs.update(receptor_inputs)
    
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
    
    complex_coodinates = pt.load(argSet["parameters"]["complex_coordinate_filename"][0], argSet["parameters"]["complex_parameter_filename"][0])
    receptor = pt.strip(complex_coodinates, argSet["parameters"]["ligand_mask"][0])
    ligand = pt.strip(complex_coodinates, argSet["parameters"]["receptor_mask"])
     
    
    ligand_name = os.path.join(receptor_ligand_path[0], argSet["parameters"]["ligand_mask"][0].strip(":"))
    receptor_name = os.path.join(receptor_ligand_path[1], argSet["parameters"]["receptor_mask"].strip(":"))
 
    file_number = 0
    while os.path.exists(f"{ligand_name}_{file_number:03}.parm7"):
        file_number +=1    
    pt.write_parm(f"{ligand_name}_{file_number:03}.parm7", ligand.top)
    pt.write_traj(f"{ligand_name}_{file_number:03}.ncrst", ligand)
    
    ligand_inputs = (f"{ligand_name}_{file_number:03}.parm7", f"{ligand_name}_{file_number:03}.ncrst.1")
    argSet["ligand_parameter_filename"] = [ligand_inputs[0]]
    argSet["ligand_coordinate_filename"] = [ligand_inputs[1]]
    
    if not argSet["ignore_receptor"]:
        try:
            pt.write_parm(f"{receptor_name}_{0:03}.parm7",receptor.top)
            pt.write_traj(f"{receptor_name}_{0:03}.ncrst",receptor)
        except:
            sys.exit(f"The receptor file exist {receptor_name}_{0:03}. Use --ignore_receptor flag to prevent duplicate runs")
        receptor_inputs = (f"{receptor_name}_{0:03}.parm7", f"{receptor_name}_{0:03}.ncrst.1")
        argSet["receptor_parameter_filename"] = [receptor_inputs[0]]
        argSet["receptor_coordinate_filename"] = [receptor_inputs[1]]
    else:
        argSet["receptor_parameter_filename"] = [f"{receptor_name}_{0:03}.parm7"]
        
    
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

def get_mdins(config, toil):

    equil_mdins = config["replica_exchange_parameters"]["equilibration_replica_mdins"]
    remd_mdins = config["replica_exchange_parameters"]["remd_mdins"]
    #list of equilibration and remd mdins being imported by toil
    equilibration_import_mdins = []
    remd_import_mdins = []
    
    for equil, remd in itertools.zip_longest(equil_mdins, remd_mdins, fillvalue=-1):
        equilibration_import_mdins.append(toil.importFile("file://" + os.path.abspath(os.path.join(equil))))
        remd_import_mdins.append(toil.importFile("file://" + os.path.abspath(os.path.join(remd))))

    simulation_mdins = {
            "equilibrate_mdins" : equilibration_import_mdins,
            "remd_mdins" : remd_import_mdins
        }
    
    return simulation_mdins

def import_restraint_files(config, toil):
    flat_bottom = config["parameters"]["flat_bottom_restraints"] 
    import_flat_bottom = []
    for restraint in flat_bottom:
        #import_flat_bottom.append(toil.importFile("file://" + os.path.abspath(os.path.join(restraint))))
        import_flat_bottom.append(os.path.abspath(os.path.join(restraint)))
    flat_bottom = {
            "flat_bottom_restraints" : import_flat_bottom
    }
    return flat_bottom

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
    