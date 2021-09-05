
import os, os.path
import implicit_solvent_ddm.restraints as restraints
import re 
import pandas as pd 

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
     print("argSet['parameters']['mdin']", argSet['parameters']['mdin'])
       #check whether an mdin file was provided
        #try: parent_job
        #except NameError: parent_job = None
     
   #  parent_job = Job.wrapJobFn(initilized_jobs)
       #iterate through the parameters within the config file
       #for key in argSet['parameters']:
   
     if argSet['parameters']['ligand_parm']:
          parmtop_key = 'ligand_parm'
          coordinate_key = 'ligand_coord'
          ligand_file, ligand_filename, ligand_rst, ligand_rst_filename = getfiles(toil, argSet, parmtop_key, coordinate_key)
   
     if argSet['parameters']['receptor_parm']:
          parmtop_key = 'receptor_parm'
          coordinate_key = 'receptor_coord'

          receptor_file, receptor_filename, receptor_rst, receptor_rst_filename = getfiles(toil, argSet, parmtop_key, coordinate_key)

     if argSet['parameters']['complex']:
          parmtop_key = 'complex'
          coordinate_key = 'complex_rst'

          complex_file, complex_filename, complex_rst, complex_rst_filename = getfiles(toil, argSet, parmtop_key, coordinate_key)

     data_inputs = {
          'ligand_file': ligand_file, 
          'ligand_filename': ligand_filename, 
          'ligand_rst': ligand_rst, 
          'ligand_rst_filename': ligand_rst_filename,
          'receptor_file' : receptor_file,
          'receptor_filename': receptor_filename,
          'receptor_rst': receptor_rst,
          'receptor_rst_filename': receptor_rst_filename,
          'complex_file': complex_file,
          'complex_filename': complex_filename,
          'complex_rst': complex_rst,
          'complex_rst_filename': complex_rst_filename
          }
     
     df = pd.DataFrame(data=data_inputs)

     return df
 
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
    solute_file: list
        The list will contain toil.fileStore.FileID of the imported file (parameter file)
    solute_filename: list
         List of solute file names (ex: solute.parm7)
    solute_rst_file: list toil.fileStore.FileID
         List of jobStoreFileID of the imported coordinate file
    solute_rst_filename: list 
         List of solute coordinate file names (ex: solute.ncrst)
     '''
     solute_file = []
     solute_filename = []
     solute_rst_file = []
     solute_rst_filename = []
     #output_dir = []
     num_of_solutes = len(argSet['parameters'][parm_key])
     for solute in argSet['parameters'][parm_key]:

          solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute)
          solute_file.append(toil.importFile("file://" + os.path.abspath(os.path.join(solute))))
          solute_filename.append(re.sub(r".*/([^/.]*)",r"\1",solute))
          solute_rst_file.append(toil.importFile("file://" + os.path.abspath(os.path.join(argSet['parameters'][coord_key][-num_of_solutes]))))
          solute_rst_filename.append(re.sub(r".*/([^/.]*)",r"\1",argSet['parameters'][coord_key][-num_of_solutes]))
          #output_dir.append(os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(state)))
          num_of_solutes = num_of_solutes -1 

     return solute_file, solute_filename, solute_rst_file, solute_rst_filename
          
def get_output_dir(solute, state):
     '''
     A designated directory path to export output data

     Parameters
     ----------
     solute: str
         The solute file name 
     state: int 
         To designate a unique step/state 

     Returns
     -------
     output_dir: str 
         A path to a specific directory where the output data will be exported. x
     '''
     solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute)
     output_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(state))
     
     return output_dir 
