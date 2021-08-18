#!/usr/bin/env python

'''
This script is specialized to run MD on the SAMPL4 dataset

It can be run as a standalone MD script with current available options.
It is also run by a wrapper script that implements the current cycle theory developed for mbarton's thesis
This wrapper script run_mdgb.py utilizes the functions defined herein

Most steps in the current cycle use the parm file defined for the original end-state strucutres of the above cycle.
However a few functions will alter the .parm file andplace that file in working directory, these are mostly for what as of this current writing are parm files use in steps 6-Lamba

If restraints are used, they are called by various options and the coordinate files must be created prior to MD run using the script restraint_maker.py

Current option:
   solute: specfies the list of solute parm files to run MD on.  all original files for MD are taken from this directory unless other options specify otherwise. 
   igb: specifies igb setting
   saltcon: ion concentration
   gbsaL specifies GBSA setting
   amber_version: Amber version
   mpi: number of proessors to use if using CPU's
   pbs: queue name
   cuda: for use with GPU's
   runtype: prod or equil, equil is currenlty deactivated
   state_label: A string the helps unqiuely specify the restraint step.  This label is entered as an MD step if the directory structre itslef cannot uniquly determine the state it contains.  This label can be used to alter an entry in the directory structre to any string desired, and then be serached for in post processing.

   SPECIFIES RESTRAINT FORCES  (These pull from a direcotry of pre-created restraint coordinate specifications):
   drest: specifies distance restraint force constant for Boresch Orientational restraints
   arest: specifies angle restraint force constant for Boresch Orientational restraints         
   trest: specifies torsion restraint force constant for Boresch Orientational restraints         
   freeze: specifies conformational restraint force constant 

   OPTIONS that alter the parm file contents:
   charge: alters the charges on the ligand atom.
   add_Exclusions: Uses parmed to exclude interactions between lignad and receptor.
   interpolate: Uses a pre-created set of interpolation parm files to stratify a step within the cycle.  Currenlty only works with turning on and off charges.

'''

#from _typeshed import NoneType
import simrun
import restraint_finder as findrest 
import shutil
import sys
import os, os.path
import re
import parmed as pmd
import logging
import yaml
import subprocess 
from string import Template
from argparse import ArgumentParser
from toil.common import Toil
from toil.job import Job
 
#logging.basicConfig(filename='example.log', level=logging.INFO)
#logging.debug('This message should go to the log file')
#logging.info('So should this')
#logging.warning('And this, too')
#logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

#log = logging.getLogger(__name__)

def initilized_jobs(job):
    #job.log("Hello world, I have a message: {}".format(message))
    job.log('initialize_jobs')

def main(job, solute_file, solute_filename, solute_rst, solute_rst_filename, mdin_file, mdin_filename, output_dir, state_label, argSet,message):
    
    job.log("this is the current job running {}".format(message))
    tempDir = job.fileStore.getLocalTempDir()
    solute = job.fileStore.readGlobalFile(solute_file,  userPath=os.path.join(tempDir, solute_filename))
    rst = job.fileStore.readGlobalFile(solute_rst, userPath=os.path.join(tempDir, solute_rst_filename))
    mdin = job.fileStore.readGlobalFile(mdin_file , userPath=os.path.join(tempDir, 'mdin'))
    if state_label == 9 or 2 or 7:
        restraint_file = job.fileStore.importFile("file://" + write_empty_restraint_file())
        restraint = job.fileStore.readGlobalFile( restraint_file, userPath=os.path.join(tempDir,'restraint.RST'))

    #if state_label == 3:
        # restraint_file = 
    #subprocess.check_call(["mpirun", "-np","2","pmemd.MPI", "-O", "-i", mdin, "-p", solute, "-c", rst])
    subprocess.check_call(["pmemd", "-O", "-i", mdin, "-p", solute, "-c", rst])
    mdout_filename = "mdout"
    mdinfo_filename = "mdinfo"
     restrt_filename = "restrt"
    mdcrd_filename = "mdcrd"

    mdout_file = job.fileStore.writeGlobalFile(mdout_filename)
    mdinfo_file = job.fileStore.writeGlobalFile(mdinfo_filename)
    restrt_file = job.fileStore.writeGlobalFile(restrt_filename)
    mdcrd_file = job.fileStore.writeGlobalFile(mdcrd_filename)
    
    
    job.fileStore.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    job.fileStore.exportFile(mdinfo_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdinfo")))
    job.fileStore.exportFile(restrt_file, "file://" + os.path.abspath(os.path.join(output_dir,"restrt")))
    job.fileStore.exportFile(mdcrd_file, "file://" + os.path.abspath(os.path.join(output_dir,"mdcrd")))
    if message == 'complex':
         return restrt_file
    #job.fileStore.readGlobalFile(mdinfo_file, userPath=os.path.join(output_dir, mdinfo_filename))    


def find_restraints(job, complex_file, complex_filename, complex_coord, complex_coord_filename, restraint_type, output_dir):
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
     
     

def run_simrun(argSet, dirstruct = "dirstruct"):


   struct = simrun.getDirectoryStructure(dirstruct)
   #iterate through solutes

   for key in argSet['parameters']:
       if key == 'complex':
           complex_state = 7
           while complex_state <= 9:
               for complex in argSet['parameters'][key]:
                   argSet['solute'] = complex
                   solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                   pathroot = re.sub(r"\.[^.]*","",argSet['solute'])
                   print('pathroot',pathroot)
                   root = os.path.basename(pathroot)
                   print('root',root)
                   argSet['state_label'] = complex_state
                   run = simrun.getRun(argSet)
               complex_state = complex_state + 1
       if key == 'ligand_parm':
           ligand_state = 2
           while ligand_state <= 5:
               for ligand in argSet['parameters'][key]:
                   argSet['solute'] = ligand
                   solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                   argSet['state_label'] = ligand_state
                   run = simrun.getRun(argSet)
               ligand_state = ligand_state + 1

       if key == 'receptor_parm':
           receptor_state = 2
           while receptor_state <= 5:
               for receptor in argSet['parameters'][key]:
                   argSet['solute'] = receptor
                   solute = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", argSet['solute'])
                   argSet['state_label'] = receptor_state
                   run = simrun.getRun(argSet)
               receptor_state = receptor_state + 1
       

def make_mdin_file(state_label):
    mdin_path =os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)) + "/templates/mdgb.mdin")
    with open(mdin_path) as t:
        template = Template(t.read())
    if state_label == 9:
        final_template = template.substitute(
            nstlim=1000,
            ntx=1,
            irest=0,
            dt=0.001,
            igb = 2,
            saltcon = 0.3,
            gbsa=1,
            temp0=298,
            ntpr=10,
            ntwx=10,
            cut=999,
            nmropt=1
            )
        with open('mdin', "w") as output:
            output.write(final_template)
        return os.path.abspath('mdin')

def getfiles(toil,solute, key, state, solute_coord):
    print('solute', solute)
    solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",solute)
    solute_file =  toil.importFile("file://" + os.path.abspath(os.path.join(solute)))
    print('solute_file', solute_file)
    solute_filename = re.sub(r".*/([^/.]*)",r"\1",solute)
    solute_rst  = toil.importFile("file://" + os.path.abspath(os.path.join(solute_coord)))
    solute_rst_filename =  re.sub(r".*/([^/.]*)",r"\1",solute_coord)
    output_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(state))

    return solute_file, solute_filename, solute_rst, solute_rst_filename, output_dir

def export_files(mdout_file, mdinfo_file, restrt_file, mdcrd_file, output_dir):
    print( "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    Toil.exportFile(mdout_file, "file://" + os.path.abspath(os.path.join(output_dir, "mdout")))
    Toil.exportFile(mdinfo_file,  "file://" + os.path.abspath(os.path.join(output_dir, "mdinfo")))
    Toil.exportFile(restrt_file, "file://" + os.path.abspath(os.path.join(output_dir, "restrt")))
    Toil.exportFile(mdcrd_file, "file://" + os.path.abspath(os.path.join(output_dir,"mdcrd")))


def alter_parm_file(parm7_file, rst7_file, filepath, charge_val, Exclusions):
    
    '''
    #This function is called for parm altering options.
    #Currently those options include:
        add_Exclusions: add exclusions between host an guest
        charge: alters the values of the ligands charges
        '''
    complex_traj = pmd.load_file(parm7_file, xyz = rst7_file)
    print ('parm7_file',parm7_file)
    print ('filepath',filepath)
    if Exclusions == True:
        print('addExclusions turned on!  Excluding ligand/receptor electrostatic interactions')
        complex_traj = pmd.load_file(parm7_file, xyz = rst7_file)
        pmd.tools.actions.addExclusions(complex_traj, '!:CB7', ':CB7').execute()
    if charge_val is not None:
        print("Ligand Charges turned off!")
        pmd.tools.actions.change(complex_traj, 'charge', '!:CB7', charge_val).execute()     
    complex_traj.save(filepath+".parm7")

def use_interpolation_parms(inter_parm, working_dir):
   '''
    This function is for use with interpolating between tunring charges on and off.
    Pre-created parm files are required.

   '''
   print(inter_parm, working_dir)
   run.symlink(inter_parm, working_dir+'.parm7')
    

    
def Add_Exclusions(parm7_file, rst7_file, filepath):
    #This fuction uses parmed's addExclusions to turn off ligand/recptor interactions in a complex
    print ('parm7_file',parm7_file)
    print ('rst7_file',rst7_file)
    print ('filepath',filepath)
    complex_traj = pmd.load_file(parm7_file, xyz = rst7_file)
    #Log this exclusion
    pmd.tools.actions.addExclusions(complex_traj, '!:CB7', ':CB7').execute()  
    complex_traj.save(filepath+".parm7")  
    complex_traj.save(filepath+".ncrst")


def write_empty_restraint_file():
    #This function creates an empty restraint file in the case the no restraints are desired for a current run.
    #This function added by mbarton
    file = open("restraint.RST","w+")
    file.write("")
    file.close()
    return os.path.abspath("restraint.RST")

def create_restraint_file(restraint_filetype, target_directory, root, freeze_force, orient_forces):

    print('restraint_filetype: ',restraint_filetype)
    print('target_directory: ',target_directory)
    #log = logging.getLogger(__name__)
    #log.setLevel(logging.INFO)
    print('root: ',root)
    freeze_force = str(freeze_force)
    orient_forces[0] = str(orient_forces[0])
    orient_forces[1] = str(orient_forces[1])
    orient_forces[2] = str(orient_forces[2])
    print('freeze_force: ', freeze_force)
    print('orientational forces[d,a,t]: ', orient_forces)
    #remove old restraint file if found, this avoids appending restraints into an existing file.
    if os.path.exists(target_directory+"/restraint.RST"):
        print ('Old restraint file detected, deleting file!')
        os.remove(target_directory+"/restraint.RST")
    
    #Freeze restraints only, copy freeze restraint file to target directory, also rewrites freeze force constant to value given
    if restraint_filetype == 1:
        if root.startswith('split_m') or root.startswith('non_split_m') :
            system = root.partition("split_mol")[2]
            print('system: ',system)
            #shutil.copyfile("freeze_restraints_folder/cb7-mol"+system+"_ligand_restraint.FRST", target_directory+'/restraint.RST')
            #re-write force constant value
            file = open(target_directory+'/restraint.RST',"a+")
            rest_file = open("freeze_restraints_folder/cb7-mol"+system+"_ligand_restraint.FRST")
            for line in rest_file:
                line = line.replace("rk2=frest, rk3=frest,", "rk2="+freeze_force+", rk3="+freeze_force+",")
                file.write(line)
        elif root.startswith('cb7-mol'):
            system = root.partition("mol")[2]
            print('system: ',system)
            #shutil.copyfile("freeze_restraints_folder/cb7-mol"+system+"_complex_restraint.FRST", target_directory+'/restraint.RST')
            file = open(target_directory+'/restraint.RST',"a+")
            rest_file = open("freeze_restraints_folder/cb7-mol"+system+"_complex_restraint.FRST")
            for line in rest_file:
                line = line.replace("rk2=frest, rk3=frest,", "rk2="+freeze_force+", rk3="+freeze_force+",")
                file.write(line)
        elif root.startswith('split_cb70') or root.startswith('split_cb71') or root.startswith('non_split_cb70') or root.startswith('non_split_cb71'):
            system = root.partition("split_cb7")[2]
            print('system: ',system)
            #shutil.copyfile("freeze_restraints_folder/cb7-mol"+system+"_receptor_restraint.FRST", target_directory+'/restraint.RST')
            file = open(target_directory+'/restraint.RST',"a+")
            rest_file = open("freeze_restraints_folder/cb7-mol"+system+"_receptor_restraint.FRST")
            for line in rest_file:
                line = line.replace("rk2=frest, rk3=frest,", "rk2="+freeze_force+", rk3="+freeze_force+",")
                file.write(line)
        elif root == 'cb7':
            print ('This is a receptor only')
        else:
            print('/nUnknown restraint filetype, must be 1, 2 or 3, is ', restraint_filetype)
            stop
        print('Log TEST')

        logfile = open(target_directory+'/restraint.log', "a+")
        logfile.write('\n Restraint file should containt only freeze/conformational restraints')
        logfile.write('\n'+str(rest_file.name)+' was copied to '+str(file.name))
        logfile.close()

    #orientational restraints only, copy 6 param rest file to target directory
    elif restraint_filetype == 2:
        system = root.partition("mol")[2]
        rest_file_2 = "orientational_restraints_folder/cb7-mol"+system+"_orientational_restraint.RST"
        if root.startswith('m'):
            print ('\nThis is a ligand only, orientational restraints can only be used on a complex system')
            stop
        elif root.startswith('cb7-mol'):
            shutil.copyfile(rest_file_2, target_directory+'/restraint.RST')
        elif root == 'cb7':
            print ('\nThis is a receptor only, orientational restraints can only be used on a complex system')
            stop
        else:
            print('/nUnknown restraint filetype, must be 1, 2 or 3, is ', restraint_filetype)
            stop
        logfile = open(target_directory+'/restraint.log', "a+")
        logfile.write('\n Restraint file should containt only orientational  restraints')
        logfile.write('\n'+rest_file_2+" was copied to "+target_directory+"/restraint.RST")
        logfile.close()

    #Both freeze and 6 parameter orientational restraints, concatonate both freeze and 6 param restraints, write contcatonated file to target directory
    elif restraint_filetype == 3:
        #print('Combined Freeze and Orientational restraints not currenlty supported')
        system = root.partition("mol")[2]
        print('system: ',system)
        file = open(target_directory+'/restraint.RST',"a+")
        orient_rest_file = open("orientational_restraints_folder/cb7-mol"+system+"_orientational_restraint.RST")
        #orient_rest_file = orient_rest_file.replace('&end','')#eliminate EOF string on first part of new file
        freeze_rest_file = open("freeze_restraints_folder/cb7-mol"+system+"_complex_restraint.FRST")
        #orient_rest_content = orient_rest_file.readlines()
        line_replace_counter = 1
        for line in orient_rest_file:
            line = line.replace("&end", "")
            line = line.replace("rk2=drest, rk3=drest,", "rk2="+orient_forces[0]+", rk3="+orient_forces[0]+",")
            line = line.replace("rk2=arest, rk3=arest,", "rk2="+orient_forces[1]+", rk3="+orient_forces[1]+",")
            line = line.replace("rk2=trest, rk3=trest,", "rk2="+orient_forces[2]+", rk3="+orient_forces[2]+",")
            file.write(line)
        for line in freeze_rest_file:
            line = line.replace("rk2=frest, rk3=frest,", "rk2="+freeze_force+", rk3="+freeze_force+",")
            file.write(line)

        logfile = open(target_directory+'/restraint.log', "a+")
        logfile.write('\n Restraint file should containt both freeze and orientational restraints')
        logfile.write('\n'+str(orient_rest_file.name)+" and "+str(freeze_rest_file.name)+" were concatinated into "+str(file.name))
        logfile.close()
        #log.info(orient_rest_file+" and "+freeze_rest_file+" were concatinated into "+file)
        #orient_rest_content.rstrip('&end')  #orient_rest_content.replace('&end','')#eliminate EOF string on first part of new file
        freeze_rest_content = freeze_rest_file.readlines()
        #concat_rest_string = orient_rest_content + freeze_rest_content
        #for line in freeze_rest_content:
        #    line = line.replace("rk2=50.0, rk3=50.0,", "rk2="+freeze_force+", rk3="+freeze_force+",")
        #file.writelines(freeze_rest_content)
        #file.write("\n&end")
        file.close()
        #print('New file string', concat_rest_string)
        
        

    else:
        print("Invalid restraint_filetype passed to function create_restraint_file. \n"+restraint_filetype+" was passed, should be 1,2 or 3.")

simrun = simrun.SimRun("mdgb", description = '''Perform molecular dynamics with GB or in vacuo''')

if __name__ == "__main__":
    
   """Creates directory structure and runs MD siulations for entire free energy cycle.
   
   Parameters
   ----------
   None
   
   """
   parser = Job.Runner.getDefaultArgumentParser()
   parser.add_argument('--config_file', nargs='*', type=str, required=True, help="configuartion file with input parameters")
   options = parser.parse_args()
   options.logLevel = "INFO"
   options.clean = "always"
   
   config = options.config_file[0]
   with open(config) as file:
       config = yaml.load(file,Loader=yaml.FullLoader)
   argSet = {}
   argSet.update(config)

   run_simrun(argSet)

   with Toil(options) as toil:
        
       #check whether an mdin file was provided 
       if argSet['parameters']['mdin'] is None:
           file_name = 'mdin'
           print('No mdin file specified. Generating one automaticalled called: %s' %str(file_name))
           mdin = make_mdin_file(state_label=9)
           mdin_file = toil.importFile("file://" + os.path.abspath(os.path.join(mdin)))
           mdin_filename= 'mdin'
           
       try: parent_job
       except NameError: parent_job = None 
       complex_outputs = {}
       #iterate through the parameters within the config file 
       #for key in argSet['parameters']:
       if argSet['parameters']['ligand_parm']:
           key = 'ligand_parm'
           num_of_ligands = len(argSet['parameters'][key])
           ligand_state = 2
           for ligand in argSet['parameters'][key]:
               ligand_file, ligand_filename, ligand_rst, ligand_rst_filename, output_dir = getfiles(toil,ligand,key, ligand_state, argSet['parameters']['ligand_coord'][-num_of_ligands])
               if parent_job is None:
                   parent_job = Job.wrapJobFn(main, ligand_file, ligand_filename, ligand_rst, ligand_rst_filename, mdin_file, mdin_filename, output_dir, ligand_state , argSet, "ligand job")
                   #parent_job.addChildJobFn(main, ligand_file, ligand_filename, ligand_rst, ligand_rst_filename, mdin_file, mdin_filename, output_dir, ligand_state , argSet, "ligand job")
                   #ligand job will be wrap into a child function 
                   #ligand_job = main_job.addChildJobFn(main, ligand_file, ligand_filename, ligand_rst, ligand_rst_filename, mdin_file, mdin_filename, output_dir, ligand_state , argSet, "ligand job")
               else:
                   parent_job.addChildJobFn(main, ligand_file, ligand_filename, ligand_rst, ligand_rst_filename, mdin_file, mdin_filename, output_dir, ligand_state , argSet, "ligand job")
                   
               num_of_ligands = num_of_ligands -1 

       if argSet['parameters']['receptor_parm']:
           key = 'receptor_parm'
           num_of_receptors = len(argSet['parameters'][key])
           receptor_state = 2
           for receptor in argSet['parameters'][key]:
               receptor_file,receptor_filename, receptor_rst, receptor_rst_filename, output_dir = getfiles(toil,receptor,key,receptor_state, argSet['parameters']['receptor_coord'][-num_of_receptors])

                   #receptor_job = ligand_job.addChildJobFn(main, receptor_file,receptor_filename, receptor_rst, receptor_rst_filename, mdin_file, mdin_filename, output_dir, receptor_state, argSet, "receptor job")
               parent_job.addChildJobFn(main, receptor_file,receptor_filename, receptor_rst, receptor_rst_filename, mdin_file, mdin_filename, output_dir, receptor_state, argSet, "receptor job")
                   #receptor_job = main_job.addChildJobFn(main, receptor_file,receptor_filename, receptor_rst, receptor_rst_filename, mdin_file, mdin_filename, output_dir, receptor_state, argSet, "receptor job")
               num_of_receptors = num_of_receptors -1 

       if argSet['parameters']['complex']:
           key = 'complex'
           num_of_complexes = len(argSet['parameters'][key])
           complex_state = 9
           for complexes in argSet['parameters'][key]:
               complex_file, complex_filename, complex_rst, complex_rst_filename, output_dir = getfiles(toil,complexes,key, complex_state, argSet['parameters']['complex_rst'][-num_of_complexes])
               # a job from input files
               complex_job = parent_job.addChildJobFn(main, complex_file, complex_filename, complex_rst, complex_rst_filename, mdin_file, mdin_filename,output_dir, complex_state , argSet, "complex")
               
               solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1", complexes)
               
               #print('complex_file' , complex_file)
               #print('complex_job.rv(0)', complex_job.rv(0))
               restraint_job = complex_job.addFollowOnJobFn(find_restraints, complex_file, complex_filename, complex_job.rv(), complex_rst_filename, 1, os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(7)))
               
               
                       #complex_outputs[re.sub(r".*/([^/.]*)\.[^.]*",r"\1", complexes)] = main_job.rv()
               
                       #restraints_job = main_job.addChildJobFn(

#main_job.addChildJobFn(main, complex_file, complex_filename, main_job.rv(), "restrt", mdin_file, mdin_filename, os.path.join(os.path.dirname(os.path.abspath('__file__')),'mdgb/'+ solu + '/' + str(7)), 7 , argSet, "complex")
                     

               num_of_complexes = num_of_complexes - 1


       toil.start(parent_job)
       #toil.start(ligand_job) 
