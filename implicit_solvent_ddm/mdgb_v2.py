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

import simrun
import shutil
import sys
import os, os.path
import re
#import parmed as pmd
import logging

#logging.basicConfig(filename='example.log', level=logging.INFO)
#logging.debug('This message should go to the log file')
#logging.info('So should this')
#logging.warning('And this, too')
#logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
log = logging.getLogger(__name__)

#import restraint_finder as findrest -SA removed for now



def alter_parm_file(parm7_file, rst7_file, filepath, charge_val, Exclusions):
    '''
    This function is called for parm altering options.
    Currently those options include:
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
    

    
#def Add_Exclusions(parm7_file, rst7_file, filepath):
    #This fuction uses parmed's addExclusions to turn off ligand/recptor interactions in a complex
    #print ('parm7_file',parm7_file)
    #print ('rst7_file',rst7_file)
    #print ('filepath',filepath)
    #complex_traj = pmd.load_file(parm7_file, xyz = rst7_file)
    #Log this exclusion
    #pmd.tools.actions.addExclusions(complex_traj, '!:CB7', ':CB7').execute()  
    #complex_traj.save(filepath+".parm7")  
    #complex_traj.save(filepath+".ncrst")

def write_empty_restraint_file():
    #This function creates an empty restraint file in the case the no restraints are desired for a current run.
    #This function added by mbarton
    file = open("restraint.RST","w+")
    file.write("")
    file.close()

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

for argSet in simrun.args.generateSets():

    # pathroot = pathname entered for solute on command line
    pathroot = re.sub(r"\.[^.]*$","",argSet["solute"]) 
    #pathroot = pathroot.replace('split_', '')
    #sourcepath = os.path.dirname(pathroot) #Sourc direcotry path
    root = os.path.basename(pathroot) #source filename
    #root = root.replace('split_', '')
    cuda = argSet["cuda"]  #This is here, but is flipped and repeated later 
    replica = argSet["replica"]
    freeze_restraint = str(argSet["freeze"]) #this will be a number that represents the force constant
    drest = str(argSet["drest"])
    arest = argSet["arest"]
    trest = argSet["trest"]
    orient_rest = [drest, arest, trest]
    addExc = argSet["add_Exclusions"]
    runtype = argSet["runtype"]
    chrg = argSet["charge"]
    interpolate = argSet["interpolate"]
    #r1_input_type = state_label
    
    

    print("cuda:", cuda)
    print("root:",root)
    print("replica:", replica)
   
    argSet["systembasename"] = root
    #dirstruct_equil_input = simrun.getDirectoryStructure("dirstruct-gb-input")
    #if runtype = "prod", set input dirstruct = "equil output dirstruct"
    struct = simrun.getDirectoryStructure("dirstruct-gb-input")
    #print ("root: ", root) 
    argSet['cuda'] = cuda #Why is this here, but repeated from above.
     
    # run and directory
    if runtype == 'equil':
        run = simrun.getRun(argSet, dirstruct="dirstruct-equil-ouput")
        nstlim = 1000000
    elif runtype == 'prod':
        run = simrun.getRun(argSet, dirstruct="dirstruct-prod-ouput")
        #nstlim = 1000000 #nanosecond if dt = 0.001
        nstlim = 10000000 #10 nanoseconds if dt = 0.001
        #nstlim = 20000000 #20 nanoseconds if dt = 0.001
        #nstlim = 40000000 #40 nanoseconds if dt = 0.001
        #nstlim = 80000000 #80 nanoseconds if dt = 0.001
        #nstlim = 100000000 #100ns if dt = 0.001
    else:
        print("No such runtype, runtyp must be 'equil' or 'prod'")
        break

    #run and directory
    #run = simrun.getRun(argSet)
    print ("run.path:", run.path)
    #print("targetpath:", targetpath)
    working_filepath = os.path.join(run.path,root)
    #print ("filepath:", os.path.join(run.path,root))
    print ("working_filepath:", working_filepath)

    sfx=""
    if argSet["mpi"]:
        sfx=".MPI"
    solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",argSet["solute"])

    
    # time steps
    dt = 0.001 # 0.001 = femtosecond, 1 = picosecond
    #steps_per_ps = int(1/dt)
    ntpr = int(1/dt) 
    #ntpr = 1
    ntwx = ntpr
    ntr = 0

    if cuda == False:
        executable = "sander"
    elif cuda == True:
        executable = "pmemd.cuda"
    #if restraint_wt == None:
        #ntr = 0
    #elif restraint_wt != None:
        #ntr = 1
    print('drest: ', drest)
    print('freeze_restraint:', freeze_restraint)
    if drest != '0' or freeze_restraint != '0':
        nmropt = 1
        if drest == '0' and freeze_restraint != '0':
            create_restraint_file(1, run.path, root, freeze_restraint, orient_rest)
        elif drest != '0' and freeze_restraint == '0':
            create_restraint_file(2, run.path, root, freeze_restraint, orient_rest)
        elif drest != '0' and freeze_restraint != '0':
            create_restraint_file(3, run.path, root, freeze_restraint, orient_rest)
        else:
            print("invalid restraint input")
            stop
    elif drest == '0' and arest == '0' and trest == '0' and freeze_restraint == '0':
        nmropt = 0
        print ("Running with no restraints")
        write_empty_restraint_file()
    irest = 0 #if irest = 0, initial velocities are assigned randomly rather than taken from a restart file.
    ntx = 1
    restart = ""
    #irest = 1
    #ntx=5
    fceread = 1
    #restart = ".rst"
    #try:
    #    if runtype == 'equil' or run.runnumber == 0:
    #        irest = 0
    #        ntx = 1
    #        fceread = 0
    #        restart = ""
    #except:
    #    pass

    template = executable+restart+'.cmd'
    print ('template:', template)
    
    #Add Exclusions if exclusion flag is turned on
    ##Only step 4 and 6 have a different .parm7 file, it is created here if assExc = True, else it is copied from 
    print("addExc: ", addExc)
    print("chrg", chrg)
    if addExc == True or chrg is not None:
        print("Hurray!!!")
        alter_parm_file(pathroot+'.parm7', pathroot+'.ncrst', working_filepath, chrg, addExc)

    # if interpolations flag is on, inerpoaltion parm files are copied from repo direcotry to working directory
    elif interpolate is not None and addExc == False and chrg is None:
        #run.symlink(inter_parm, working_dir+'.parm7')
        use_interpolation_parms(interpolate, working_filepath)
        
    # if no parm altering option above are turned on, parm file is copied from original strucutre root direcotry
    elif addExc == False and chrg is None and interpolate is None:
        print('addExclusions turned off!,  Charges not altered.  Using Original input parm file for MD run.')
        run.symlink(pathroot+'.parm7',working_filepath+'.parm7')
    
    else:
        print("Invalid paramaters")
        System.exit("Invalid paramaters")
        
    #All MD's start with their pathroot .ncrst file
    run.symlink(pathroot+'.ncrst',working_filepath+'.ncrst')
    
    run.writeTemplate("mdgb.mdin",
                      dt = dt, ntr = ntr, nstlim = nstlim,
                      ntpr = ntpr, ntwx = ntwx,
                      irest = irest, ntx = ntx, fceread=fceread,
                      nmropt = nmropt, fcewrite = argSet["nstlim"]
                      )
    
    cmds = ""
    if argSet["pbs"] is None:
        cmds = " ".join(run.fillTemplate("module"))+"\n"
    cmds += " ".join(run.fillTemplate(template,
                             SFX=sfx,
                             pathroot=working_filepath,
                             root=root,
                             ))
    
    run.submitTemplate(cmds, setup="module add "+argSet["amber_version"],
                             solv=argSet['igb'],solu=solu
                       )

