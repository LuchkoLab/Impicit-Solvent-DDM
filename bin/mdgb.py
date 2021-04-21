#!/usr/bin/env python

'''
mbartons additions 

Added templates/option to run on CUDA(GPU's)
    added cuda boolean option to json
Run replica pairs to find threshold times
Added TI/icfe options to this file
Added restraint file writing functions and options
'''

import simrun
import os, os.path
import re
import yaml 
#import parmed as pmd
import logging
log = logging.getLogger(__name__)

#import restraint_finder as findrest 

def Add_Exclusions(parm7_file, rst7_file, filepath):
    #This fuction uses parmed's addExclusions to turn off ligand/recptor interactions in a complex
    complex_traj = pmd.load_file(parm7_file, xyz = rst7_file)
    #Log this exclusion
    pmd.tools.actions.addExclusions(complex_traj, '!:CB7', ':CB7').execute()  
    complex_traj.save(filepath+".parm7")  
    complex_traj.save(filepath+".rst7")

def write_empty_restraint_file():
    #This function creates an empty restraint file in the case the no restraints are desired for a current run.
    #This function added by mbarton
    file = open("restraint.RST","w+")
    file.write("")
    file.close()

def write_complex_restraints(parm7_file, rst7_file, r1_input_type):
    #This finds the restraints of a complex.  It currenlty it wont run on a ligand.                                            
    #This function added by mbarton                           
    #Logging example from bin/simrun/pbssubmitter.py
    #self.log.info("Chaining to previous submission "+self.lastJobID+" using : "+self.args["pbschain"])


    atom_R3, atom_R2, atom_R1, atom_L1, atom_L2, atom_L3, dist_rest, lig_angrest, rec_angrest, lig_torres, rec_torres, central_torres = findrest.remote_run_complex(pathroot+'.parm7', pathroot+'.rst7', r1_input_type)

    run.writeTemplate("restraint.RST",
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
    
    #pathroot = re.sub(r"\.[^.]*$","",argSet["solute"])
    #dirname = os.path.dirname(pathroot)
    #run.copy(dirname+"/restraint.RST", dirname+"/restraints_test_written.trst")

simrun = simrun.SimRun("mdgb", description = '''Perform molecular dynamics with GB or in vacuo''')

for argSet in simrun.args.generateSets():
    #read in config yaml file 
    with open(argSet['config']) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    #update argset with config input parameters 
    argSet.update(config['initial_MD'])

    
    # pathroot = pathname entered for solute on command line
    pathroot = re.sub(r"\.[^.]*$","",argSet["solute"])
    root = os.path.basename(pathroot)
    cuda = argSet["cuda"]  #This is here, but is flipped and repeated later 
    replica = argSet["replica"]
    restraint_type = argSet["restraint_type"]
    addExc = argSet["add_Exclusions"]
    runtype = argSet["runtype"]
    r1_input_type = argSet["r1"]

    print("cuda:", cuda)
    print("root:",root)
    print("replica:", replica)
   
    argSet["systembasename"] = root
    #dirstruct_equil_input = simrun.getDirectoryStructure("dirstruct-gb-input")
    #if runtype = "prod", set input dirstruct = "equil output dirstruct"
    struct = simrun.getDirectoryStructure("dirstruct-gb-input")
    #print ("Struct: ", struct.args) #THIS DOES NOT HELP WRITE THE NEW DIRECTORY
    print ("root: ", root) #THIS DOES NOT HELP WRITE THE NEW DIRECTORY
    #print ("filepath:", os.path.join(run.path,root+'.parm7'))
    #print ("pathroot: ", pathroot) THIS DOES NOT HELP WRITE THE NEW DIRECTORY
    #argSet['forcefield'] = struct.fromPath2Dict(os.path.dirname(argSet['solute']))['forcefield']
    argSet['cuda'] = cuda #Why is this here, but repeated from above, I am following suit
     
    # run and directory
    if runtype == 'equil':
        run = simrun.getRun(argSet, dirstruct="dirstruct-equil-ouput")
        nstlim = 1000000
    elif runtype == 'prod':
        run = simrun.getRun(argSet, dirstruct="dirstruct-prod-ouput")
        nstlim = 100000000
    else:
        print("No such runtype, runtyp must be 'equil' or 'prod'")
        break

    # run and directory
    #run = simrun.getRun(argSet)
    print ("run:", run.path)
    print ("filepath:", os.path.join(run.path,root))

    sfx=""
    if argSet["mpi"]:
        Sfx=".MPI"
    solu = re.sub(r".*/([^/.]*)\.[^.]*",r"\1",argSet["solute"])

    
    # time steps
    dt = 0.001
    steps_per_ps = int(1./dt)
    ntpr = int(10/dt) # once per 10 ps
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
    if restraint_type != 0:
        nmropt = 1
    elif restraint_type == 0:
        nmropt = 0
    #if run.runnumber == 0:
     #   irest = 0
      #  ntx = 1
       # fceread = 0
    restart = ""
    #else:
      #  irest = 1
     #   ntx=5
       # fceread = 1
       # restart = ".rst"

    template = executable+restart+'.cmd'
    print ('template:', template)

    #write restraint file per specified restraints (Complex, Ligand or None)
    print('restraint_type:', restraint_type)
    if restraint_type == 2:
        write_complex_restraints(pathroot+'.parm7', pathroot+'.rst7', r1_input_type)
    elif restraint_type == 1:
        write_ligand_restraints(pathroot+'.parm7', pathroot+'.rst7')
    elif restraint_type == 0:
        print ("Running with no restraints")
        write_empty_restraint_file()

    #Add Exclusions if exclusion flag is turned on
    if addExc == True:
        print('addExclusions turned on!  Excluding ligand/receptor electrostatic interactions')
        Add_Exclusions(pathroot+'.parm7', pathroot+'.rst7', os.path.join(run.path,root))

    elif addExc == False:
        print('addExclusions turned off!')
        run.symlink(pathroot+'.parm7',os.path.join(run.path,root+'.parm7'))
        run.symlink(pathroot+'.rst7',os.path.join(run.path,root+'.rst7'))
                                                                                                                                              
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
                             pathroot=pathroot,
                             root=root,
                             ))
    run.submitTemplate(cmds, setup="module add "+argSet["amber_version"],
                             solv=argSet['igb'],solu=solu
                       )


