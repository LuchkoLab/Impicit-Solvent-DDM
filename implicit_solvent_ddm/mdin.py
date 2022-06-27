import itertools
import os
from dataclasses import dataclass
from string import Template
from tkinter import Y

import yaml


def get_mdins(job, user_mdin_args):
    """Writes all mdins for intermidate states 

    Parameters
    ----------
    user_mdin_args: str
        A user specified yaml file containing mdin arguments  

    Returns:
    --------
    default_mdin: 
    no_solvent_mdin:
    post_mdin:
    post_nosolv: 
    """
    temp_dir = job.fileStore.getLocalTempDir()
   
    default_mdin = job.fileStore.writeGlobalFile(make_mdin_file(user_mdin_args, '_mdin'))
    no_solvent_mdin = job.fileStore.writeGlobalFile(make_mdin_file(user_mdin_args, 'no_solv_mdin', turn_off_solvent=True))
    post_mdin = job.fileStore.writeGlobalFile(make_mdin_file(user_mdin_args, 'post_mdin', post_process=True))
    post_nosolv = job.fileStore.writeGlobalFile(make_mdin_file(user_mdin_args, 'post_nosolv_mdin', turn_off_solvent=True, post_process=True))
    
    return (default_mdin, no_solvent_mdin, post_mdin, post_nosolv)

def make_mdin_file(yaml_args, mdin_name, turn_off_solvent=False, post_process=False):
    """ Creates an molecular dynamics input file

    Function will fill a template and write an MD input file

    Parameters
    ----------
    yaml_args: yaml.File
        A user specified yaml file which contains mdin args
    mdin_name: str
        A unique mdin filename 
    turn_off_solvent: bool
        Set igb=6 if turn_off_solvent=True
    post_process: bool
        Set imin=5 and ntx=5 if post_process=True 
    Returns
    -------
    mdin: str
        Absolute path where the MD input file was created.
    """
    with open(yaml_args) as fH:
        mdin_args = yaml.safe_load(fH)
    
    mdin_path = os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)) + "/templates/mdgb.mdin")
    
    #general setting 
    imin = 0 
    ioutfm = 0
    if post_process:
        imin = 5
        ioutfm = 1
    with open(mdin_path) as t:
        template = Template(t.read())
    if turn_off_solvent: 
        final_template = template.substitute(
            imin=imin,
            nstlim=mdin_args["nstlim"],
            ntx=1,
            irest=0,
            ioutfm=ioutfm,
            dt=mdin_args["dt"],
            igb = 6,
            saltcon = 0.0,
            rgbmax=mdin_args["rgbmax"],
            gbsa=mdin_args["gbsa"],
            temp0=mdin_args["temp0"],
            ntpr=mdin_args["ntpr"],
            ntwx=mdin_args["ntwx"],
            cut=mdin_args["cut"],
            ntc= mdin_args["ntc"],
            nmropt=1,
            restraint= "$restraint"
            )
        
    else:
        final_template = template.substitute(
            imin=imin,
            nstlim=mdin_args["nstlim"],
            ntx=1,
            irest=0,
            ioutfm=ioutfm,
            dt=mdin_args["dt"],
            igb =mdin_args["igb"],
            saltcon =mdin_args["saltcon"],
            rgbmax=mdin_args["rgbmax"],
            gbsa=mdin_args["gbsa"],
            temp0=mdin_args["temp0"],
            ntpr=mdin_args["ntpr"],
            ntwx=mdin_args["ntwx"],
            cut=mdin_args["cut"],
            ntc= mdin_args["ntc"],
            nmropt=1,
            restraint= "$restraint"
            )


    with open(mdin_name, "w") as output:
        output.write(final_template)
    return os.path.abspath(mdin_name)
 