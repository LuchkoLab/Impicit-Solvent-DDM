import itertools
import os
import random
from dataclasses import dataclass
from string import Template
from tkinter import Y

import yaml
from toil.common import FileID

DEFAULT_MDIN_ARGS = {
    "nstlim": 5000000,
    "dt": 0.002,
    "saltcom": 0.3,
    "rgbmax": 999.0,
    "igb": 2,
    "gbsa": 0,
    "temp0": 298,
    "ntpr": 250,
    "ntwx": 250,
    "cut": 999,
    "ntc": 2,
}


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

    default_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(user_mdin_args, "_mdin")
    )
    no_solvent_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(user_mdin_args, "no_solv_mdin", turn_off_solvent=True)
    )
    post_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(user_mdin_args, "post_mdin", post_process=True)
    )
    post_nosolv = job.fileStore.writeGlobalFile(
        make_mdin_file(
            user_mdin_args, "post_nosolv_mdin", turn_off_solvent=True, post_process=True
        )
    )

    return (default_mdin, no_solvent_mdin, post_mdin, post_nosolv)


def make_mdin_file(mdin_args, mdin_name, turn_off_solvent=False, post_process=False):
    """Creates an AMBER format input file

    Function will fill a template and write an MD input file

    Parameters
    ----------
    mdin_args: dict
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
    # with open(yaml_args) as fH:
    #     mdin_args = yaml.safe_load(fH)

    mdin_path = os.path.abspath(
        os.path.dirname(os.path.realpath(__file__)) + "/templates/mdgb.mdin"
    )

    temp_mdin_args = DEFAULT_MDIN_ARGS.copy()
    temp_mdin_args.update(mdin_args)

    # general setting
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
            nstlim=temp_mdin_args["nstlim"],
            ntx=1,
            irest=0,
            ioutfm=ioutfm,
            dt=temp_mdin_args["dt"],
            igb=6,
            saltcon=0.0,
            extdiel=0.0,
            rgbmax=temp_mdin_args["rgbmax"],
            gbsa=temp_mdin_args["gbsa"],
            temp0=temp_mdin_args["temp0"],
            ntpr=temp_mdin_args["ntpr"],
            ntwx=temp_mdin_args["ntwx"],
            cut=temp_mdin_args["cut"],
            ntc=temp_mdin_args["ntc"],
            nmropt=1,
            restraint="$restraint",
        )

    else:
        final_template = template.substitute(
            imin=imin,
            nstlim=temp_mdin_args["nstlim"],
            extdiel=78.5,
            ntx=1,
            irest=0,
            ioutfm=ioutfm,
            dt=temp_mdin_args["dt"],
            igb=temp_mdin_args["igb"],
            saltcon=temp_mdin_args["saltcon"],
            rgbmax=temp_mdin_args["rgbmax"],
            gbsa=temp_mdin_args["gbsa"],
            temp0=temp_mdin_args["temp0"],
            ntpr=temp_mdin_args["ntpr"],
            ntwx=temp_mdin_args["ntwx"],
            cut=temp_mdin_args["cut"],
            ntc=temp_mdin_args["ntc"],
            nmropt=1,
            restraint="$restraint",
        )

    with open(mdin_name, "w") as output:
        output.write(final_template)
    return os.path.abspath(mdin_name)


def make_mdin(job, mdin_args, extdiel=78.5, turn_off_solvent=False, post_process=False):
    """Creates an AMBER format input file

    Function will fill a template and write an MD input file

    Parameters
    ----------
    mdin_args: dict
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
    # with open(yaml_args) as fH:
    #     mdin_args = yaml.safe_load(fH)
    scratchFile = job.fileStore.getLocalTempFile()
    
    mdin_path = os.path.abspath(
        os.path.dirname(os.path.realpath(__file__)) + "/templates/mdgb.mdin"
    )

    temp_mdin_args = DEFAULT_MDIN_ARGS.copy()
    temp_mdin_args.update(mdin_args)

    # general setting
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
            extdiel=0.0,
            nstlim=temp_mdin_args["nstlim"],
            ntx=1,
            irest=0,
            ioutfm=ioutfm,
            dt=temp_mdin_args["dt"],
            igb=6,
            saltcon=0.0,
            rgbmax=temp_mdin_args["rgbmax"],
            gbsa=temp_mdin_args["gbsa"],
            temp0=temp_mdin_args["temp0"],
            ntpr=temp_mdin_args["ntpr"],
            ntwx=temp_mdin_args["ntwx"],
            cut=temp_mdin_args["cut"],
            ntc=temp_mdin_args["ntc"],
            nmropt=1,
            restraint="$restraint",
        )

    else:
        final_template = template.substitute(
            imin=imin,
            extdiel=extdiel,
            nstlim=temp_mdin_args["nstlim"],
            ntx=1,
            irest=0,
            ioutfm=ioutfm,
            dt=temp_mdin_args["dt"],
            igb=temp_mdin_args["igb"],
            saltcon=temp_mdin_args["saltcon"],
            rgbmax=temp_mdin_args["rgbmax"],
            gbsa=temp_mdin_args["gbsa"],
            temp0=temp_mdin_args["temp0"],
            ntpr=temp_mdin_args["ntpr"],
            ntwx=temp_mdin_args["ntwx"],
            cut=temp_mdin_args["cut"],
            ntc=temp_mdin_args["ntc"],
            nmropt=1,
            restraint="$restraint",
        )

    with open(scratchFile, "w") as output:
        output.write(final_template)
    
    return job.fileStore.writeGlobalFile(scratchFile)



def generate_replica_mdin(
    job, mdin_input: FileID, temperatures: list,  runtype="remd"
) -> list[FileID]:
    """Writes a series of equilibration/relaxtions and production/remd AMBER mdin files.

    Parameters:
    ----------
    job: toil.job
    mdin_input: FileID
        mdin template for REMD simulation 
    temperatures: list[int]
        a list of temperatures for each replica mdin   
    Returns:
        list[FileID]: _description_
    """
    tempdir = job.fileStore.getLocalTempDir()

    # read in template mdin
    read_replica_mdin = job.fileStore.readGlobalFile(
        mdin_input, userPath=os.path.join(tempdir, os.path.basename(mdin_input))
    )

    
    replica_mdin_IDs = []
    generated_seeds = []

    # read replica template in temporary directory
    with open(read_replica_mdin) as temp:
        template = Template(temp.read())

    for index, temperature in enumerate(temperatures, start=1):

        # get unique random seed
        ig = generate_random_seeds(generated_seeds)
        # append to exisiting random seeds
        generated_seeds.append(ig)

        # update the temperature and ig seed
        replica_mdin = template.substitute(
            temp=temperature,
            ig=ig,
            restraint="$restraint",
        )
        # write a replica mdin
        mdin_filename = f"mdin.{runtype}.{index:03}"
        with open(mdin_filename, "w") as mdin:
            mdin.write(replica_mdin)

        replica_mdin_IDs.append(
            job.fileStore.writeGlobalFile(os.path.abspath(mdin_filename))
        )
    job.fileStore.logToMaster(f"replica mdins {replica_mdin_IDs}")
    
    return replica_mdin_IDs

    
   


def generate_random_seeds(seeds: list):
    """Random seed generator 
    Generates unique random integer to used used in replica exchange MDIN. 
    
    Parameters:
    -----------
    list_seeds: list[int]
        A list of unique generated intger values
    Returns:
        new_seed: int 
        A unique random generated integer value.  
    """
    new_seed = random.randrange(0, 32767)

    while new_seed in seeds:
        new_seed = random.randrange(0, 32767)

    return new_seed


if __name__ == "__main__":
    replica_mdin = "/nas0/ayoub/Impicit-Solvent-DDM/new_replicas/mdin.temp"
    mdins = generate_replica_mdin(
        mdin_input=replica_mdin, temperatures=[269.5, 300.0, 334.0], runtype="equil"
    )
    print(mdins)
