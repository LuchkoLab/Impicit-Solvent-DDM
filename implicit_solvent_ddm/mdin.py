"""
Simple functions to create all the required MD input files (i.e. 'mdin').
"""
import itertools
import os
import random
from dataclasses import dataclass
from string import Template
from tkinter import Y
import re
import yaml
from toil.common import FileID


def generate_extdiel_mdin(job, user_mdin_ID: FileID, gb_extdiel: float) -> FileID:
    """Write an mdin with unique external dielectric for generalize Born solvent"""

    mdin_global = job.fileStore.readGlobalFile(user_mdin_ID)

    return job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "gb_extdiel_mdin", gb_extdiel=gb_extdiel)
    )


def get_mdins(job, user_mdin_ID: FileID):
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

    mdin_global = job.fileStore.readGlobalFile(user_mdin_ID)

    default_mdin = job.fileStore.writeGlobalFile(make_mdin_file(mdin_global, "_mdin"))
    no_solvent_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "no_solv_mdin", turn_off_solvent=True)
    )
    post_mdin = job.fileStore.writeGlobalFile(
        make_mdin_file(mdin_global, "post_mdin", post_process=True)
    )
    post_nosolv = job.fileStore.writeGlobalFile(
        make_mdin_file(
            mdin_global, "post_nosolv_mdin", turn_off_solvent=True, post_process=True
        )
    )

    return (default_mdin, no_solvent_mdin, post_mdin, post_nosolv)


def make_mdin_file(
    user_mdin_file,
    mdin_name,
    gb_extdiel=78.5,
    turn_off_solvent=False,
    post_process=False,
):
    """Rewrite users AMBER mdin file for specific thermodynamic states

    Parameters
    ----------
    user_mdin_file: FileID
        User provided mdin for thermodyamic states
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

    # general setting
    imin = "imin = 0"
    ioutfm = "ioutfm = 0"
    ntx = "ntx=1"
    irest = "irest=0"
    extdiel = f"extdiel={gb_extdiel}"
    # arguments for post-analysis
    if post_process:
        imin = "imin = 5"
        ioutfm = "ioutfm = 1"

    with open(user_mdin_file, "r") as output:
        data = output.readlines()

    new_mdin = ""
    for line in data:
        line = re.sub(r"extdiel\s*=\s*\$extdiel", extdiel, line)
        # Vacuum state
        if turn_off_solvent:
            line = re.sub(r"saltcon\s*=\s*\d+\.?\d+", "saltcon=0.0", line)
            line = re.sub(r"igb\s*=\s*\d+", "igb=6", line)
            line = re.sub(r"extdiel\s*=\s*\$extdiel", "extdiel=0.0", line)

        if "imin" in line:
            line = re.sub(r"imin\s*=\s*\d+", imin, line)
        if "ioutfm" in line:
            line = re.sub(r"ioutfm\s*=\s*\d+", ioutfm, line)
        if "irest" in line:
            line = re.sub(r"irest\s*=\s*\d+", irest, line)
        if "ntx" in line:
            line = re.sub(r"ntx\s*=\s*\d+", ntx, line)

        new_mdin += line

    with open(mdin_name, "w") as output:
        output.write(new_mdin)
    return os.path.abspath(mdin_name)


def generate_replica_mdin(
    job, mdin_input: FileID, temperatures: list, runtype="remd"
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
