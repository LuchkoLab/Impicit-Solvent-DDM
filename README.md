Implicit_Solvent_DDM
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/implicit_solvent_ddm/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/implicit_solvent_ddm/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Implicit_Solvent_DDM/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Implicit_Solvent_DDM/branch/master)


Development of python command-line interface to simplify an absolute binding free energy cycle 
# Implicit-Solvent-DDM

## Design Objectives

* run MD all of the states for our restrained DDM cycle
    * should submit to slurm/PBS queue or run on local resources
    * run one state or multiple states with a single call
    * run with one molecule or multiple molecules with a single call
    * config file to define 
        * the cycle,
        * all states,
        * MD parameters for each state
    * can use template files for inputs
* evaluate each trajectory in all other states
    * should submit to slurm/PBS queue or run on local resources
    * run one state or multiple states with a single call
    * run with one molecule or multiple molecules with a single call
    * automatically uses the correct combination of trajector, parameter and MD settings
    * uses the same config file(s) as MD
* Run MBAR on final evaluated trajectories.
    * can just run locally
    * run one state or multiple states with a single call
    * run with one molecule or multiple molecules with a single call
    * won't fail on the whole calculation if one part fails
    * uses the same config file(s) as MD

## Exectuables

1. Runs MD
2. Re-evaluate trajectories
3. Parse data into dataframes
4. Run MBAR
5. create restraints

## Config file

* one of JSON, YAML or config.ini

### Elements

* top level should define states in thermodynamic cycle as an ordered list of names
* each name then gets a section which includes
    * templates to be used
    * values for template variables
    * any other settings 

### Copyright

Copyright (c) 2021, LuckoLab


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
