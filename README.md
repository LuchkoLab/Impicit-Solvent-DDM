Implicit_Solvent_DDM
==============================
Automated, efficient absolute binding free energy workflow. 
Implicit solvent ddm project is a python tool for performing fully automated binding free energy calculations (ABFEs) using Molecular Dynamics. Here, we implemented an approach that utilizes faster methods, such as generalize Born surface area, which reduces computational costs with the use of implicit solvent models.  We adapted double decoupling method (DDM) that uses Boresch restraints[1] and GB solvent to enhance convergence. Due to the complexity of performing ABFEs, this python workflow package  performs all calculations of the thermodynamic cycle, post-processing, and data analysis.  


![Thermocycle_3_16_2021](https://user-images.githubusercontent.com/75343244/183561767-35dc6bb4-b329-418a-a86b-b9f4717eff1d.jpg)


[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/implicit_solvent_ddm/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/implicit_solvent_ddm/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Implicit_Solvent_DDM/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Implicit_Solvent_DDM/branch/master)

## Setup/Install
  1. `conda create name mol_ddm_env python=3.8`
  1. `git pull git@github.com:LuchkoLab/Impicit-Solvent-DDM.git`
  2.  `python setup.py sdist`
  3.  `pip install dist/*`
 
## YAML config file format
   The yaml config file is required to be able to run the program. The YAML config file specifies input parameters needed from the user, such as endstate `.parm7, .ncsrst` files of the complex system. As well system parameters, which program supports AMBER engines such as Sander, PMEMD & .MPI
```yaml

system_parameters:
  working_directory: '/nas0/ayoub/Impicit-Solvent-DDM/'
  executable: "sander.MPI" # executable machine for MD choice : [sander, sander.MPI, pmemed, pemed.MPI, pmeded.CUDA]
  mpi_command: "srun" # system dependent /


endstate_parameter_files:
  complex_parameter_filename: structs/complex/cb7-mol01.parm7 # list of topology file of a complex
  complex_coordinate_filename: structs/complex/cb7-mol01.rst7 # list of coordinate file of a complex  

number_of_cores_per_system:
    complex_ncores: 2 #total number of cores per job
    ligand_ncores: 1 #total number of cores per ligand simulation
    receptor_ncores: 1 #total number of cores per ligand simulation

AMBER_masks:
    receptor_mask: ':CB7' # list of Amber masks denoting receptor atoms in respected complex file
    ligand_mask: ':M01' # list of Amber masks denoting ligand atoms in respected complex file

workflow:
  endstate_method: remd #options REMD or 0 (meaning no endstate simulation will be performed just intermidates)endstate_method: REMD #options REMD, MD or 0 (meaning no endstate simulation will be performed just intermidates) 
  endstate_arguments:
    nthreads_complex: 16
    nthreads_receptor: 8
    nthreads_ligand: 8
    ngroups: 8 
    target_temperature: 300 # list of temperatures or temp.dat file
    remd_template_mdin: remd.template
    equilibrate_mdin_template: equil.template
    temperatures: [300.00, 327.32, 356.62, 388.05, 421.77, 457.91, 496.70, 500.00]

  intermidate_states_arguments:
    mdin_intermidate_config: intermidate_args.yaml #intermidate mdins required states 3-8
    igb_solvent: 2 #igb [1,2,3,7,8]
    temperature: 300
    exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4] # list exponent values 2**p 
    exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8] # list exponent values 2**p 
    restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]
    charges_lambda_window:  [0.5, 1.0]
    gb_extdiel_windows: [0.1, 0.2, 0.5]

```
## intermidate mdin arguments (mdin_intermidate_config)^^
  For intermidate MD parameters use an .yml file 
```yaml
#mdin required input parameters for intermidates 
nstlim: 1000
dt: 0.002
igb: 2
saltcon: 0.3
rgbmax: 999.0
gbsa: 0
temp0: 298
ntpr: 10
ntwx: 10
cut: 999
ntc: 2  
```
  
## Equilibration example Template  
The number of input files(len(temperatures)) will be generated automatically for the user. Only requirment is an template for equil and remd-production. 
User must  placed a $ig and $restraint key within both template as shown below. 
The number of temperature specify within the config file will generate the number of corresponding input files. 
```text 
 Equilibration (example)
 &cntrl
   irest=0, ntx=1, 
   nstlim=25000, dt=0.004,
   ntt=3, gamma_ln=1.0,
   temp0=$temp, ig=$ig,
   ntc=2, ntf=2, nscm=1000,
   ntb=0, igb=2,
   cut=999.0, rgbmax=999.0,
   ntpr=1, ntwx=1,
   nmropt=1, ioutfm=1,
   saltcon=0.025, gbsa = 0
 /
 &wt TYPE='END'
 /
DISANG=$restraint

```
```text 
TREMD (example)
 &cntrl
   irest=1, ntx=5, 
   nstlim = 1, dt=0.004, 
   numexchg = 2500000,
   ntt=3, gamma_ln=1.0,
   temp0=$temp, ig=$ig,
   ntc=2, ntf=2, nscm=0,
   ntb=0, igb=2,
   cut=999.0, rgbmax=999.0,
   ntpr=250, ntwx=250,
   ioutfm=1, nmropt=1,
   saltcon=0.15, gbsa = 0
 /
 &wt TYPE='END'
 /
DISANG=$restraint

```
## Quick Start (running locally) 

   `run_implicit_ddm.py file:jobstore --config_file config.yml --workDir working_directory`

## SLURM batch file (preferred) submission 
```bash
#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --time=09:30:00
#SBATCH --job-name=implicit_ddm
#SBATCH --export=all

pwd

run_implicit_ddm.py file:jobstore_test --config_file new_workflow.yaml 
```
## Ignore receptor flag
 If running mutiple HOST/Guests system with the same Host system use the `--ignore_receptor` flag to prevent any repeated receptor only simulations 


## Design Objectives

- [x] run MD all of the states for our restrained DDM cycle
- [x] should submit to slurm/PBS queue or run on local resources
- [x] run one state or multiple states with a single call
- [ ] run with one molecule or multiple molecules with a single call
    - [x] config file to define 
        * the cycle,
        * all states,
        * MD parameters for each state
    - [ ] can use template files for inputs
- [x] evaluate each trajectory in all other states
    * should submit to slurm/PBS queue or run on local resources
    * run one state or multiple states with a single call
    * run with one molecule or multiple molecules with a single call
    * automatically uses the correct combination of trajector, parameter and MD settings
    * uses the same config file(s) as MD
- [ ] Run MBAR on final evaluated trajectories.
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
