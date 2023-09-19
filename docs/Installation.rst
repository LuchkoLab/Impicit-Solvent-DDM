Installation
===============
This page details how to get started with Implicit_Solvent_DDM. 
To install implicit_solvent_ddm, you will need to create an conda enviroment 

1. git pull git@github.com:LuchkoLab/Impicit-Solvent-DDM.git
2. conda env create -f devtools/conda-envs/test_env.yaml
3. conda activate mol_ddm_env
4. python setup.py sdist
5. pip install dist/*

Usagae
------
One installed, you can use the package. Quick example out to run the command executable.

run_implicit_ddm.py file:jobstore --config_file script_examples/config_files/basic_md_config.yaml --workDir path/to/working/directory 

