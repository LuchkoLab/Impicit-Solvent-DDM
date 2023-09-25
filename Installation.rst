Installation
===============
This page details how to get started with Implicit_Solvent_DDM. 
To install implicit_solvent_ddm, you will need to create an conda enviroment 

1. git clone git@github.com:LuchkoLab/Impicit-Solvent-DDM.git
2. cd Impicit-Solvent-DDM/
2. conda env create -f devtools/conda-envs/test_env.yaml (note:Try an interactive session for faster creation of env.)
3. conda activate isddm_env
4. python setup.py sdist
5. pip install dist/*

Usagae
------
One installed, you can use the package. Quick example out to run the command executable.

run_implicit_ddm.py file:jobstore --config_file script_examples/config_script1.yaml --workDir path/to/working/directory 

