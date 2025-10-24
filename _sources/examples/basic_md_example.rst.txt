Running Basic MD at the endstate simulation 
#####################################################


Running host-guest (CB7) absolute free energy calculation example:
==================================================================
All the example inputs scripts can be found in ``script_examples`` directrory. Lets first take a look at our example YAML config file ``config_script1.yaml``. 

.. code-block:: yaml

    hardware_parameters:
        working_directory: working_directory
        executable: "pmemd.MPI" # executable machine for MD choice : [sander, sander.MPI, pmemd, pmemd.MPI, pmeded.CUDA]
        mpi_command: srun # system dependent /
        CUDA: False # run on GPU
        output_directory_name: basic_md_dir
        cache_directory_output: path/to/cache_directory

    endstate_parameter_files:
        complex_parameter_filename: script_examples/structs/cb7-mol01_Hmass.parm7 # path to topology file; ["path/to/complex.parm7"]
        complex_coordinate_filename: script_examples/structs/cb7-mol01.rst7 # path to coordinate ["path/to/complex.ncrst"]list of coordinate file of a complex

    number_of_cores_per_system:
        complex_ncores: 2 #total number of cores per job
        ligand_ncores: 1 #total number of cores per ligand simulation
        receptor_ncores: 1 #total number of cores per ligand simulation

    AMBER_masks:
        receptor_mask: ":CB7" # list of Amber masks denoting receptor atoms in respected complex file
        ligand_mask: ":M01" # list of Amber masks denoting ligand atoms in respected complex file

    workflow:
        endstate_method: basic_md #options REMD or 0 (meaning no endstate simulation will be performed just intermidates)endstate_method: REMD #options REMD, MD or 0 (meaning no endstate simulation will be performed just intermidates) 
        endstate_arguments:
            md_template_mdin: /nas0/ayoub/Impicit-Solvent-DDM/new_replicas/basic_md.template
        intermediate_states_arguments:
            mdin_intermidate_config: script_examples/user_intermidate_mdin_args.yaml #intermidate mdins required states 3-8
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.584963, 3, 3.584963, 4]  # list exponent values 2**p 
            exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.584963, 7, 7.584963, 8]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]

The ``endstate_method`` is set to ``basic_md`` which will run a basic MD simulation at the endstates (states 1 and 8) :ref:`ddm_cycle-label`.
Every key within the YAML config file has been covered in previous section :ref:`my-reference-label`.


Basic MD mdin template (``md_template_mdin``) 
---------------------------------------------
Only requirment is to set ``DISANG=$restraint`` the restraint file will be loaded in during the workflow.

.. code-block:: bash 

 Relaxation 
 &cntrl
   irest=0, ntx=1, 
   nstlim=25000, dt=0.004,
   ntt=3, gamma_ln=1.0,
   temp0=300, ig=-1,
   ntc=2, ntf=2, nscm=1000,
   ntb=0, igb=2,
   cut=999.0, rgbmax=999.0,
   ntpr=1, ntwx=1,
   nmropt=1, ioutfm=1,
   saltcon=0.3, gbsa = 0
 /
 &wt TYPE='END'
 /
 DISANG=$restraint  

Running entire workflow
-----------------------
The recommend way for running this workflow is by submitting an batch file. This ensures the correct number of resrources and the submission of a single job for entire workflow. Example of batch file for SLURM schedular: 

.. code-block:: bash 

    #!/bin/bash
    #SBATCH --partition=main
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=12
    #SBATCH --time=09:30:00
    #SBATCH --job-name=ex_01
    #SBATCH --export=all

    pwd

    mkdir /scratch/username/

    run_implicit_ddm.py file:jobstore_example_01 --config_file script_examples/config_files/basic_md_config.yaml --workDir /scratch/username/

