.. _user_defined_example-label:

User Provided Endstate Simulation 
#################################


Running host-guest (CB7) absolute free energy calculation example:
==================================================================
If the user wishes to run there own endstate simulation they just need to provide there own ``.parm7`` and ``.nc`` files. 
Please remove all solvent molecules from the ``.parm7`` and ``.nc`` files if used prior. As-well make sure the ``.parm7`` file has the recommened bondi radii for specified GB solvent model (check AMBER manual). 
All the example inputs scripts can be found in ``script_examples`` directrory. Lets first take a look at our example YAML config file ``config_script1.yaml``. 

.. code-block:: yaml

    hardware_parameters:
        working_directory: working_directory
        executable: "pmemd.MPI" # executable machine for MD choice : [sander, sander.MPI, pmemd, pmemd.MPI, pmeded.CUDA]
        mpi_command: srun # system dependent /
        CUDA: False # run on GPU
        output_directory_name: basic_md_dir
        cache_directory_output: path/to/cache_directory
        top_directory_name: basic_md_dir

    endstate_parameter_files:
        complex_parameter_filename: script_examples/structs/cb7-mol01_Hmass.parm7 # path to topology file; ["path/to/complex.parm7"]
        complex_coordinate_filename: script_examples/structs/cb7-mol01_user_provided_traj.nc # path to coordinate ["path/to/complex.ncrst"]list of coordinate file of a complex
        ligand_parameter_filename:  script_examples/structs/M01.parm7
        ligand_coordinate_filename: script_examples/structs/M01_user_provided_traj.nc
        receptor_parameter_filename: script_examples/structs/CB7.parm7
        receptor_coordinate_filename: script_examples/structs/CB7_user_provided_traj.nc

    number_of_cores_per_system:
        complex_ncores: 2 #total number of cores per job
        ligand_ncores: 1 #total number of cores per ligand simulation
        receptor_ncores: 1 #total number of cores per ligand simulation

    AMBER_masks:
        receptor_mask: ":CB7" # list of Amber masks denoting receptor atoms in respected complex file
        ligand_mask: ":M01" # list of Amber masks denoting ligand atoms in respected complex file

    workflow:
        endstate_method: 0 #options REMD or 0 (meaning no endstate simulation will be performed just intermidates)endstate_method: REMD #options REMD, MD or 0 (meaning no endstate simulation will be performed just intermidates) 
        intermediate_states_arguments:
            mdin_intermidate_config: script_examples/user_intermidate_mdin_args.yaml #intermidate mdins required states 3-8
            igb_solvent: 2 #igb [1,2,3,7,8]
            temperature: 300
            exponent_conformational_forces: [
                -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 
                2.584963, 3, 3.584963, 4
            ]  # list exponent values 2**p 
            exponent_orientational_forces: [
                -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 
                6.584963, 7, 7.584963, 8
            ]  # list exponent values 2**p 
            restraint_type: 2 # choices: [ 1: CoM-CoM, 2: CoM-Heavy_Atom, 3: Heavy_Atom-Heavy_Atom, must be 1, 2 or 3 ]

The ``endstate_method`` is set to ``0`` which assumes the user provided there own endstate simulation. 
Please note if ``endstate_method`` is set to ``0`` the user must provided the following keys values for ``ligand_parameter_filename``, ``ligand_coordinate_filename``, ``receptor_parameter_filename``, and ``receptor_coordinate_filename``.
Every key within the YAML config file has been covered in previous section :ref:`my-reference-label`.

Running entire workflow
-----------------------
The recommend way for running this workflow is by submitting an batch file. This ensures the correct number of resrources and the submission of a single job for entire workflow. Example of batch file for SLURM schedular: 

.. code-block:: bash 

    #!/bin/bash
    #SBATCH --partition=main
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=12  # Use high number for post-energy analysis (imin=5) parallelization
    #SBATCH --time=09:30:00
    #SBATCH --job-name=ex_01
    #SBATCH --export=all

    pwd

    mkdir /scratch/username/

    run_implicit_ddm.py file:jobstore_example_03 --config_file script_examples/config_files/user_defined_endstate.yaml --workDir /scratch/username/

Performance Considerations
---------------------------

**High CPU Count**: Use a high number of CPUs (12+ recommended) for optimal performance. The Sander post-energy analysis (imin=5) runs on CPUs and completion time scales linearly with the number of CPUs - more CPUs = faster completion.

