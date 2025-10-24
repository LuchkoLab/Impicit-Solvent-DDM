.. _updates-label:

Updates and Changelog
======================

This section documents the major updates and changes made to the Implicit Solvent DDM package across different versions.

Version 1.1.1 - GPU Acceleration Support
=========================================

**Release Date**: October 2025  
**Status**: Latest Development Release

Major Features
--------------

GPU Acceleration Support
~~~~~~~~~~~~~~~~~~~~~~~~~

The package now includes comprehensive GPU acceleration support for molecular dynamics simulations using CUDA-enabled AMBER executables.

**Key Features:**
- **Automatic GPU Detection**: The system automatically detects available GPUs when ``CUDA: True`` is set
- **Flexible GPU Allocation**: Support for specifying the number of GPUs per simulation
- **CUDA-Aware Job Scheduling**: Intelligent job distribution across available GPU resources
- **Fallback Support**: Graceful fallback to CPU execution when GPUs are unavailable

**Supported GPU Executables:**
- ``pmemd.cuda`` - GPU-accelerated PMEMD engine
- Custom CUDA-enabled AMBER executables

Configuration Changes
~~~~~~~~~~~~~~~~~~~~~

New Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following new parameters have been added to the configuration system:

**System Settings:**
- ``CUDA`` (bool): Enable/disable GPU acceleration (default: ``False``)
- ``num_accelerators`` (int): Number of GPUs to request (default: ``0`` - auto-detect)

**Example Configuration:**

.. code-block:: yaml

    system_parameters:
      executable: "pmemd.cuda"  # GPU-enabled executable
      mpi_command: "srun"       # or "mpirun"/"mpiexec"
      CUDA: True                # Enable GPU acceleration
      num_accelerators: 2       # Number of GPUs (0 = auto-detect)
      output_directory_name: "gpu_simulation"

**Complete CUDA Configuration Example:**

.. code-block:: yaml

    # CUDA-enabled configuration example
    system_parameters:
      executable: "pmemd.cuda"
      mpi_command: "srun"
      CUDA: True
      num_accelerators: 2       # Use 2 GPUs
      memory: "10G"            # Increased memory for GPU jobs
      disk: "20G"              # Increased disk space
      output_directory_name: "cuda_ddm_run"

    endstate_parameter_files:
      complex_parameter_filename: "path/to/complex.parm7"
      complex_coordinate_filename: "path/to/complex.rst7"

    number_of_cores_per_system:
      complex_ncores: 8        # CPU cores for complex simulation
      ligand_ncores: 4         # CPU cores for ligand simulation  
      receptor_ncores: 4       # CPU cores for receptor simulation

    AMBER_masks:
      receptor_mask: ":RECEPTOR"
      ligand_mask: ":LIGAND"

    workflow:
      endstate_method: "basic_md"
      endstate_arguments:
        md_template_mdin: "path/to/template.mdin"
      intermediate_states_arguments:
        mdin_intermediate_file: "path/to/intermediate.mdin"
        igb_solvent: 2
        temperature: 300
        exponent_conformational_forces: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        exponent_orientational_forces: [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        restraint_type: 2

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

**SystemSettings Class Updates:**
- Added ``CUDA`` boolean field with automatic GPU detection
- Added ``num_accelerators`` field with intelligent defaults
- Enhanced ``__post_init__`` method for GPU environment validation

**Simulation Class Enhancements:**
- Updated command-line argument generation for CUDA executables
- Enhanced GPU-aware job scheduling logic
- Improved environment variable handling for CUDA_VISIBLE_DEVICES

**Runner Class Improvements:**
- Added GPU job batching and resource management
- Implemented intelligent GPU allocation across simulation batches
- Enhanced error handling for GPU resource conflicts

**Key Code Changes:**

In ``config.py``:

.. code-block:: python

    @dataclass
    class SystemSettings:
        CUDA: bool = field(default=False)
        num_accelerators: int = field(default=0)
        
        def __post_init__(self):
            if self.CUDA and self.num_accelerators == 0:
                try:
                    from numba import cuda
                    self.num_accelerators = len(cuda.gpus)
                except ImportError:
                    raise RuntimeError("CUDA requested but 'cuda' module not available.")

In ``simulations.py``:

.. code-block:: python

    def setup(self):
        if self.CUDA and self.system_type in ["complex", "receptor"]:
            self.exec_list.append(self.executable)
        # ... rest of setup logic

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

- **GPU Acceleration**: Significant speedup for large system simulations
- **Resource Optimization**: Better utilization of available computational resources
- **Memory Management**: Enhanced memory handling for GPU-accelerated simulations
- **Job Scheduling**: Improved parallel execution with GPU-aware scheduling

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

- All existing CPU-only configurations remain fully functional
- Default behavior unchanged (CPU execution)
- No breaking changes to existing API or configuration format
- Seamless upgrade path from previous versions

Migration Guide
~~~~~~~~~~~~~~~

**For Existing Users:**

To enable GPU acceleration, simply add the following to your configuration:

.. code-block:: yaml

    system_parameters:
      executable: "pmemd.cuda"  # Change from "pmemd.MPI" to "pmemd.cuda"
      CUDA: True                # Add this line
      num_accelerators: 1       # Optional: specify number of GPUs

**Hardware Requirements:**
- CUDA-enabled GPU with sufficient memory
- CUDA-compatible AMBER installation
- Appropriate CUDA drivers and runtime

**Software Dependencies:**
- AMBER with CUDA support
- CUDA toolkit (version 10.0 or higher recommended)
- Python packages: numba (for GPU detection)

GPU-Enabled PyMBAR Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU-accelerated free energy analysis using PyMBAR, you can enable JAX CUDA support to leverage GPU computing for MBAR calculations.

**Installation Requirements:**

Follow the JAX installation guide for NVIDIA GPU support with CUDA 12:

.. code-block:: bash

    # Install JAX with CUDA 12 support
    pip install -U "jax[cuda12]"

**Verification:**

Check your CUDA version:

.. code-block:: bash

    nvcc --version

**Configuration:**

Once JAX with CUDA support is installed, PyMBAR will automatically detect and use GPU acceleration when available. The analysis will be performed using JAX's GPU-accelerated operations, significantly speeding up MBAR calculations for large datasets.

**GPU Allocation:**

Note that only one GPU will be used per simulation job. This means each individual simulation (such as complex lambda windows, endstate simulations, charge lambda windows, etc.) will utilize a single GPU for acceleration. Multiple simulations can run in parallel across different GPUs when available.

**Benefits:**
- Accelerated MBAR free energy calculations
- Faster convergence for large simulation datasets
- Reduced analysis time for complex systems
- Automatic GPU detection and utilization

**References:**
- `JAX Installation Guide <https://docs.jax.dev/en/latest/installation.html>`_
- `JAX CUDA Support <https://docs.jax.dev/en/latest/installation.html#nvidia-gpu>`_

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- GPU memory requirements may be higher than CPU simulations
- Some small systems may not benefit significantly from GPU acceleration
- CUDA_VISIBLE_DEVICES environment variable management requires careful configuration in multi-GPU setups



Version 1.0.0 - Initial Stable Release
======================================

**Release Date**: December 19, 2024  
**Status**: First Stable Release for Publication

This version represents the first stable release of the Implicit Solvent DDM package, containing the exact code used for the paper submission and publication.

**Key Features:**
- Complete DDM workflow implementation
- Implicit solvent support (GBSA models)
- Multi-engine compatibility (AMBER executables)
- Parallel computing support (SLURM/PBS)
- Automated restraint generation
- Temperature replica exchange (TREMD)
- Integrated MBAR analysis

For detailed information about v1.0.0 features, see :ref:`Installation <installation-label>` and :ref:`Implementation Details <ddm_cycle-label>`.

Related Documentation
--------------------

- :ref:`Installation <installation-label>` - Setup instructions including GPU requirements
- :ref:`Configuration <config-label>` - Complete configuration reference
- :ref:`Examples <examples-label>` - Usage examples including GPU configurations
- :ref:`API Documentation <api-label>` - Technical API reference
