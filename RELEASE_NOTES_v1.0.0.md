# Release Notes - Implicit Solvent DDM v1.0.0

**Release Date**: December 19, 2024  
**Version**: 1.0.0  
**Status**: First Stable Release for Publication  
**Paper**: This version represents the code used for the paper submission and publication

## Overview

This release represents the first stable version of the Implicit Solvent DDM package, designed for automated absolute binding free energy calculations using molecular dynamics with implicit solvent models. **This version contains the exact code used for the paper submission and publication**, ensuring reproducibility of the published results.

## What's New in v1.0.0

### Core Features
- **Automated Binding Free Energy Workflow**: Complete implementation of the double decoupling method (DDM) with Boresch restraints
- **Implicit Solvent Support**: Integration with generalized Born surface area (GBSA) models for efficient calculations
- **Multi-Engine Compatibility**: Support for various AMBER molecular dynamics engines
- **Parallel Computing**: Full support for high-performance computing environments with SLURM/PBS

### Key Capabilities

#### 1. Thermodynamic Cycle Automation
- Automated setup and execution of all states in the binding free energy cycle
- Support for complex, ligand, and receptor-only simulations
- Configurable intermediate states with lambda parameter sweeps

#### 2. Restraint Management
- Automated generation of Boresch restraints
- Multiple restraint types: CoM-CoM, CoM-Heavy_Atom, Heavy_Atom-Heavy_Atom
- Configurable restraint parameters and force constants

#### 3. Temperature Replica Exchange
- Full TREMD support for enhanced sampling
- Configurable temperature ranges and exchange protocols
- Automated equilibration and production phases

#### 4. Analysis and Post-Processing
- Integrated MBAR analysis for free energy calculations
- Automated trajectory evaluation and data processing
- Statistical analysis and error estimation

### Technical Specifications

#### Supported AMBER Engines
- `sander` - Standard AMBER sander
- `sander.MPI` - MPI parallel sander
- `pmemd` - PMEMD engine
- `pmemd.MPI` - MPI parallel PMEMD
- `pmemd.CUDA` - GPU-accelerated PMEMD

#### Configuration System
- YAML-based configuration files
- Flexible parameter specification
- Support for multiple simulation systems

#### Workflow Management
- Toil-based workflow orchestration
- Fault tolerance and error recovery
- Scalable job submission and management

## Installation and Setup

### Prerequisites
- Python 3.5 or higher
- AMBER molecular dynamics package
- Conda package manager
- SLURM or PBS job scheduler (for HPC environments)

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/LuchkoLab/Impicit-Solvent-DDM.git
cd Impicit-Solvent-DDM

# Create conda environment
conda env create -f devtools/conda-envs/test_env.yaml
conda activate mol_ddm_env

# Build and install package
python setup.py sdist
pip install dist/*
```

### Quick Start
```bash
# Basic usage
run_implicit_ddm.py file:/path/to/job-store \
    --config_file config.yaml \
    --workDir /path/to/working/directory

# SLURM batch job example
#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --time=09:30:00
#SBATCH --job-name=implicit_ddm

run_implicit_ddm.py file:/scratch/my-job-store \
    --config_file config.yaml \
    --workDir /scratch/working_directory
```

## Configuration

The package uses YAML configuration files to specify all simulation parameters. Key configuration sections include:

- **System Parameters**: Working directory, executable, MPI settings
- **Endstate Parameters**: Complex, ligand, and receptor topology/coordinate files
- **Workflow Settings**: Simulation methods, temperature ranges, lambda windows
- **Computational Resources**: Core allocation, job scheduling parameters

## Documentation

Complete documentation is available including:
- API reference for all modules
- Configuration file examples
- Workflow tutorials
- Troubleshooting guides
- Best practices for HPC environments

## Performance and Scalability

- **Parallel Efficiency**: Optimized for multi-core and multi-node calculations
- **Memory Management**: Efficient handling of large trajectory files
- **Storage Optimization**: Compressed output and intermediate files
- **Fault Tolerance**: Automatic recovery from job failures

## Citation and Publication

**This version (v1.0.0) contains the exact code used for the paper submission and publication.** This ensures:
- **Reproducibility**: Exact code used in published results
- **Citation**: Clear version reference for academic citations
- **Stability**: Frozen codebase for paper reproducibility
- **Documentation**: Complete documentation for method validation
- **Examples**: All configurations and examples from the paper

## Support and Community

- **GitHub Repository**: https://github.com/LuchkoLab/Impicit-Solvent-DDM
- **Documentation**: Available in the `docs/` directory
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Contributing**: See CONTRIBUTING.md for development guidelines

## Future Development

The next major release (v1.1.0) will include:
- GPU acceleration support
- Enhanced error handling and recovery
- Additional analysis methods
- Improved user interface
- Extended documentation and tutorials

The major refactored version (v2.0.0) will include:
- Complete refactoring from gpu_enable_merge branch
- Significant performance improvements
- Enhanced workflow management
- Advanced GPU support

## Acknowledgments

This project is based on the Computational Molecular Science Python Cookiecutter template and builds upon established methods in molecular dynamics and free energy calculations.

---

**For questions or support, please contact the development team or open an issue on GitHub.**
