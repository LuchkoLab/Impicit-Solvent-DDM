# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

**Note**: This version represents the code used for the paper submission and publication. This release ensures reproducibility of the published results.

### Added
- Comprehensive documentation with Sphinx
- API documentation for all modules
- Example configurations and workflows
- Support for multiple AMBER engines (sander, sander.MPI, pmemd, pmemd.MPI, pmemd.CUDA)
- Automated restraint generation for Boresch restraints
- Temperature replica exchange molecular dynamics (TREMD) support
- Intermediate state calculations with configurable lambda windows
- Post-processing and MBAR analysis capabilities
- SLURM batch job submission support
- YAML configuration system for workflow parameters

### Features
- **Core Workflow**: Automated absolute binding free energy calculations using implicit solvent models
- **DDM Implementation**: Double decoupling method with Boresch restraints and GB solvent
- **Multi-Engine Support**: Compatible with various AMBER MD engines
- **Parallel Processing**: Support for multi-core and multi-node calculations
- **Flexible Configuration**: YAML-based configuration system
- **Analysis Tools**: Built-in trajectory analysis and MBAR post-processing

### Technical Details
- Python command-line interface
- Toil workflow management integration
- Support for generalized Born surface area (GBSA) models
- Configurable restraint types (CoM-CoM, CoM-Heavy_Atom, Heavy_Atom-Heavy_Atom)
- Temperature and lambda parameter sweeps
- Automated file management and job scheduling

### Documentation
- Complete API documentation
- Installation and setup instructions
- Configuration file examples
- Workflow tutorials and examples
- SLURM batch job templates

## [Unreleased]

### Planned
- GPU acceleration support
- Enhanced error handling and recovery
- Additional analysis methods
- Improved user interface
- Extended documentation

---

## Release Notes for v1.0.0

This release represents the first stable version of the Implicit Solvent DDM package suitable for publication and production use. The package provides a complete workflow for performing absolute binding free energy calculations using molecular dynamics with implicit solvent models.

### Key Capabilities
1. **Automated Workflow**: Complete automation of the thermodynamic cycle for binding free energy calculations
2. **Implicit Solvent Models**: Efficient calculations using generalized Born surface area methods
3. **Restraint Implementation**: Automated generation and application of Boresch restraints
4. **Parallel Computing**: Support for high-performance computing environments
5. **Analysis Pipeline**: Integrated post-processing and statistical analysis

### System Requirements
- Python 3.5+
- AMBER molecular dynamics package
- SLURM or PBS job scheduler (optional)
- Toil workflow management system

### Installation
```bash
git clone https://github.com/LuchkoLab/Impicit-Solvent-DDM.git
cd Impicit-Solvent-DDM
conda env create -f devtools/conda-envs/test_env.yaml
conda activate mol_ddm_env
python setup.py sdist
pip install dist/*
```

### Usage
```bash
run_implicit_ddm.py file:/path/to/job-store --config_file config.yaml --workDir /path/to/working/directory
```

This version is ready for publication and provides a robust foundation for binding free energy calculations in molecular dynamics simulations.

## [Unreleased]

### Planned for v1.1.0
- GPU acceleration support
- Enhanced error handling and recovery
- Additional analysis methods
- Improved user interface
- Extended documentation

### Planned for v2.0.0
- Major refactoring from gpu_enable_merge branch
- Significant performance improvements
- Enhanced workflow management
- Advanced GPU support
