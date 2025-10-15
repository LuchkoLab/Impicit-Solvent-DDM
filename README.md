# Implicit Solvent DDM

**Automated, efficient absolute binding free energy calculations using implicit solvent models.**

Implicit Solvent DDM is a Python package for performing fully automated binding free energy calculations (ABFEs) using Molecular Dynamics. The package implements the double decoupling method (DDM) with Boresch restraints and generalized Born surface area (GBSA) models to reduce computational costs while maintaining accuracy.

**📄 Paper**: [Automated Workflow for Absolute Binding Free Energy Calculations with Implicit Solvent and Double Decoupling](https://arxiv.org/abs/2509.21808) - *arXiv:2509.21808*


##  Documentation

**[View Complete Documentation](https://luchkolab.github.io/Impicit-Solvent-DDM/)**

The documentation includes:
- **Installation Guide** - Setup instructions and requirements
- **Configuration Examples** - YAML configuration with GPU support
- **Workflow Structure** - Detailed DDM workflow phases
- **API Reference** - Complete module documentation
- **Updates & Changelog** - Latest features including GPU acceleration

##  Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/LuchkoLab/Impicit-Solvent-DDM.git
cd Impicit-Solvent-DDM

# 2. Create conda environment
conda env create -f devtools/conda-envs/test_env.yaml
conda activate isddm_env

# 3. Install package
python setup.py sdist
pip install dist/*

# 4. Run simulation
run_implicit_ddm.py file:/path/to/job-store \
    --config_file config.yaml \
    --workDir /path/to/working/directory
```

##  Key Features

- **GPU Acceleration** - CUDA support for faster simulations
- **Automated Workflow** - Complete DDM cycle automation
- **Implicit Solvent** - GBSA models for efficient calculations
- **AMBER Support** - Sander, PMEMD, and GPU-enabled AMBER executables
- **HPC Ready** - SLURM/PBS job scheduling support
- **MBAR Analysis** - Integrated free energy calculations

## 📖 What's New in v1.1.1

- **GPU Support** - CUDA acceleration for MD simulations
- **Enhanced Documentation** - Comprehensive guides and examples
- **Improved Workflow** - Better error handling and recovery

---

**For detailed information, examples, and API reference, visit our [complete documentation](https://luchkolab.github.io/Impicit-Solvent-DDM/).**

*Last updated: January 2025*

## Citation

If you use this software in your research, please cite:

```bibtex
@misc{ayoub2025automated,
  title={Automated Workflow for Absolute Binding Free Energy Calculations with Implicit Solvent and Double Decoupling},
  author={Steven Ayoub and Michael Barton and David A. Case and Tyler Luchko},
  year={2025},
  eprint={2509.21808},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2509.21808}
}
```
