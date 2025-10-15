# Release Notes - Implicit Solvent DDM v1.1.1

## Feature Updates

### GPU Acceleration Support
This release introduces comprehensive GPU acceleration capabilities for molecular dynamics simulations:

- **CUDA-Enabled AMBER Executables**: Full support for `pmemd.cuda` and `pmemd.cuda.MPI` executables
- **Automatic GPU Detection**: The system automatically detects available GPUs when `CUDA: True` is set and `num_accelerators: num_gpus_per_simulation`
- **GPU Resource Management**: Each simulation can utilize dedicated GPU resources through proper CUDA device assignment
- **Hybrid CPU/GPU Workflows**: Complex and receptor simulations run on GPU while ligand simulations can run on CPU for optimal resource utilization

#### GPU Configuration
```yaml
system_parameters:
  executable: "pmemd.cuda"  # or "pmemd.cuda.MPI"
  mpi_command: srun
  CUDA: True  # Enable GPU acceleration
  num_accelerators: 2  # Number of GPUs to use per Simulation job. (The workflow will automatically detect total gpus avaiable)
```

### Workflow Modularization & Optimization
Complete restructuring of the DDM workflow for improved reliability and performance:

- **Phase-Based Architecture**: Workflow now operates in 7 distinct phases with clear dependencies
- **Parallel MBAR Computations**: MBAR analysis now runs in parallel across different systems instead of sequentially
- **Enhanced Progress Tracking**: Added detailed phase progression messages throughout the workflow

#### Workflow Phases
1. **Phase 1**: Setup and Preparation
2. **Phase 2**: Endstate Simulations  
3. **Phase 3**: System Decomposition and Restraint Generation
4. **Phase 4**: Intermediate State Simulations Setup
5. **Phase 5**: Intermediate State Simulations Execution
6. **Phase 6**: Energy Post-Processing and Analysis
7. **Phase 7**: Free Energy Computation and Consolidation

##  Improvements

### Performance Enhancements
- **Reduced CPU Allocation for GPU Jobs**: Complex and receptor simulations now use minimal CPU cores (0.1) when running on GPU
- **Optimized Memory Management**: Improved handling of large trajectory files and energy data
- **Faster MBAR Analysis**: Parallel execution of MBAR computations across different molecular systems

### Code Quality & Maintainability
- **Function Renaming**: `run_post_processing_and_analysis` → `compute_free_energy_and_consolidate` for clarity
- **Simplified Configuration**: Removed unnecessary `update_config_endstate` function
- **Cleaner Debug Logging**: Reduced verbose logging while maintaining essential progress information
- **Better Error Handling**: Improved error messages and workflow recovery

### Documentation & Examples
- **Updated Configuration Examples**: All example config files now include GPU configuration options
- **Enhanced Documentation**: Improved documentation structure and image display
- **Clearer Phase Messages**: Better progress tracking with descriptive phase completion messages

## 🐛 Bug Fixes

### Workflow Execution
- **Fixed Job Dependencies**: Resolved issues with Toil job dependency chains
- **Corrected Data Flow**: Fixed data promise handling between workflow phases
- **MBAR KeyError Resolution**: Fixed rounding mismatches in lambda_window indices
- **Command Line Arguments**: Fixed `logLevel` and `clean` option handling

### Adaptive Workflow Features
- **Removed Adaptive Lambda Windows**: Simplified workflow by removing adaptive features that caused complexity
- **Enforced Exponential Averaging**: Consistent flat-bottom restraint handling
- **Improved Restraint Management**: Better handling of restraint file generation and application

##  Dependencies & Environment

### Updated Dependencies
- **Toil**: Updated to version 5.12.0 (removed CWL dependency)
- **Conda Environment**: Added `pyyaml`, `numba`, and `setuptools<70` to support GPU features
- **Author Information**: Updated contact email from CSUN to UCI

### System Requirements
- **CUDA Support**: Requires CUDA-compatible AMBER installation for GPU features
- **Python Environment**: Compatible with Python 3.7+ environments
- **Memory Requirements**: Optimized for systems with 5GB+ memory per simulation

##  Breaking Changes

### Removed Features
- **Adaptive Lambda Windows**: This feature has been removed for workflow stability
- **CWL Support**: Removed Toil CWL dependency (use standard Toil workflows)
- **Legacy Configuration**: Some older configuration formats may need updates

### API Changes
- **Function Renames**: `run_post_processing_and_analysis` function has been renamed
- **Job Dependencies**: Internal job dependency structure has changed (should not affect user code)

##  Full Changelog

### Commits in this Release
- `e275f03` - Updated conda environment to include pyyaml, numba and setuptools<70
- `ccf70b1` - Resolved diverge issues with workflow_phases.py
- `469c514` - refactor: improve workflow execution and cleanup debug logging
- `70803b0` - docs: Improve documentation structure and fix image display issues
- `256b754` - refactor: rename post-processing function and improve docstrings
- `f2a0e0b` - Refactor workflow phases and improve job orchestration
- `e24342e` - Fix Toil job dependency chain using parent job coordination pattern
- `7ca1a46` - Fix Toil job dependency chain in decompose_system_and_generate_restraints
- `4004e4e` - Clean up workflow_phases.py: remove unused variables and simplify code
- `2afe9bd` - Fix workflow data flow: return proper data promises between phases
- `0e2edbb` - Merge refactor/workflow-modularization into gpu_enabled_merge
- `35d96ff` - refactor: modularize DDM workflow into separate phases
- `6b71f98` - Skip adaptive lambda windows and execute pymbar on completed windows
- `c228602` - Remove adaptive feature and enforce exponential averaging for flat-bottom restraints
- `f12e05f` - fix(mbar): resolve KeyError from rounding mismatches in lambda_window indices


---

**Version**: 1.1.1  
**Compatibility**: Python 3.7+, AMBER 18+, CUDA 10.0+
