# Release Notes - Implicit Solvent DDM v1.1.2

## Critical Bug Fixes

### GPU Distribution & Utilization
This release addresses critical GPU underutilization issues and improves GPU resource distribution across simulation jobs:

- **Fixed GPU Underutilization**: Removed `mpiexec` from `pmemd.cuda` runs to prevent GPU underutilization
- **Fixed Multi-Process GPU Bug**: Resolved critical issue where multiple processes were created on a single GPU, now limited to one process per GPU
- **GPU Isolation**: Each GPU now runs only one simulation job at a time with proper `CUDA_VISIBLE_DEVICES` assignment
- **Sequential GPU Execution**: Implemented proper GPU batching with sequential execution per GPU device
- **Enhanced GPU Distribution**: Each simulation job now runs on a separate GPU with proper device assignment
- **MPI Command Handling**: Fixed handling of `mpi_command=None` for CUDA simulations to prevent underutilization

#### GPU Distribution Improvements
```yaml
system_parameters:
  executable: "pmemd.cuda"
  mpi_command:   # No MPI for single GPU jobs, leave blank or remove from yaml config 
  CUDA: True
  num_accelerators: 1  # One GPU per simulation job
```

### System Configuration Fixes
- **Default GPU Count**: Set default `num_accelerators` to 1 for better GPU distribution instead of auto-detection
- **CUDA Device Assignment**: Proper `CUDA_VISIBLE_DEVICES` assignment for each GPU job
- **Sequential GPU Execution**: Jobs are now chained sequentially per GPU to maximize utilization

## Improvements

### Workflow Execution
- **GPU Job Batching**: GPU jobs are now properly distributed across available devices
- **GPU Environment Isolation**: Each GPU job gets its own `CUDA_VISIBLE_DEVICES` environment variable
- **Sequential Execution**: Jobs on the same GPU run sequentially to prevent resource conflicts
- **CPU/GPU Separation**: Clear separation between CPU and GPU job execution paths
- **Resource Optimization**: Better resource allocation for mixed CPU/GPU workflows

### Code Quality
- **Simplified System Names**: Standardized receptor and ligand system naming (`receptor_system`, `ligand_system`)
- **Cleaner GPU Logic**: Removed commented-out GPU batching code and implemented proper distribution
- **Better Error Handling**: Improved GPU detection and assignment logic

### GPU Environment Management
- **Environment Variable Assignment**: Each GPU job gets its own `sim.env["CUDA_VISIBLE_DEVICES"]` assignment
- **GPU ID Distribution**: Jobs are distributed across GPUs using `gpu_id = i % num_gpus` pattern
- **Sequential Job Chaining**: Jobs on the same GPU are chained using `addFollowOn()` for sequential execution
- **Resource Isolation**: Prevents multiple simulations from competing for the same GPU resources
- **One Process Per GPU**: Fixed bug where multiple processes were created on a single GPU, now strictly one process per GPU

## 📦 Dependencies & Environment

### Updated Dependencies
- **Toil**: Updated to version 8.2.0 (from 5.12.0)
- **Version Bump**: Updated package version to 1.1.2

### System Requirements
- **CUDA Support**: Requires CUDA-compatible AMBER installation
- **GPU Memory**: Optimized for single GPU per simulation job
- **Resource Management**: Better handling of multi-GPU systems

##  Performance Improvements

### GPU Utilization
- **Eliminated GPU Underutilization**: Fixed issues where GPUs were not being fully utilized
- **Fixed Multi-Process GPU Bug**: Resolved critical issue where multiple processes were created on a single GPU
- **Proper Resource Distribution**: Each simulation now gets dedicated GPU resources
- **GPU Environment Isolation**: Each GPU job runs with isolated `CUDA_VISIBLE_DEVICES` environment
- **Sequential GPU Execution**: Prevents GPU resource conflicts while maintaining efficiency
- **Better Resource Tracking**: Enhanced logging for GPU job distribution and execution

### Workflow Efficiency
- **Faster GPU Jobs**: Removed unnecessary MPI overhead for single GPU jobs
- **Better Resource Allocation**: Optimized CPU/GPU job separation
- **Improved Job Chaining**: Sequential execution per GPU prevents resource conflicts

## Configuration Changes

### Breaking Changes
- **Default GPU Count**: `num_accelerators` now defaults to 1 instead of auto-detection
- **MPI Command Handling**: `mpi_command=None` now properly handled for CUDA simulations
- **System Naming**: Standardized system naming conventions

### Migration Guide
For existing configurations:
1. Update `num_accelerators` to 1 for single GPU per job
2. Set `mpi_command: null` for CUDA simulations to avoid underutilization
3. Verify GPU device assignment in logs

## Testing & Validation

### Test Updates
- **Re-enabled Workflow Tests**: Fixed and re-enabled comprehensive workflow tests
- **GPU Distribution Testing**: Added validation for proper GPU job distribution
- **Resource Utilization Testing**: Verified GPU utilization improvements

## Full Changelog

### Commits in this Release
- `019dc50` - Fix GPU underutilization by removing mpiexec from pmemd.cuda runs
- `f14f659` - feat: implement GPU batching with sequential execution per GPU
- `a596854` - Updated to version 1.1.2, Bug fix release
- `7d4589f` - Updated Toil to version 8.2.0
- `25a99bb` - Fix GPU distribution bug in MD simulations

### Key Files Modified
- `implicit_solvent_ddm/config.py` - Fixed default GPU count and MPI command handling
- `implicit_solvent_ddm/runner.py` - Implemented proper GPU batching and sequential execution
- `implicit_solvent_ddm/simulations.py` - Fixed CUDA execution list handling
- `implicit_solvent_ddm/alchemical.py` - Standardized system naming
- `setup.py` - Updated version and Toil dependency

---

**Version**: 1.1.2  
**Release Type**: Bug Fix Release  
**Compatibility**: Python 3.7+, AMBER 18+, CUDA 10.0+  
**Critical**: This release fixes GPU underutilization issues that significantly impact simulation performance
