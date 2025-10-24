# Release Notes - Implicit Solvent DDM v1.1.3

## Critical Bug Fixes

### COM Restraint Array Limit Fix
This release addresses critical Fortran crashes in flat-bottom restraints caused by excessive atom selection in the `igr1` list:

- **Fixed Fortran Array Overflow**: Limited receptor atom selection to CA atoms only (`@CA`) for protein-like systems to prevent Fortran crashes
- **Optimized Ligand Selection**: Restricted ligand atom selection to heavy atoms only (`!@H*`) to reduce array size
- **Fixed COM Restraint Template**: Updated restraint template format to prevent parsing errors
- **Prevented Memory Issues**: Significantly reduced the number of atoms in restraint calculations (10-100x reduction)

#### Technical Details
The fix addresses the core issue where:
- **Before**: All receptor atoms were selected (`self.receptor_mask`), potentially including thousands of atoms
- **After**: Only CA atoms are selected for protein-like systems (`self.receptor_mask & @CA`), typically 10-100x fewer atoms
- **Before**: All ligand atoms were selected (`self.ligand_mask`), including hydrogens
- **After**: Only heavy atoms are selected (`self.ligand_mask & !@H*`), reducing array size

#### System Detection Logic
Added intelligent system detection:
- **Protein-like systems**: Contains Cα atoms → use receptor Cα atoms only
- **Host-guest systems**: No Cα atoms → use all heavy atoms for receptor

#### Template Format Fix
Updated the COM restraint template to use proper AMBER format:
```diff
- iat = -1,-2
+ iat=-1,-1,
- &end
+ /
```

### GPU Configuration Fixes
- **Fixed GPU Limit**: Limited `num_accelerators` to 1 to prevent multi-GPU conflicts
- **GPU Stability**: Ensured only one GPU per simulation to avoid resource conflicts

### Post-Processing Improvements
- **Fixed File Cleanup**: Prevented premature deletion of mdout files during post-processing
- **Enhanced Export**: Improved file export handling for simulation outputs

## Improvements

### Restraint Generation
- **Enhanced Atom Selection**: More precise atom selection for restraint calculations
- **Reduced Computational Overhead**: Fewer atoms in restraint calculations improve performance
- **Better Memory Management**: Prevents memory allocation issues in large systems
- **Improved Stability**: Eliminates Fortran array limit crashes in complex systems
- **Intelligent System Detection**: Automatically detects protein vs host-guest systems

### Code Quality
- **Cleaner Template Format**: Standardized restraint template formatting
- **Better Error Handling**: Improved robustness in restraint generation
- **Optimized Selection Logic**: More efficient atom selection algorithms
- **Enhanced Logging**: Added detailed logging for restraint atom selection

### Conformational Restraints
- **Flat-Bottom Enhancement**: Added configurable flat-bottom width (`delta_flat = 0.5`)
- **Distance Range**: Restraints now use distance ranges instead of single values
- **Better Template**: Updated conformational restraint template format

## Performance Improvements

### Memory Usage
- **Reduced Memory Footprint**: Significantly lower memory usage for restraint calculations
- **Faster Restraint Generation**: Optimized atom selection improves generation speed
- **Better Scalability**: Handles larger protein systems without crashes

### System Compatibility
- **Large Protein Support**: Now handles systems with thousands of atoms without Fortran crashes
- **Complex Ligand Support**: Better handling of large ligand molecules
- **Improved Reliability**: More stable restraint generation across different system sizes
- **Host-Guest Systems**: Proper handling of small molecule systems

## Configuration Changes

### Breaking Changes
- **Atom Selection**: Receptor restraints now use CA atoms only for protein-like systems (previously all atoms)
- **Template Format**: Updated COM restraint template format for better AMBER compatibility
- **GPU Limitation**: `num_accelerators` is now limited to 1 to prevent conflicts

### Migration Guide
For existing configurations:
1. No configuration changes required - the fix is automatic
2. Restraint generation will be more efficient and stable
3. Large systems that previously crashed will now work correctly
4. GPU configurations will be automatically limited to 1 accelerator

## Testing & Validation

### Test Updates
- **Re-enabled Workflow Tests**: Fixed and re-enabled comprehensive workflow tests
- **Large System Testing**: Added validation for systems that previously caused Fortran crashes
- **Restraint Generation Testing**: Verified proper atom selection and template generation
- **System Type Detection**: Validated protein vs host-guest system detection

## Full Changelog

### Commits in this Release
- `2d4b5d8` - Fixed restrained atom selection for flat-bottom restraints. If the system is protein-like (contains Cα atoms), select receptor Cα atoms. Otherwise, assume a small host–guest system and select all heavy atoms for the receptor.
- `6833f54` - Fixed restrained atom selection for flat-bottom restraints. If the system is protein-like (contains Cα atoms), select receptor Cα atoms. Otherwise, assume a small host–guest system and select all heavy atoms for the receptor.
- `68cc562` - Section on bug fixes
- `2e3b8bc` - Release v1.1.3: Fix COM restraint array limit - Fix Fortran crashes by limiting receptor atoms to CA only (@CA) - Limit ligand atoms to heavy atoms only () - Update COM restraint template format - Reduce memory usage 10-100x in restraint calculations - Add comprehensive release notes
- `d0702b9` - Merge branch 'main' into fix/com-restraint-array-limit
- `943ffbe` - Merge branch 'patch/fix-gpu-distribution' into fix/com-restraint-array-limit
- `ade1c28` - Fixed merge issues within config.py, accepted the main branch changes
- `1c915a5` - feat: enhance restraint generation and GPU job scheduling
- `f34b778` - Fix Fortran crash during post-processing energy analysis

### Key Files Modified
- `implicit_solvent_ddm/restraints.py` - Fixed atom selection for COM restraints (CA atoms only for receptor, heavy atoms only for ligand)
- `implicit_solvent_ddm/templates/restraints/COM.restraint` - Updated template format
- `implicit_solvent_ddm/templates/restraints/conformational_restraints.template` - Enhanced flat-bottom restraints
- `implicit_solvent_ddm/config.py` - Fixed GPU accelerator limit
- `implicit_solvent_ddm/postTreatment.py` - Fixed file cleanup issues
- `implicit_solvent_ddm/simulations.py` - Improved file export handling
