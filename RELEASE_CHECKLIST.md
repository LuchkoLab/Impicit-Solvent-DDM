# Release Checklist for v1.0.0

## Pre-Release Checklist

### Code Quality
- [ ] All tests pass
- [ ] Code is properly documented
- [ ] No critical bugs or issues
- [ ] Version numbers are consistent across all files
- [ ] Dependencies are properly specified

### Documentation
- [x] CHANGELOG.md created and updated
- [x] RELEASE_NOTES_v1.0.0.md created
- [x] API documentation is complete
- [x] Installation instructions are clear
- [x] Configuration examples are provided

### Version Management
- [x] Version 1.0.0 specified in setup.py
- [x] Version information in __init__.py
- [x] Git tags prepared for release

### Files to Create/Update
- [x] CHANGELOG.md
- [x] RELEASE_NOTES_v1.1.1.md
- [x] RELEASE_CHECKLIST.md (this file)

## Release Process

### 1. Final Testing
```bash
# Run all tests
python -m pytest tests/

# Check installation
python setup.py sdist
pip install dist/*

# Verify package works
python -c "import implicit_solvent_ddm; print(implicit_solvent_ddm.__version__)"
```

### 2. Create Git Tag
```bash
# Create and push the release tag
git tag -a v1.0.0 -m "Release version 1.0.0 - First stable release for publication"
git push origin v1.0.0
```

### 3. Create GitHub Release
1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Choose tag: v1.0.0
4. Release title: "Implicit Solvent DDM v1.0.0 - First Stable Release"
5. Copy content from RELEASE_NOTES_v1.0.0.md
6. Attach source distribution files if needed

### 4. Post-Release Tasks
- [ ] Update main branch with any final changes
- [ ] Prepare for next development cycle
- [ ] Update development documentation
- [ ] Notify users of the release

## Files Created for Release

### Documentation Files
- `CHANGELOG.md` - Complete changelog following Keep a Changelog format
- `RELEASE_NOTES_v1.0.0.md` - Detailed release notes for users
- `RELEASE_CHECKLIST.md` - This checklist for release process

### Version Information
- Current version: 1.0.0 (in setup.py)
- Git versioning: Configured with versioneer
- Package name: implicit_solvent_ddm

## Release Notes Summary

**This release (v1.0.0) represents the exact code used for the paper submission and publication.** This ensures reproducibility of published results and provides a stable reference for academic citations.

This release includes:
- Complete automated workflow for binding free energy calculations
- Support for multiple AMBER engines
- Implicit solvent model integration
- Boresch restraint automation
- Temperature replica exchange support
- MBAR analysis capabilities
- SLURM/PBS job scheduling
- Comprehensive documentation

## Next Steps After Release

1. **Paper Reproducibility**: This v1.0.0 release ensures the exact code used in the paper is preserved and citable
2. **Merge gpu_enable_merge branch**: After creating this release, you can safely merge your refactor changes
3. **Update version**: Consider bumping to v2.0.0 for the refactored version
3. **Documentation updates**: Update docs for new features in the refactored version
4. **Testing**: Ensure all new features work correctly in the merged version

## Important Notes

- **Paper Reproducibility**: This v1.0.0 release contains the exact code used for the paper submission and publication
- **Academic Citation**: This version provides a stable, citable reference for the published work
- **Stable Release**: The current main branch represents a stable, tested version used in the paper
- **Future Development**: The gpu_enable_merge branch contains significant refactoring for future releases
- **Version Strategy**: After this release, merge refactor changes as v2.0.0 due to significant changes

## Contact and Support

For questions about this release:
- GitHub Issues: https://github.com/LuchkoLab/Impicit-Solvent-DDM/issues
- Documentation: Available in docs/ directory
- Email: steven.ayoub.362@my.csun.edu
