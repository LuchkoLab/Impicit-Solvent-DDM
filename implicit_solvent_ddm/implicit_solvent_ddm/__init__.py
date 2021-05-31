"""
Implicit_Solvent_DDM
Development of python command-line interface to simplify an absolute binding free energy cycle 
"""

# Add imports here
from .functions import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
