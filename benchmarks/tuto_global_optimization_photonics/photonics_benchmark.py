from problems.brag_mirror import brag_mirror
from problems.ellipsometry_inverse import ellipsometry
from problems.sophisticated_antireflection_design import (
    sophisticated_antireflection_design,
)
from problems.grating2D import grating2D

try:
    from problems.plasmonic_nanostructure import plasmonic_nanostructure
except ImportError as _pygdm2_error:  # pyGDM2 is optional, see below
    _PYGDM2_ERROR = _pygdm2_error

    def plasmonic_nanostructure(*args, **kwargs):
        """Placeholder: the plasmonic problem needs pyGDM2.

        pyGDM2 is not in the ``examples`` group because it requires
        ``numpy<=2.3``, which would hold back numpy for the whole project.
        Install it separately (``pip install pyGDM2``) to use this problem.
        """
        raise ImportError(
            "plasmonic_nanostructure needs pyGDM2: pip install pyGDM2"
        ) from _PYGDM2_ERROR
