"""ETC package exports."""

if __package__ in (None, ""):
	import sys
	from pathlib import Path

	package_root = Path(__file__).resolve().parent.parent
	if str(package_root) not in sys.path:
		sys.path.insert(0, str(package_root))

	import etc.utils as utils
	import etc.parse_ids as parse_ids
	from etc.hamiltonian import Hamiltonian, precompute, H
else:
	from . import utils
	from . import parse_ids
	from .hamiltonian import Hamiltonian, precompute, H

__all__ = ["utils", "parse_ids", "Hamiltonian", "precompute", "H"]
