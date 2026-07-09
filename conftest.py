"""Make the repo-root `connect4` package importable for pytest without installing."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
