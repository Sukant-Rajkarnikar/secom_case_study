"""SECOM data-mining workflow (organized into week1 through week6 folders)."""

import sys

from . import week1, week2, week3, week4, week5, week6
from .week1 import cli, data, eda
from .week2 import health, preparation, report
from .week3 import impute

# Keep top-level module paths (e.g. `from secom_analysis.data import ...`) working seamlessly
# alongside `from secom_analysis.week1.data import ...`
sys.modules[f"{__name__}.data"] = data
sys.modules[f"{__name__}.eda"] = eda
sys.modules[f"{__name__}.cli"] = cli
sys.modules[f"{__name__}.health"] = health
sys.modules[f"{__name__}.report"] = report
sys.modules[f"{__name__}.preparation"] = preparation
sys.modules[f"{__name__}.impute"] = impute

__version__ = "0.1.0"
__all__ = [
    "week1",
    "week2",
    "week3",
    "week4",
    "week5",
    "week6",
    "data",
    "eda",
    "cli",
    "health",
    "report",
    "preparation",
    "impute",
]
