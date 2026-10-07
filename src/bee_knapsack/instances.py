"""Benchmark instances and data loaders.

The three small instances below are embedded directly so the project runs
without any data files. Their optima were verified by exhaustive search.

* P08  - from the KNAPSACK_01 data set (J. Burkardt), 24 items.
* f2, f8 - low-dimensional instances (f2_l-d_kp_20_878, f8_l-d_kp_*_10000).

The larger Pisinger instances (knapPI_<type>_<n>_1000_1) are read from
``data/``. Type 1 = uncorrelated, 2 = weakly correlated, 3 = strongly
correlated profits and weights. Their optima were verified by dynamic
programming.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Instance:
    name: str
    capacity: float
    weights: tuple
    values: tuple
    optimum: float = None   # known optimal profit, if available

    @property
    def num_items(self):
        return len(self.weights)


P08 = Instance(
    name="P08",
    capacity=6404180,
    weights=(382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150,
             823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
             323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684),
    values=(825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457,
            1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277,
            2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257,
            369261),
    optimum=13549094,
)

F2 = Instance(
    name="f2",
    capacity=878,
    weights=(92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96,
             70, 48, 14, 58),
    values=(44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17,
            75, 29, 75, 63),
    optimum=1024,
)

F8 = Instance(
    name="f8",
    capacity=10000,
    weights=(983, 982, 981, 980, 979, 978, 488, 976, 972, 486, 486, 972, 972,
             485, 485, 969, 966, 483, 964, 963),
    values=(981, 980, 979, 978, 977, 976, 487, 974, 970, 485, 485, 970, 970,
            484, 484, 976, 974, 482, 962, 961),
    optimum=9767,
)

BUILTIN = {inst.name: inst for inst in (P08, F2, F8)}


def load_instance(path, capacity, optimum=None, name=None,
                  weight_col="weights", value_col="values"):
    """Load an instance from a CSV (preferred) or pickled DataFrame.

    The file must contain one row per item with a weight and a value column.
    The capacity is not stored in the file and must be passed explicitly.
    """
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_pickle(path)
    return Instance(
        name=name or path.stem,
        capacity=capacity,
        weights=tuple(df[weight_col].astype(float)),
        values=tuple(df[value_col].astype(float)),
        optimum=optimum,
    )


# --------------------------------------------------------------------------
# Pisinger instances (files in data/)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# name: (file stem, capacity, optimum)
PISINGER = {
    "PI_1_100":   ("knapPI_1_100_1000_1",   1000,  9147),
    "PI_1_200":   ("knapPI_1_200_1000_1",   1000,  11238),
    "PI_1_500":   ("knapPI_1_500_1000_1",   2543,  28857),
    "PI_2_100":   ("knapPI_2_100_1000_1",   995,   1514),
    "PI_2_200":   ("knapPI_2_200_1000_1",   1008,  1634),
    "PI_2_500":   ("knapPI_2_500_1000_1",   2543,  4566),
    "PI_3_100":   ("knapPI_3_100_1000_1",   997,   2397),
    "PI_3_200":   ("knapPI_3_200_1000_1",   997,   2697),
    "PI_3_500":   ("knapPI_3_500_1000_1",   2517,  7117),
    "PI_1_10000": ("knapPI_1_10000_1000_1", 49877, 563647),
}


def read_pisinger(path):
    """Read a knapPI text file.

    Format: one line "value weight" per item, optionally followed by a line
    with the reference 0/1 solution (separated by a blank line or not).
    Returns (values, weights, reference_solution or None).
    """
    values, weights, reference = [], [], None
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) > 2:                     # the reference solution line
            reference = tuple(int(t) for t in tokens)
        else:
            values.append(float(tokens[0]))
            weights.append(float(tokens[1]))
    return tuple(values), tuple(weights), reference


def load_pisinger(name, data_dir=DATA_DIR):
    """Load one of the instances listed in PISINGER by name (e.g. "PI_2_500")."""
    stem, capacity, optimum = PISINGER[name]
    values, weights, _ = read_pisinger(Path(data_dir) / f"{stem}.txt")
    return Instance(name=name, capacity=capacity, weights=weights,
                    values=values, optimum=optimum)
