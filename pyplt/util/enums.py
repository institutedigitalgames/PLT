# Preference Learning Toolbox
# Copyright (C) 2018 Institute of Digital Games, University of Malta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""This module contains a number of classes defining different types of enumerated constants used throughout PLT."""

from enum import Enum


class ParamType(Enum):
    """Class specifying enumerated constants for parameter types.

    Extends `enum.Enum`.
    """

    INT = 0
    FLOAT = 1
    FLOAT_POSITIVE = 2


class PLAlgo(Enum):
    """Class specifying enumerated constants for preference learning algorithms.

    Extends `enum.Enum`.
    """

    RANKSVM = 1
    BACKPROPAGATION = 2
    BACKPROPAGATION_SKLEARN = 3
    NEUROEVOLUTION = 4
    RANKNET = 5


class EvaluatorType(Enum):
    """Class specifying enumerated constants for evaluators.

    Extends `enum.Enum`.
    """
    NONE = 0
    HOLDOUT = 1
    KFCV = 2


class FSMethod(Enum):
    """Class specifying enumerated constants for feature selection methods.

    Extends `enum.Enum`.
    """
    NONE = 0
    N_BEST = 1
    SFS = 2
    SBS = 3


class KernelType(Enum):
    """Class specifying enumerated constants for kernels used by RankSVM.

    Extends `enum.Enum`.
    """
    LINEAR = 0
    RBF = 1
    POLY = 2


class DataSetType(Enum):
    """Class specifying enumerated constants for types of ranks present in data sets.

    Extends the class :class:`enum.Enum`.
    """
    PREFERENCES = 1
    ORDERED = 2


class FileType(Enum):
    """Class specifying enumerated constants for data file types.

    Extends the class :class:`enum.Enum`.
    """
    OBJECTS = 1
    RANKS = 2
    SINGLE = 3


class NormalizationType(Enum):
    """Class specifying enumerated constants for data normalization methods.

    Extends the class :class:`enum.Enum`.
    """
    NONE = 0
    # BINARY = 1
    MIN_MAX = 1
    Z_SCORE = 2


class ActivationType(Enum):
    """Class specifying enumerated constants for types of activation functions used by Backpropagation.

    Extends `enum.Enum`.
    """
    LINEAR = 0
    SIGMOID = 1
    RELU = 2
