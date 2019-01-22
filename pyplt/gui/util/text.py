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

"""This module contains a number of helper functions for displaying text properly in the GUI."""

from string import capwords

from pyplt.util.enums import NormalizationType
from pyplt.util.enums import FSMethod, PLAlgo, KernelType
from pyplt.plalgorithms.backprop_tf import ActivationType


def real_type_name(type_name):
    """Receive a method type name and return the grammatically-correct version of its name.

    :param type_name: the method type name as used internally by PLT.
    :type type_name: str
    :return: the grammatically-correct version of type_name.
    :rtype: str
    """
    if type_name == NormalizationType.MIN_MAX.name:
        return "MinMax"
    elif type_name == NormalizationType.Z_SCORE.name:
        return "Z-Score"
    elif type_name == FSMethod.SFS.name:
        return type_name
    elif type_name == PLAlgo.RANKSVM.name:
        return "RankSVM"
    elif type_name == KernelType.RBF.name:
        return type_name
    elif type_name == KernelType.POLY.name:
        return "Polynomial"
    elif ActivationType.SIGMOID.name in type_name:
        return str(type_name).replace(ActivationType.SIGMOID.name, "Sigmoid")
    else:  # other / unknown
        return capwords(type_name)  # e.g. Linear, Backpropagation, Holdout


def real_param_name(param_name):
    """Receive a parameter name and return the grammatically-correct version of its name.

    :param param_name: the parameter name as used internally by PLT.
    :type param_name: str
    :return: the grammatically-correct version of param_name.
    :rtype: str
    """
    if param_name == "gamma":
        return "        \u03b3 "
    elif '_' in param_name:
        param_name = str(param_name).replace('_', ' ')
        # ^ replace underscore before capwords() not through sep so that even one-word names like 'epoch' get
        # capitalized by capwords
    try:
        return capwords(param_name)
    except:  # probably not a string...
        return param_name
