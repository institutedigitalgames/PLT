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


class Evaluator:
    """Base class for all evaluation (validation or testing) methods."""

    _eval_type = None
    _description = ""
    _debug = False
    _params = {}
    _name = ""

    _train_accuracy = None
    _eval_accuracy = None
    _result_model = None

    def __init__(self, description="A validation/testing method.", name="", debug=False, **kwargs):
        """Initializes the Evaluator object.

        :param description: a description of the evaluation method (default "A validation/testing method.").
        :type description: str, optional
        :param name: the name of the evaluation method (default "").
        :type name: str, optional
        :param debug: specifies whether or not to print notes to console for debugging purposes (default False).
        :type debug: bool, optional
        :param kwargs: any additional parameters for the evaluation method.
        """
        self._name = name
        self._description = description
        self._debug = debug
        self._params = {}
        for key in kwargs:
            # print(key)
            # print(kwargs[key])
            self._params[key] = kwargs[key]

    # Getters and setters

    def get_name(self):
        """Get the name of the evaluation method.

        :return: the name of the evaluation method.
        :rtype: str
        """
        return self._name

    def get_description(self):
        """Get the description of the evaluation method.

        :return: the description of the evaluation method.
        :rtype: str
        """
        return self._description

    def get_params(self):
        """Return all additional parameters of the evaluation method (if applicable).

        :return: a dict containing all additional parameters of the evaluation method with the parameter names
            as the dict's keys and the corresponding parameter values as the dict's values (if applicable).
        :rtype: dict
        """
        return self._params

    def get_params_string(self):
        """Return a string representation of all additional parameters of the evaluation method (if applicable).

        :return: the string representation of all additional parameters of the evaluation method (if applicable).
        :rtype: str
        """
        return "{" + self._get_param_string(self._params) + "}"

    def _get_param_string(self, params):
        """Internal recursive method for the construction of a string representation of additional method parameters.

        :param params: a parameter or list of parameters to be included in the string.
        :type params: dict
        :return: a string representation of the given parameters.
        :rtype: str
        """
        ret = ""
        for p in params:
            if len(ret) > 0:
                ret += "; "
            if isinstance(params[p], tuple):
                ret += " {" + self._get_param_string(params[p]) + "} "
            else:
                ret += str(p) + ": " + str(params[p])
        return ret
