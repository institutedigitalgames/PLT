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

"""This package defines a number of utility classes for backend processes of PLT."""


class AbortFlag:
    """This utility class assists the termination of experiments before completion."""

    def __init__(self):
        """Initializes a stopping flag variable to False (boolean).

        The stopping variable indicates whether or not the experiment should be stopped.
        """
        self._stop = False

    def stop(self):
        """Set the stopping flag to True."""
        self._stop = True
        print("STOP!")

    def stopped(self):
        """Get the stopping flag which indicates whether or not the experiment should be stopped.

        :return: the stopping flag.
        :rtype: bool
        """
        return self._stop
