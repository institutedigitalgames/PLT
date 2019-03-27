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

"""The graphical user interface (GUI) component of PLT is managed by the :mod:`pyplt.gui` subpackage and run via the
:mod:`pyplt.main_gui` script. The remaining submodules and subpackages within this package focus on the backend
functionality and the application programming interface (API) component of PLT.
"""

import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("root path:")
# print(ROOT_PATH)

__version__ = "0.2.1"
