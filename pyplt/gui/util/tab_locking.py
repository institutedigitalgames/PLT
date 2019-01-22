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

"""This module defines two classes which manage the locking and unlocking of tabs in `ttk.Notebook` widgets."""

import tkinter as tk


# Abstract class i.e. cannot be instantiated
class LockableTab(tk.Frame):
    """Base class for creating `ttk.Notebook` tabs that are easily locked and unlocked.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent, parent_window):
        """Initializes the tab.

        The locking and unlocking mechanism works by switching between which child frame of the base frame self._base
        is raised. The 'locked' frame of standard type LockedFrame is raised when the tab is to be locked whereas the
        'unlocked' frame is raised when the tab is to be unlocked. The 'unlocked' frame is specified by the user by
        implementing the abstract method get_normal_frame(). The method extends the __init__() method in tk.Frame.

        :param parent: the parent widget of this LockableTab widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this LockableTab widget.
        :type parent_window: `tkinter.Toplevel`
        """
        self._parent = parent
        self._parent_window = parent_window
        self._normal_frame = None

        tk.Frame.__init__(self, self._parent)

        self._base = tk.Frame(self)
        self._base.pack(side='top', fill='both', expand=True)
        self._base.grid_rowconfigure(0, weight=1)
        self._base.grid_columnconfigure(0, weight=1)

        self._frames = None

        self._frames = {'locked': LockedFrame(self._base),
                        'unlocked': self.get_normal_frame()}
        for f in self._frames.values():
            f.grid(row=0, column=0, sticky="nsew")

    def lock(self):
        """Raise the 'locked' child frame of the base frame over the 'unlocked' child frame of the base frame."""
        if self._frames is not None:
            frame = self._frames['locked']
            frame.tkraise()

    def unlock(self):
        """Raise the 'unlocked' child frame of the base frame over the 'locked' child frame of the base frame."""
        if self._frames is not None:
            frame = self._frames['unlocked']
            frame.tkraise()

    def get_base_frame(self):
        """Return the base frame widget."""
        return self._base

    def get_normal_frame(self):
        """Abstract method to be implemented in subclasses to return the 'unlocked' frame."""
        pass


class LockedFrame(tk.Frame):
    """Standard class to be used as the 'locked' frame in a LockableTab object.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent):
        """Populates the frame with a simple label indicating the tab's status as 'locked'.

        :param parent: the parent widget of this LockableFrame widget.
        :type parent: `tkinter widget`
        """
        tk.Frame.__init__(self, parent)
        message = tk.Label(self, text="You must load your data set before proceeding to this section.")
        message.pack(anchor=tk.CENTER, expand=True)
