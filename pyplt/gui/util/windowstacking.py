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

from enum import Enum
import tkinter as tk
from tkinter import ttk
"""This module contains classes and functions enabling `tkinter.Toplevel` windows to be stacked on top of each other."""

ON_TOP = 0
SIDE = 1


class Mode(Enum):
    """Class specifying enumerated constants for the different modes of window stacking.

    Extends `enum.Enum`.
    """
    WITH_CLOSE = 0
    OPEN_ONLY = 1  # meaning we will likely override/replace the on_close method


def place_window(self, parent, position=SIDE):
    """Set the geometry of child window to a given position relative to the parent window's geometry.

    :param self: the new window being stacked on top.
    :type self: `tkinter.Toplevel`
    :param parent: the parent of the new window widget to be stacked above it.
    :type parent: `tkinter.Toplevel`
    :param position: the position at which to place the new window relative to the parent window's geometry. If 0, the
        new window will be placed directly on top of the parent window. If 1, the new window will be placed to the side
        of the parent window (default 1).
    :type position: int: 0 or 1, optional
    """
    self.update_idletasks()

    x = parent.winfo_rootx()
    y = parent.winfo_rooty()
    parent_height = parent.winfo_height()
    parent_width = parent.winfo_width()
    self_height = self.winfo_height()
    self_width = self.winfo_width()

    screen_width = self.winfo_screenwidth()

    screen_height = self.winfo_screenheight()
    centered_y = (screen_height/2) - (self_height/2)

    if position == ON_TOP:  # (centered)
        geom = "+%d+%d" % (x + parent_width/2 - self_width/2, y + parent_height/2 - self_height/2)
    elif position == SIDE:  # RIGHT (default)
        offset_x = x + parent_width + 5
        # print("proposed offset_x: " + str(offset_x))
        # first check if it's going to go out of screen!
        if (offset_x + self_width) > screen_width:
            offset_x = x - (20 + self_width)  # open to parent's left instead
            print("won't fit in screen. switch to left instead. new offset_x: " + str(offset_x))
        geom = "+%d+%d" % (offset_x, centered_y)  # , y
    else:  # otherwise do nothing
        return
    self.wm_geometry(geom)


def disable_parent(self, parent, mode):
    """Change state of parent window to 'disabled'.

    :param self: the new window being stacked on top.
    :type self: `tkinter.Toplevel`
    :param parent: the parent of the new window widget to be stacked above it.
    :type parent: `tkinter.Toplevel`
    :param mode: the window stacking mode.
    :type mode: :class:`pyplt.gui.util.windowstacking.Mode`
    """
    if mode == Mode.WITH_CLOSE:
        # Enable re-enabling of parent window
        self.protocol("WM_DELETE_WINDOW", lambda: on_close(self, parent))

    # Disable parent window
    _toggle_state(parent, self, 'disable')


def _toggle_state(widget, new_window, new_state, tab_exception=False):
    """Toggle the state of every child widget of the given widget between 'normal' and 'disabled'.

    :param widget: the widget of which all children widget states are to be toggled.
    :type widget: `tkinter widget`
    :param new_window: the new window widget to be stacked upon `widget` if applicable (can be None if
        a stacked window is being closed).
    :type new_window: `tkinter.Toplevel` or None
    :param new_state: the state to which all children of widget are to be changed.
    :type new_state: str: `disable` or `disabled` or `active` or `normal`
    :param tab_exception: flag indicating whether widget is a currently-active `ttk.Notebook` tab (default False).
    :type tab_exception: bool, optional
    """
    # recursive function to make sure all levels are disabled!
    if (len(widget.winfo_children()) == 0) and (not (isinstance(widget, tk.Frame) or isinstance(widget, ttk.Frame))):
        # except (empty) frames bc they don't have a state attribute...
        # print(str(widget))
        # print(str(new_state))
        if not isinstance(widget, ttk.Separator):  # no need to disable separator...
            if isinstance(widget, ttk.Scrollbar) or isinstance(widget, ttk.Scale) or isinstance(widget, ttk.Treeview):
                if new_state == 'disable':
                    new_state = 'disabled'
                elif new_state == 'active' or new_state == 'normal':
                    new_state = '!disabled'
                widget.state([new_state])
                widget.event_generate("<<PLTStateToggle>>")
            else:
                widget.configure(state=new_state)
                widget.event_generate("<<PLTStateToggle>>")
    else:
        c = 0
        for child in widget.winfo_children():
            # go through all children of the widget (unless it's the new window)
            if child == new_window:
                continue
            # print("Toggle child state ("+new_state+") : " + str(child))
            # print("Toggle child state ("+new_state+") : " + children[c])
            # print("Toggle child state (" + new_state + ") : " + str(children2[c]))
            if "optionmenu" in str(child):
                if new_state == 'disable':
                    new_state = 'disabled'
                elif new_state == 'active':
                    new_state = 'normal'
                child.config(state=new_state)
                child.event_generate("<<PLTStateToggle>>")
            elif isinstance(child, ttk.Notebook):
                # special case for notebook frames!
                t = 0
                selected = child.select()
                for tab in range(len(child.winfo_children())):
                    # print(selected)
                    if not (child.index(selected) == tab):
                        # print(str(children[c]) + ", " + str(children2[c]))
                        # print(tab)
                        if new_state == 'active':
                            new_state = 'normal'
                        child.tab(tab, state=new_state)
                        # CANNOT GENERATE PLTStateToggle EVENT FOR OTHER NOTEBOOK TABS!
                    else:
                        # but disable all children of current tab
                        _toggle_state(child.winfo_children()[tab], new_window, new_state, tab_exception=True)
                    t += 1
            else:
                if not (isinstance(widget, ttk.Notebook)) or tab_exception:  # no tabs except current tab
                    # ^ no need to en/disable widgets in the tab since the tabs themselves are en/disabled
                    _toggle_state(child, new_window, new_state, tab_exception)
            c += 1


def on_close(new_window, parent):
    """Restore the parent window and all its children widgets to their 'normal' state and bring it back on top.

     The parent window is elevated back to the topmost level of the stack of windows.

    This method is toggled when a stacked window is closed and its stacking mode was set to WITH_CLOSE.

    :param new_window: the stacked window being closed.
    :type new_window: `tkinter.Toplevel`
    :param parent: the parent window of new_window.
    :type parent: `tkinter.Toplevel`
    """
    # Enable parent window
    _toggle_state(parent, new_window, 'normal')

    new_window.attributes('-topmost', False)
    # Destroy self
    new_window.destroy()
    # print("FINISHED UNSTACKING...")

    # restore window stacking order
    parent.deiconify()
    parent.attributes('-topmost', True)
    parent.update()
    parent.attributes('-topmost', False)


def stack_window(new_win, parent_win):
    """Elevate the new window to the topmost level of the stack of windows.

    :param new_win: the new window to be stacked on top.
    :type new_win: `tkinter.Toplevel`
    :param parent_win: the parent window of new_win.
    :type parent_win: `tkinter.Toplevel`
    """
    new_win.lift(parent_win)
    parent_win.attributes('-topmost', False)
    new_win.attributes('-topmost', True)
    new_win.attributes('-topmost', False)
