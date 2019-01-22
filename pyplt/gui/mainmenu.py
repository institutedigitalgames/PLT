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

import tkinter as tk
from tkinter import font
import os

from pyplt import ROOT_PATH
from pyplt.gui.experiment.singleexperimentwindow import SingleExperimentWindow
from pyplt.gui.beginnermenu import BeginnerMenu
from pyplt.gui.util import colours
from pyplt.gui.util.help import AboutBox, MainHelpDialog


class MainMenu:
    """Main menu window widget.

    This menu allows the user to select whether to run PLT in a Beginner mode or Advanced mode.
    """

    def __init__(self, master):
        """Initializes and populates the main menu window widget.

        :param master: the root window on which the main menu will be displayed.
        :type master: `tkinter.Tk`
        """
        self._master = master

        text_frame = tk.Frame(self._master)
        text_frame.pack(padx=20, pady=20)

        ebrima_big = font.Font(family='Ebrima', size=12, weight=font.NORMAL)
        # ebrima_small = font.Font(family='Ebrima', size=10, weight=font.NORMAL)

        # Welcome text
        tk.Label(text_frame, text="Welcome to Preference Learning Toolbox (PLT)!", font=ebrima_big).pack()
        tk.Label(text_frame, text="Select your preferred mode:").pack(pady=(20, 0))

        button_frame = tk.Frame(self._master)
        button_frame.pack(padx=50, pady=(0, 20))

        self._beginner_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/beginner_128_30_01.png"))
        self._advanced_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/advanced_128_30_01.png"))
        self._help_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/help.png"))
        self._about_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/about.png"))

        # Options
        opt1_btn = tk.Button(button_frame, command=self._beginner_menu, image=self._beginner_img, relief='flat', bd=0)
        opt1_btn.grid(row=0, column=0, padx=100, pady=(20, 10), sticky='ew')
        opt2_btn = tk.Button(button_frame, command=self._advanced_menu, image=self._advanced_img, relief='flat', bd=0)
        opt2_btn.grid(row=1, column=0, padx=100, pady=(0, 20), sticky='ew')

        nav_frame = tk.Frame(self._master, bg=colours.NAV_BAR)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        about_btn = tk.Button(nav_frame, command=self._about_box, image=self._about_img, relief='flat', bd=0,
                              highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR,
                              highlightthickness=0, background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        about_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        help_btn = tk.Button(nav_frame, command=self._help_dialog, image=self._help_img, relief='flat', bd=0,
                             highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR,
                             highlightthickness=0, background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        help_btn.pack(side=tk.LEFT, padx=10, pady=10)

    def _about_box(self):
        """Open a window containing text on the details of the PLT software and its license."""
        AboutBox(self._master)

    def _help_dialog(self):
        """Open a help dialog window to assist the user."""
        MainHelpDialog(self._master)

    def _beginner_menu(self):
        """Open the PLT menu for beginner users when the 'Beginner' button is clicked.

        Instantiates a :class:`pyplt.gui.beginnermenu.BeginnerMenu` widget.
        """
        self.se_win = BeginnerMenu(self._master)

    def _advanced_menu(self):
        """Open the PLT menu for advanced users when the 'Advanced' button is clicked.

        Instantiates a :class:`pyplt.gui.experiment.singleexperimentwindow.SingleExperimentWindow` widget.
        """
        self.se_win = SingleExperimentWindow(self._master)
