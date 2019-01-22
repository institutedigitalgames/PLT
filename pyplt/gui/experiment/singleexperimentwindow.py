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
from tkinter import ttk
import os

from pyplt import ROOT_PATH
from pyplt.gui.experiment.dataset.loading import DataLoadingTab
from pyplt.gui.experiment.featureselection.featselectiontab import FeatureSelectionTab
from pyplt.gui.experiment.preflearning.pltab import PLTab
from pyplt.gui.experiment.preprocessing.preproctab import PreProcessingTab
from pyplt.gui.util import windowstacking as ws, colours
from pyplt.gui.util.help import LoadDataHelpDialog, PreprocHelpDialog, FSHelpDialog, PLHelpDialog


class SingleExperimentWindow(tk.Toplevel):
    """GUI window for setting up and running a single experiment.

    Extends the class `tkinter.Toplevel`. Each experiment stage is encapsulated in a separate tab, together comprising
    a `ttk.Notebook` widget which is controlled by this class. This class also manages the state of the
    tabs (locked/unlocked) as well as the flow of data between the tabs.
    """

    def __init__(self, parent):
        """Initializes the `ttk.Notebook` widget in the given window.

        The `ttk.Notebook` widget is populated with tabs defined by the classes
        :class:`pyplt.gui.experiment.dataset.loading.DataLoadingTab`,
        :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingTab`,
        :class:`pyplt.gui.experiment.featureselection.featselectiontab.FeatureSelectionTab`, and
        :class:`pyplt.gui.experiment.preflearning.pltab.PLTab`.

        :param parent: the parent window on which this experiment setup window will be stacked.
        :type parent: `tkinter.Toplevel`
        """
        self._parent = parent

        tk.Toplevel.__init__(self, parent, height=250)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, parent, ws.Mode.WITH_CLOSE)
        ws.stack_window(self, parent)

        self.title("Experiment Setup (Advanced)")

        self._tabs_bar = ttk.Notebook(self)

        self._files_tab = DataLoadingTab(self._tabs_bar, self)

        self._preproc_tab = PreProcessingTab(self._tabs_bar, self, self._files_tab)
        self._fs_tab = FeatureSelectionTab(self._tabs_bar, self, self._files_tab)
        self._pl_tab = PLTab(self._tabs_bar, self, self._files_tab, self._preproc_tab, self._fs_tab)

        self._pic = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/icons/load_icon_18_light_blue.png"))
        self._pp_pic = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/icons/preprocessing_icon_18_blue.png"))
        self._fs_pic = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/icons/fs_icon_18.png"))
        self._pl_pic = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/icons/pl_icon_18.png"))

        self._tabs_bar.add(self._files_tab, image=self._pic, text="Load Data", compound=tk.LEFT, sticky='nsew')
        self._tabs_bar.add(self._preproc_tab, text="Preprocessing", image=self._pp_pic, compound=tk.LEFT,
                           sticky='nsew')
        self._tabs_bar.add(self._fs_tab, text="Feature Selection", image=self._fs_pic, compound=tk.LEFT, sticky='nsew')
        self._tabs_bar.add(self._pl_tab, text="Preference Learning", image=self._pl_pic, compound=tk.LEFT,
                           sticky='nsew')
        self._tabs_bar.pack(fill=tk.BOTH, expand=True)

        self._nav_frame = tk.Frame(self, bg=colours.NAV_BAR)
        self._nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self._next_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/next_76_30_01.png"))
        self._back_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/back_76_30_01.png"))
        self._run_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/run_128_30_01_white.png"))
        self._help_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/help.png"))

        self._back_btn = tk.Button(self._nav_frame, command=self._select_prev_tab, image=self._back_img, relief='flat',
                                   bd=0, highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR,
                                   highlightthickness=0, background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        self._next_btn = tk.Button(self._nav_frame, command=self._select_next_tab, image=self._next_img, relief='flat',
                                   bd=0, highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR,
                                   highlightthickness=0, background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        self._run_btn = tk.Button(self._nav_frame, command=self._pl_tab.run_experiment, image=self._run_img,
                                  relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                                  highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                                  activebackground=colours.NAV_BAR)

        help_btn = tk.Button(self._nav_frame, command=self._help_dialog, image=self._help_img, relief='flat', bd=0,
                             highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                             background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        help_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # Lock tabs for the first time
        self._data_loaded = False
        self._preproc_tab.lock()
        self._fs_tab.lock()
        self._pl_tab.lock()

        self._tabs_bar.bind('<<NotebookTabChanged>>', self._check_data)

        ws.place_window(self, parent, position=ws.SIDE)

    def _help_dialog(self):
        """Open a help dialog window to assist the user.

        The help shown on the dialog depends on the current tab.
        """
        curr_tab_id = self._get_curr_tab_idx()
        if curr_tab_id == 0:  # Load Data tab
            LoadDataHelpDialog(self)
        elif curr_tab_id == 1:  # Preprocessing tab
            PreprocHelpDialog(self)
        elif curr_tab_id == 2:  # FS tab
            FSHelpDialog(self)
        else:  # PL tab
            PLHelpDialog(self)

    def _check_data(self, event):
        """Check whether data has been loaded on notebook tab change and update tab states accordingly.

        This method handles <<NotebookTabChanged>> events (with respect to the `ttk.Notebook` widget). If data is
        loaded, all tabs are unlocked. Otherwise, only the `Load Data` tab is unlocked, while the others are locked.

        :param event: the <<NotebookTabChanged>> event triggering the method call.
        :type event: `tkinter widget`
        """
        # check (only once) if new data has been loaded which replaces previous data
        new_data = self._files_tab.is_new_data()
        # print("is new data? " + str(new_data))
        # print("Checkin' if data was loaded...")
        if self._files_tab.is_data_loaded():
            if not self._data_loaded:  # only do this the first time user changes tab after loading data, not every time
                self._preproc_tab.unlock()
                self._fs_tab.unlock()
                self._pl_tab.unlock()
                self._data_loaded = True
            elif new_data:
                # print("refresh preproc tab!")
                self._preproc_tab.refresh()  # refresh preproc tab!
        else:
            if self._data_loaded:  # only do this the first time user changes tab when data NOT loaded, not every time
                self._preproc_tab.lock()
                self._fs_tab.lock()
                self._pl_tab.lock()
                self._data_loaded = False

        # also update navigation bar at bottom
        self._update_nav_bar()

    def _get_curr_tab_idx(self):
        """Get the index of the `ttk.Notebook` tab which is currently open.

        :return: the index of the current tab.
        :rtype: int
        """
        return self._tabs_bar.index(self._tabs_bar.select())

    def _select_next_tab(self):
        """Programmatically switch to the `ttk.Notebook` tab immediately succeeding the one that is currently open."""
        self._tabs_bar.select(self._get_curr_tab_idx()+1)

    def _select_prev_tab(self):
        """Programmatically switch to the `ttk.Notebook` tab immediately preceding the one that is currently open."""
        self._tabs_bar.select(self._get_curr_tab_idx()-1)

    def _update_nav_bar(self):
        """Update the navigation bar interface at the bottom of the window according to the currently open tab.

        The navigation bar allows the user to navigate between tabs in the ttk.Notebook widget. If the tab is the
        first in the notebook, only a 'Next' button is shown. Conversely, if the tab is the last, a 'Back' button is
        shown together with a 'Run Experiment' button, the latter of which is disabled until a full data set has
        been loaded. Otherwise, both a 'Back' and a 'Next' button are shown.
        """
        selected_tab_idx = self._get_curr_tab_idx()

        # first remove all buttons
        self._next_btn.pack_forget()
        self._back_btn.pack_forget()
        self._run_btn.pack_forget()

        # now pack the buttons according to the current tab
        if selected_tab_idx == 3:  # Preference Learning (last tab)
            # show Back and Run Experiment buttons
            self._run_btn.pack(side=tk.RIGHT, pady=10, padx=(5, 10))
            self._back_btn.pack(side=tk.RIGHT, pady=10, padx=(10, 5))
            # also, disable Run Experiment button until data is loaded
            if self._data_loaded:
                self._run_btn.configure(state='normal')
            else:
                self._run_btn.configure(state='disable')
            self._run_btn.bind("<<PLTStateToggle>>", self._check_run_btn)  # bind
            # ^ n.b. ensured that 'Run Experiment' button, OutputLayer#nuerons checkbox and Steps 2-4 of BeginnerMenu
            # are re-disabled or re-enabled accordingly on close of stacked windows (help dialog or load params).
            # solution via binding state changes to method which ensures re-disable (or re-enable if appropriate).
        elif selected_tab_idx == 0:  # Load Data (first tab)
            # show Next button only
            self._next_btn.pack(side=tk.RIGHT, pady=10, padx=10)
        else:  # Preprocessing, Feature Selection
            # show Back and Next buttons
            self._next_btn.pack(side=tk.RIGHT, pady=10, padx=(5, 10))
            self._back_btn.pack(side=tk.RIGHT, pady=10, padx=(10, 5))

    def _check_run_btn(self, event):
        """Ensure that the 'Run Experiment' Button widget is in the correct state when a change in its state is detected.

        :param event: the <<PLTStateToggle>> event that triggered the call to this method.
        :type event: `tkinter widget`
        """
        new_state = str(self._run_btn.cget('state'))
        # print("new_state: " + str(new_state))
        if (new_state == 'disable') or (new_state == 'disabled'):
            # print("Run button state was changed (disabled)! -- ignoring.")
            return
        # print("Run button state was changed (activated)!")
        # re-disable Run Experiment button until data is loaded
        if self._data_loaded:
            # print("Data finally loaded. Unbind.")
            self._run_btn.unbind_all("<<PLTStateToggle>>")  # stop  # unbind
            self._run_btn.configure(state='normal')
            # ^ n.b. state name does not always apply - check for specific widget!
        else:
            # print("Data not loaded. Setting back to disabled.")
            self._run_btn.configure(state='disable')  # set back to disabled
            # ^ n.b. state name does not always apply - check for specific widget!
