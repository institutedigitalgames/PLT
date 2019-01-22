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

import platform
import tkinter as tk
from tkinter import ttk

from pyplt.gui.experiment.preflearning.ranksvm_menu import RankSVMMenu
from pyplt.gui.util import colours
from pyplt.util.enums import FSMethod, PLAlgo, EvaluatorType
from pyplt.gui.experiment.preflearning.evaluator_menus import HoldoutMenu, KFCVMenu
from pyplt.gui.experiment.preflearning.pltab import BackpropMenu
from pyplt.gui.util.tab_locking import LockableTab
from pyplt.gui.util import supported_methods


class FeatureSelectionTab(LockableTab):
    """GUI tab for the feature selection stage of setting up an experiment.

    Extends the class :class:`pyplt.gui.util.tab_locking.LockableTab` which, in turn, extends the
    `tkinter.Frame` class.
    """

    def __init__(self, parent, parent_window, files_tab):
        """Initializes the `FeatureSelectionTab` object.

        :param parent: the parent widget of this tab (frame) widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this tab (frame) widget.
        :type parent_window: `tkinter.Toplevel`
        :param files_tab: the `Load Data` tab.
        :type files_tab: :class:`pyplt.gui.experiment.dataset.loading.DataLoadingTab`
        """
        self._parent = parent
        self._parent_window = parent_window
        self._files_tab = files_tab
        self._frame = None
        LockableTab.__init__(self, self._parent, self._parent_window)

    def get_normal_frame(self):
        """Return a `FeatureSelectionFrame` widget for when the tab is in the 'unlocked' state.

        The `FeatureSelectionFrame` widget is instantiated only once on the first occasion that the tab is 'unlocked'.

        :return: the `FeatureSelectionFrame` widget that is visible whenever the tab is in the 'unlocked' state.
        :rtype: :class:`pyplt.gui.experiment.featureselection.featselectiontab.FeatureSelectionFrame`
        """
        if self._frame is None:
            self._frame = FeatureSelectionFrame(self.get_base_frame(), self._parent_window, self._files_tab)
        return self._frame

    def get_fs_method(self):
        """Get the feature selection method type chosen by the user via the `FeatureSelectionFrame`.

        :return: the feature selection method type chosen by the user.
        :rtype: :class:`pyplt.util.enums.FSMethod`
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_method()

    def get_fs_method_params(self):
        """Get the parameters of the feature selection method chosen by the user (if applicable).

        :return: the parameters of the feature selection method chosen by the user (if applicable).
        :rtype: list
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_method_params()

    def get_fs_algorithm(self):
        """Get the preference learning algorithm type chosen by the user via the `FeatureSelectionFrame` (if applicable).

        :return: the preference learning algorithm type chosen by the user (if applicable).
        :rtype: :class:`pyplt.util.enums.PLAlgo`
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_algorithm()

    def get_fs_algorithm_params(self):
        """Get the parameters of the preference learning algorithm chosen by the user (if applicable).

        :return: the parameters of the preference learning algorithm chosen by the user (if applicable).
        :rtype: list
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_algorithm_params()

    def get_fs_evaluator(self):
        """Get the evaluation method type chosen by the user via the `FeatureSelectionFrame` (if applicable).

        :return: the evaluation method type chosen by the user (if applicable).
        :rtype: :class:`pyplt.util.enums.EvaluatorType`
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_evaluator()

    def get_fs_evaluator_params(self):
        """Get the parameters of the evaluation method chosen by the user (if applicable).

        :return: the parameters of the evaluation method chosen by the user (if applicable).
        :rtype: list
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_evaluator_params()


class FeatureSelectionFrame(tk.Frame):
    """Frame widget that is visible whenever the `Feature Selection` tab is in the 'unlocked' state.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent, parent_window, files_tab):
        """Initializes the frame widget.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this frame widget.
        :type parent_window: `tkinter.Toplevel`
        :param files_tab: the `Load Data` tab.
        :type files_tab: :class:`pyplt.gui.experiment.dataset.loading.DataLoadingTab`
        """
        self._parent = parent
        self._parent_window = parent_window
        self._method_name = tk.StringVar(value=FSMethod.NONE.name)
        self._fs_params = None
        self._algorithm_name = None
        self._algo_menu = None
        self._algorithm_sub_menus = {}
        self._ne_menu = None
        self._algorithm_params = []
        self._evaluator_name = None
        self._evaluator_menu = None
        self._evaluator_sub_menus = {}
        self._a_hidden = False
        self._e_hidden = False
        self._files_tab = files_tab

        self._OS = platform.system()

        tk.Frame.__init__(self, parent)

        self._main_canvas = tk.Canvas(self)
        self._main_canvas.bind("<Configure>", self._on_resize)
        self._main_canvas.bind('<Enter>', self._bind_mousewheel)
        self._main_canvas.bind('<Leave>', self._unbind_mousewheel)
        self._canvas_height = self._main_canvas.winfo_reqheight()
        self._canvas_width = self._main_canvas.winfo_reqwidth()
        self._main_sub_frame = tk.Frame(self._main_canvas)
        self._main_sub_sub_frame = tk.Frame(self._main_sub_frame)

        # populate frame

        self._fs_menu = tk.Frame(self._main_sub_sub_frame, background=colours.FS_OUTER)
        self._fs_menu.pack(fill=tk.X, expand=True)

        sub_fs_frame = tk.Frame(self._fs_menu, background=colours.FS_OUTER)
        sub_fs_frame.pack(padx=75, pady=10)

        options = [key.name for key in supported_methods.supported_fs_methods.keys()]
        tk.Label(sub_fs_frame, text="Choose Feature Selection Method",
                 background=colours.FS_OUTER).grid(row=0, column=0)
        ttk.OptionMenu(sub_fs_frame, self._method_name, options[0], *options,
                       command=lambda _: self._method_update(), style='FS.PLT.TMenubutton').grid(row=0, column=1)

        # add scrollbars

        v_scroll = ttk.Scrollbar(self, orient="vertical", command=self._main_canvas.yview,
                                 style="PLT.Vertical.TScrollbar")
        v_scroll.pack(side='right', fill='y')
        self._main_canvas.configure(yscrollcommand=v_scroll.set)
        h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self._main_canvas.xview,
                                 style="PLT.Horizontal.TScrollbar")
        h_scroll.pack(side='bottom', fill='x')
        self._main_canvas.configure(xscrollcommand=h_scroll.set)

        # pack everything
        self._main_sub_sub_frame.pack()
        self._main_sub_frame.pack(fill=tk.BOTH, expand=True)  # useless line... doesn't work here it seems
        self._main_canvas.pack(side='left', expand=True, fill=tk.BOTH)

        self.c_win = self._main_canvas.create_window((0, 0), window=self._main_sub_frame, anchor='nw')
        self._main_canvas.config(scrollregion=self._main_canvas.bbox("all"))

        self._main_sub_frame.bind('<Configure>', self._on_canvas_config)

    def _on_canvas_config(self, event):
        """Update the canvas `scrollregion` to account for the entire area of the :attr:`self._main_sub_frame` widget.

        This method is bound to all <Configure> events with respect to :attr:`self._main_sub_frame`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("__ on_config called __")
        self._main_canvas.configure(scrollregion=(0, 0, self._main_sub_frame.winfo_reqwidth(),
                                                  self._main_sub_frame.winfo_reqheight()))

    def get_method(self):
        """Get the feature selection method type chosen by the user.

        :return: the feature selection method type chosen by the user.
        :rtype: :class:`pyplt.util.enums.FSMethod`
        """
        try:
            return FSMethod[self._method_name.get()]
        except AttributeError:
            return None

    def get_method_params(self):
        """Get the parameters of the feature selection method chosen by the user (if applicable).

        :return: the parameters of the feature selection method chosen by the user (if applicable).
        :rtype: list
        """
        return self._fs_params

    def get_algorithm(self):
        """Get the preference learning algorithm type chosen by the user (if applicable).

        :return: the preference learning algorithm type chosen by the user (if applicable).
        :rtype: :class:`pyplt.util.enums.PLAlgo`
        """
        try:
            if not self._a_hidden:
                return PLAlgo[self._algorithm_name.get()]
            else:
                return None
        except AttributeError:
            return None

    def get_algorithm_params(self):
        """Get the parameters of the preference learning algorithm chosen by the user (if applicable).

        :return: the parameters of the preference learning algorithm chosen by the user (if applicable).
        :rtype: list
        """
        try:
            # print("WORKING")
            if not self._a_hidden:  # in case self._algo_menu is in pack_forget state
                # print("PACKED")
                algo_name = self._algorithm_name.get()
                return self._algorithm_sub_menus[algo_name].get_params()
            else:
                # print("NOT PACKED")
                return None
        except AttributeError:  # in case self._algo_menu has never been opened yet
            # or in case the given algo menu does not have get_params() method
            # print("NOT WORKING")
            return None

    def get_evaluator(self):
        """Get the evaluation method type chosen by the user (if applicable).

        :return: the evaluation method type chosen by the user (if applicable).
        :rtype: :class:`pyplt.util.enums.EvaluatorType`
        """
        try:
            if not self._e_hidden:
                return EvaluatorType[self._evaluator_name.get()]
            else:
                return None
        except AttributeError:
            return None

    def get_evaluator_params(self):
        """Get the parameters of the evaluation method chosen by the user (if applicable).

        :return: the parameters of the evaluation method chosen by the user (if applicable).
        :rtype: list
        """
        try:
            if not self._e_hidden:
                eval_name = self._evaluator_name.get()
                if eval_name != EvaluatorType.NONE.name:
                    return self._evaluator_sub_menus[eval_name].get_params()
        except AttributeError:  # in case the given eval menu does not have get_params() method
            return None
        return None  # in case NONE is chosen

    def _method_update(self):
        """Add or hide the algorithm and evaluator menus according to the chosen feature selection method."""
        method = FSMethod[self._method_name.get()]
        if method == FSMethod.N_BEST or method == FSMethod.SFS or method == FSMethod.SBS:
            # wrapper method
            row = 1

            if self._algo_menu is None:
                self._algo_menu = tk.Frame(self._main_sub_sub_frame, bg=colours.PL_OUTER)
                self._algo_menu.pack(fill=tk.X, expand=True)  # would remain empty if fs method is not a wrapper method
                self._algorithm_name = tk.StringVar(value=PLAlgo.RANKSVM.name)

                select = tk.Frame(self._algo_menu, bg=colours.PL_OUTER)
                select.pack(padx=75, pady=10)

                # show algorithm selection menu
                tk.Label(select, text="Choose Preference Learning Algorithm",
                         bg=colours.PL_OUTER, fg='white').grid(row=row, column=0)
                options = [key.name for key in supported_methods.supported_algorithms.keys()]
                ttk.OptionMenu(select, self._algorithm_name, options[0], *options,
                               command=lambda _: self._update_algo_menu(),
                               style='PL.PLT.TMenubutton').grid(row=row, column=1)
                # force algo menu update for default algo
                self._update_algo_menu()
            else:
                self._algo_menu.pack(fill=tk.X, expand=True)
                self._a_hidden = False

            # since we are here, the fs method is definitely a wrapper method
            # so we also need an evaluator/validator
            if self._evaluator_menu is None:
                self._evaluator_menu = tk.Frame(self._main_sub_sub_frame, bg=colours.EVAL_OUTER)
                self._evaluator_menu.pack(fill=tk.X,
                                          expand=True)  # would remain empty if fs method is not a wrapper method
                self._evaluator_name = tk.StringVar(value=EvaluatorType.NONE.name)
                # show evaluator selection menu
                select = tk.Frame(self._evaluator_menu, bg=colours.EVAL_OUTER)
                select.pack(padx=75, pady=10)
                tk.Label(select, text="Choose Evaluation Method", bg=colours.EVAL_OUTER,
                         fg='white').grid(row=0, column=0)
                options = [key.name for key in supported_methods.supported_evaluation_methods.keys()]
                ttk.OptionMenu(select, self._evaluator_name, options[0], *options,
                               command=lambda _: self._update_eval_menu(),
                               style='Eval.PLT.TMenubutton').grid(row=0, column=1)
            else:
                self._evaluator_menu.pack(fill=tk.X, expand=True)
                self._e_hidden = False
        else:
            if self._algo_menu is not None:
                self._algo_menu.pack_forget()
                self._a_hidden = True
            if self._evaluator_menu is not None:
                self._evaluator_menu.pack_forget()
                self._e_hidden = True
        # force updates for canvas and scrollbar stuff
        self.update_idletasks()
        self._on_resize(None)
        self._on_canvas_config(None)

    def _update_algo_menu(self):
        """Display the algorithm parameter menu corresponding to the algorithm chosen by the user.

        The menu allows the user to specify the parameters to use for preference learning during the feature
        selection process (if applicable).
        """
        # this method is exactly the same as that in pltab!  # TODO: keep up to date with corresponding method in pltab
        algo = self._algorithm_name.get()
        sel_type = PLAlgo[algo]

        exists = False
        for algo_name, algo_menu in self._algorithm_sub_menus.items():
            if algo_name == algo:
                # menu for selected algorithm/method was already created, so just pack
                exists = True
                algo_menu.pack(fill=tk.BOTH, expand=True, padx=50, pady=(5, 15))
            else:
                # hide the menus of the other algorithms/methods, use pack_forget()
                algo_menu.pack_forget()
        if not exists:
            # create algorithm/method menu for first time
            # get corresponding GUI menu class and instantiate
            # TODO: for special cases (e.g., BACKPROPAGATION) where algorithm GUI menu constructor requires
            # additional arguments, add an if statement here
            if sel_type == PLAlgo.BACKPROPAGATION or sel_type == PLAlgo.BACKPROPAGATION_SKLEARN:
                new_menu = supported_methods.supported_algorithms[sel_type][1](self._algo_menu, self._on_resize)
            else:
                new_menu = supported_methods.supported_algorithms[sel_type][1](self._algo_menu)
            new_menu.pack(fill=tk.BOTH, expand=True, padx=50, pady=(5, 15))
            self._algorithm_sub_menus[algo] = new_menu

        # force updates for canvas and scrollbar stuff
        self.update_idletasks()
        self._on_resize(None)
        self._on_canvas_config(None)

    def _update_eval_menu(self):
        """Display the evaluation method parameter menu corresponding to the evaluation method chosen by the user.

        The menu allows the user to specify the parameters to use for evaluation during the feature
        selection process (if applicable).
        """
        # this method is exactly the same as that in pltab!!  # TODO: keep up to date with corresponding method in pltab
        eval = self._evaluator_name.get()
        sel_type = EvaluatorType[eval]

        exists = False
        for method_name, method_menu in self._evaluator_sub_menus.items():
            if method_name == eval:
                # menu for selected algorithm/method was already created, so just pack
                exists = True
                method_menu.pack(fill=tk.BOTH, expand=True, padx=50, pady=(5, 15))
            else:
                # hide the menus of the other algorithms/methods, use pack_forget()
                method_menu.pack_forget()
        if (sel_type != EvaluatorType.NONE) and (not exists):
            # create algorithm/method menu for first time
            # get corresponding GUI menu class and instantiate
            # TODO: for special cases (e.g., KFCV) where the evaluation method GUI menu constructor requires
            # additional arguments, add an if statement here
            if sel_type == EvaluatorType.KFCV:
                new_menu = supported_methods.supported_evaluation_methods[sel_type][1](self._evaluator_menu,
                                                                                       self._parent_window,
                                                                                       files_tab=self._files_tab,
                                                                                       on_resize_fn=self._on_resize)
            else:
                new_menu = supported_methods.supported_evaluation_methods[sel_type][1](self._evaluator_menu)
            new_menu.pack(fill=tk.BOTH, expand=True, padx=50, pady=(5, 15))
            self._evaluator_sub_menus[eval] = new_menu

        # force updates for canvas and scrollbar stuff
        self.update_idletasks()
        self._on_resize(None)
        self._on_canvas_config(None)

    def _on_resize(self, event):
        """Resize the canvas widget according to the user's specification via the mouse.

        This method is called whenever a <Configure> event occurs with respect to :attr:`self._main_canvas`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        if event is not None:  # otherwise use latest values of self._canvas_width and self._canvas_height
            # for forcing updates for canvas/scrollbars
            self._canvas_width = event.width
            self._canvas_height = event.height
        # print("event/canvas width = " + str(self._canvas_width))
        # print("event/canvas height = " + str(self._canvas_height))
        if self._canvas_width > self._main_sub_frame.winfo_reqwidth():
            self._main_canvas.itemconfig(self.c_win, width=self._canvas_width)
        else:
            self._main_canvas.itemconfig(self.c_win, width=self._main_sub_frame.winfo_reqwidth())
        if self._canvas_height > self._main_sub_frame.winfo_reqheight():
            self._main_canvas.itemconfig(self.c_win, height=self._canvas_height)
        else:
            self._main_canvas.itemconfig(self.c_win, height=self._main_sub_frame.winfo_reqheight())

    def _bind_mousewheel(self, event):
        """Bind all mouse wheel events with respect to the canvas to a canvas-scrolling function.

        This method is called whenever an <Enter> event occurs with respect to :attr:`self._main_canvas`.

        :param event: the <Enter> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # for Windows OS and MacOS
        self._main_canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        # for Linux OS
        self._main_canvas.bind_all("<Button-4>", self._on_mouse_scroll)
        self._main_canvas.bind_all("<Button-5>", self._on_mouse_scroll)

    def _unbind_mousewheel(self, event):
        """Unbind all mouse wheel events with respect to the canvas from any function.

        This method is called whenever a <Leave> event occurs with respect to :attr:`self._main_canvas`.

        :param event: the <Leave> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # for Windows OS and MacOS
        self._main_canvas.unbind_all("<MouseWheel>")
        # for Linux OS
        self._main_canvas.unbind_all("<Button-4>")
        self._main_canvas.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        """Vertically scroll through the canvas by an amount derived from the given <MouseWheel> event.

        :param event: the <MouseWheel> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("Scrolling FEATURE SELECTION TAB........................")
        if self._OS == 'Linux':
            if event.num == 4:
                self._main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._main_canvas.yview_scroll(1, "units")
        else:
            self._main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
