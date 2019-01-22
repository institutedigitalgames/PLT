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

import copy
import platform
import numpy as np
import tkinter as tk
from tkinter import ttk, font, messagebox
import os

from pyplt.gui.util import colours
from pyplt.gui.experiment.dataset.preview import DataSetPreviewFrame
from pyplt.gui.util.tab_locking import LockableTab
from pyplt import experiment as exp, ROOT_PATH
from pyplt.exceptions import NormalizationValueError
from pyplt.util.enums import NormalizationType, ParamType


class PreProcessingTab(LockableTab):
    """GUI tab for the data pre-processing stage of setting up an experiment.

    Extends the class :class:`pyplt.gui.util.tab_locking.LockableTab` which, in turn, extends the
    `tkinter.Frame` class.
    """

    def __init__(self, parent, parent_window, files_tab):
        """Initializes the `PreProcessingTab` widget.

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
        self.pack_configure(fill=tk.BOTH, expand=True)

    def get_normal_frame(self):
        """Return a `PreProcessingFrame` widget for when the tab is in the 'unlocked' state.

        The `PreProcessingFrame` widget is instantiated only once on the first occasion that the tab is 'unlocked'.

        :return: the `PreProcessingFrame` widget that is visible whenever the tab is in the 'unlocked' state.
        :rtype: :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingFrame`
        """
        if self._frame is None:
            self._frame = PreProcessingFrame(self.get_base_frame(), self._parent_window, self._files_tab)
            self._frame.grid_configure(sticky='nsew')
        return self._frame

    def unlock(self):
        """Override method in parent class to re-initialize the tab contents each time it switches to the 'unlocked' state.

        This is done to ensure the tab contents reflect the most recently loaded data set and is carried out by
        calling the `init_tab()` method of the
        :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingFrame` class.
        """
        print("unlocking preproctab...")
        super().unlock()
        # print("still unlocking...")
        # Create the contents of the tab a new
        self._frame.init_tab()
        # print("done unlocking.")

    def lock(self):
        """Override method in parent class to destroy the tab contents each time it switches to the 'locked' state.

        This is carried out by calling the `destroy_tab()` method of the
        :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingFrame` class.
        """
        print("locking preproctab...")
        super().lock()
        # print("still locking...")
        # Delete the contents of the tab (so that it's created a new next time)
        self._frame.destroy_tab()
        # print("done locking.")

    def refresh(self):
        """Destroy and re-initialize the tab contents.

        This is done by subsequent calls to the `destroy_tab()` and `init_tab()` methods of the
        :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingFrame` class.
        """
        # print("REFRESHING preproctab...")
        # refresh data! redraw canvas!
        self._frame.destroy_tab()
        self._frame.init_tab()

    def get_include_settings(self):
        """Get the user settings for each feature indicating whether or not it is to be included in the experiment.

        :return: a dict containing the feature names as the dict's keys and booleans indicating whether the
            corresponding feature is to be included in (True) or excluded from (False) the experiment as the
            dict's values.
        :rtype: dict of bool
        """
        return self._frame.get_include_settings()

    def get_norm_settings(self):
        """Get the user settings for each feature indicating how it is to be normalized.

        :return: a dict containing the feature names as the dict's keys and enumerated constants of type
            :class:`pyplt.util.enums.NormalizationType` indicating how the corresponding feature is to be
            normalized as the dict's values.
        :rtype: dict of :class:`pyplt.util.enums.NormalizationType`
        """
        return self._frame.get_norm_settings()

    def get_shuffle_settings(self):
        """Get the settings chosen by the user with respect to shuffling the dataset.

        :return:
            * shuffle -- specifies whether or not to shuffle the dataset at the start of the experiment execution.
            * random_seed -- optional seed used to shuffle the dataset.
        :rtype:
            * shuffle -- bool
            * random_seed -- int or None
        """
        return self._frame.get_shuffle_settings()


class PreProcessingFrame(tk.Frame):
    """Frame widget that is visible whenever the `Data Pre-Processing tab` is in the 'unlocked' state.

    Extends the class `tkinter.Frame`.
    """

    _preview_canvas = None
    _preview_frame = None

    def _on_canvas_config(self, event):
        """Update the canvas `scrollregion` to account for its entire bounding box.

        This method is bound to all <Configure> events with respect to :attr:`self._preview_frame`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("__ on_config called __")
        self._preview_canvas.configure(scrollregion=self._preview_canvas.bbox("all"))

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
        self._files_tab = files_tab
        self._data = None
        self._original_objects = None
        self._ghost_objects = None
        self._prev = None
        self._features = None
        self._include_settings = {}
        self._norm_settings = {}  # dictionary mapping features to the corresponding normalization method chosen
        self._include_all = tk.BooleanVar(value=True)
        self._normalize_all = tk.StringVar(value=NormalizationType.NONE.name)
        self._excluded_feats = None  # initialized to numpy array in init_tab()
        self._shuffle = tk.BooleanVar(value=False)
        self._random_seed = tk.StringVar(value="")

        self._OS = platform.system()

        tk.Frame.__init__(self, parent)

        main_frame = tk.Frame(self)
        main_frame.pack()

        self._settings_area = tk.Frame(main_frame, bd=2, relief='groove', bg=colours.PREPROC_BACK)
        # self._settings_area.grid(row=0, column=0, sticky='nsew', padx=(20, 10), pady=20)
        self._settings_area.grid(row=0, column=0, pady=(20, 5), sticky='ew')  # , fill=tk.BOTH, expand=True
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=2)

        self._frame_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/load_ranks_128_30_01.png"))

        tk.Label(self._settings_area, text="Pre-processing Settings", font=font.Font(family='Ebrima', size=12),
                 bg=colours.PREPROC_BACK).pack(pady=5)
        self._settings_pane = tk.Frame(self._settings_area, bg=colours.PREPROC_BACK)
        self._settings_pane.pack(fill=tk.BOTH, expand=True, padx=10)

        # Include all / Exclude all | Normalize all
        all_frame = tk.Frame(self._settings_area, bg=colours.PREPROC_FRONT)
        all_frame.pack(fill=tk.X, pady=(10, 0))
        include_frame = tk.Frame(all_frame, bg=colours.PREPROC_FRONT)
        include_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(include_frame, text="Include/exclude all", bg=colours.PREPROC_FRONT).grid(row=0, column=0)
        include_all_checkbtn = ttk.Checkbutton(include_frame, variable=self._include_all, onvalue=True, offvalue=False,
                                               style="Blue.PLT.TCheckbutton", command=self._toggle_include_all)
        include_all_checkbtn.grid(row=0, column=1)

        normalization_frame = tk.Frame(all_frame, bg=colours.PREPROC_FRONT)
        normalization_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        options = [NormalizationType.NONE.name,
                   NormalizationType.MIN_MAX.name,
                   NormalizationType.Z_SCORE.name]
        tk.Label(normalization_frame, text="Normalize all", bg=colours.PREPROC_FRONT).grid(row=0, column=0, padx=5)
        normalize_all_menu = ttk.OptionMenu(normalization_frame, self._normalize_all, options[0], *options,
                                            command=lambda _: self._toggle_normalize_all(),
                                            style='Blue.PLT.TMenubutton')
        normalize_all_menu.grid(row=0, column=1)

        shuffle_menu = tk.Frame(main_frame, bd=2, relief='groove', bg=colours.PREPROC_BACK, padx=15, pady=5)
        shuffle_menu.grid(row=1, column=0, pady=(5, 20), sticky='ew')

        # tk.Label(shuffle_menu, text="Shuffle Dataset?", font=font.Font(family='Ebrima', size=12),
        #          bg=colours.PREPROC_BACK).pack(pady=5)
        shuffle_sub_menu = tk.Frame(shuffle_menu, bg=colours.PREPROC_BACK)
        shuffle_sub_menu.pack()

        tk.Label(shuffle_sub_menu, text="Shuffle dataset?").grid(row=0, column=0)
        ttk.Checkbutton(shuffle_sub_menu, variable=self._shuffle, onvalue=True, offvalue=False,
                        command=self._toggle_seed_menu,
                        style="PLT.TCheckbutton").grid(row=0, column=1, padx=(10, 0))

        self._seed_menu = tk.Frame(shuffle_menu)
        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        tk.Label(self._seed_menu, text="Random seed (optional): ", bg=colours.PL_INNER).grid(row=0, column=0,
                                                                                             sticky='w',
                                                                                             padx=(0, 10), pady=5)
        entry = ttk.Entry(self._seed_menu, width=7, textvariable=self._random_seed,
                          validate="all", validatecommand=self._vcmd + (ParamType.INT.name,), style='PL.PLT.TEntry')
        entry.grid(row=0, column=1, pady=5)

        self.grid_columnconfigure(0, weight=1)  # make settings_area and shuffle_menu equal in width

    def _toggle_seed_menu(self):
        """Show or hide shuffle submenu for inputting random seed parameter."""
        if self._shuffle.get():
            self._seed_menu.pack()
            # # make it appear properly
            # self.update_idletasks()
            # self._on_resize_fn(None)
        else:
            self._seed_menu.pack_forget()

    def _toggle_include_all(self):
        """Toggle the include/exclude setting on or off for all features at one go.

        The method is called whenever the 'Include all' checkbox is toggled on or off. If the checkbox is toggled on,
        all features are included; otherwise they are all excluded (and each of their corresponding checkboxes in the
        GUI are updated accordingly).
        """
        f_id = 0
        for f_name in self._include_settings:
            if f_id >= 1:
                self._include_settings[f_name].set(self._include_all.get())
            f_id += 1

    def _toggle_normalize_all(self):
        """Set the normalization to the given method for all features at one go.

        The method is called whenever a normalization method is selected via the 'Normalize all' drop-down menu.
        All features are then normalized using this method (and each of their corresponding drop-down menus in the
        GUI are updated accordingly).
        """
        sel = self._normalize_all.get()
        for f_name in self._norm_settings:
            self._norm_settings[f_name].set(sel)

    def destroy_tab(self):
        """Destroy the contents of the tab."""
        # print("Destroyin' preproc tab")
        self._data = None
        self._features = None
        self._include_settings = {}
        self._norm_settings = {}
        self._preview_frame = None
        self._preview_canvas = None
        # destroy any remaining children e.g. scrollbars!
        for child in self._settings_pane.winfo_children():
            child.destroy()

    def init_tab(self):
        """Initialize (or re-initialize) the contents of the tab."""
        # print("init_tab called (preproc tab)")
        if self._preview_canvas is None:
            data = self._files_tab.get_data()
            if isinstance(data, tuple):
                self._data = data[0].copy()  # objects only
            else:
                self._data = data.copy()  # single file
                self._data = self._data.iloc[:, :-1]  # ignore the ratings column!
            self._features = self._data.columns

            self._preview_canvas = tk.Canvas(self._settings_pane, bg=colours.CANVAS_BACK)
            self._preview_canvas.bind('<Enter>', self._bind_mousewheel)
            self._preview_canvas.bind('<Leave>', self._unbind_mousewheel)

            # set scrollbars
            v_scroll = ttk.Scrollbar(self._settings_pane, orient="vertical", command=self._preview_canvas.yview,
                                     style="Yellow.PLT.Vertical.TScrollbar")
            v_scroll.pack(side='right', fill='y')
            self._preview_canvas.configure(yscrollcommand=v_scroll.set)
            h_scroll = ttk.Scrollbar(self._settings_pane, orient="horizontal", command=self._preview_canvas.xview,
                                     style="Yellow.PLT.Horizontal.TScrollbar")
            h_scroll.pack(side='bottom', fill='x')
            self._preview_canvas.configure(xscrollcommand=h_scroll.set)

            self._redraw_settings(self._features)

            self._preview_canvas.pack(side='left', expand=True, fill=tk.BOTH)

    def _bind_mousewheel(self, event):
        """Bind all mouse wheel events with respect to the canvas to a canvas-scrolling function.

        This method is called whenever an <Enter> event occurs with respect to :attr:`self._preview_canvas`.

        :param event: the <Enter> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # for Windows OS and MacOS
        self._preview_canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        # for Linux OS
        self._preview_canvas.bind_all("<Button-4>", self._on_mouse_scroll)
        self._preview_canvas.bind_all("<Button-5>", self._on_mouse_scroll)

    def _unbind_mousewheel(self, event):
        """Unbind all mouse wheel events with respect to the canvas from any function.

        This method is called whenever a <Leave> event occurs with respect to :attr:`self._preview_canvas`.

        :param event: the <Leave> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # for Windows OS and MacOS
        self._preview_canvas.unbind_all("<MouseWheel>")
        # for Linux OS
        self._preview_canvas.unbind_all("<Button-4>")
        self._preview_canvas.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        """Vertically scroll through the canvas by an amount derived from the given <MouseWheel> event.

        :param event: the <MouseWheel> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("Scrolling PRE-PROCESSING TAB.............................")
        if self._OS == 'Linux':
            if event.num == 4:
                self._preview_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._preview_canvas.yview_scroll(1, "units")
        else:
            self._preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _redraw_settings(self, features):
        """Draw the GUI area containing the include/exclude and normalization settings for each of the given features.

        The include/exclude settings consist of a checkbox (`ttk.Checkbutton` widgets)
        for each feature indicating whether or not the feature is to be included in the experiment. The
        normalization settings consist of a drop-down menu (`ttk.OptionMenu` widgets)
        for each feature indicating how it is to be normalized.

        :param features: the features extracted from the objects data being pre-processed.
        :type features: list of str
        """
        # print("redraw settings...")
        self._include_settings = {}
        self._norm_settings = {}

        # Initialize settings for all features (include all with no normalization)
        for feat in self._features:
            self._include_settings[feat] = tk.BooleanVar(value=True)
            self._norm_settings[feat] = tk.StringVar(value=NormalizationType.NONE.name)
            # n.b. can then go back to enum via NormalizationType[given str name]

        # Destroy old stuff first
        if self._preview_frame is not None:
            # print("Destroying frame and canvas...")
            self._preview_frame.destroy()
            self._preview_canvas.delete("all")

        # Now redraw
        self._preview_frame = tk.Frame(self._preview_canvas, relief='groove', bg=colours.CANVAS_BACK)

        self._preview_frame.bind('<Configure>', self._on_canvas_config)

        self._preview_canvas.create_window((0, 0), window=self._preview_frame, anchor='nw')

        r = 0
        cn = 0
        columns = ['Include?', 'Feature', 'Normalization']
        for col_name in columns:
            item = tk.Label(self._preview_frame, text=col_name, background=colours.PREVIEW_DEFAULT, fg='white',
                            borderwidth=2, relief='raised')
            if len(str(col_name)) < 10:
                item.configure(width=10)
            item.grid(row=r, column=cn, sticky='nsew')
            cn += 1
        r += 1
        f = 1
        for feat in features[1:]:  # V.Imp.: excluding first feature (i.e., ID)!
            c = 0
            for col in columns:
                checkbtn = None
                optmenu = None
                if col == 'Include?':
                    item = tk.Frame(self._preview_frame, borderwidth=1, relief='groove')
                    checkbtn = ttk.Checkbutton(item, variable=self._include_settings[feat], onvalue=True,
                                               offvalue=False)
                    checkbtn.pack()
                    # item.invoke()  # True by default
                elif col == 'Feature':
                    item = tk.Label(self._preview_frame, text=str(feat), borderwidth=1, relief='groove')
                else:
                    options = [NormalizationType.NONE.name,
                               NormalizationType.MIN_MAX.name,
                               NormalizationType.Z_SCORE.name]
                    item = tk.Frame(self._preview_frame, borderwidth=1, relief='groove')
                    optmenu = ttk.OptionMenu(item, self._norm_settings[feat], options[0], *options)
                    optmenu.pack(fill=tk.BOTH, expand=True)
                if r % 2 == 0:
                    if checkbtn is not None:
                        checkbtn.configure(style="Gray.PLT.TCheckbutton")
                    elif optmenu is not None:
                        optmenu.configure(style="Gray.PLT.TMenubutton")
                    item.configure(background='#f2f2f2')
                else:
                    if checkbtn is not None:
                        checkbtn.configure(style="White.PLT.TCheckbutton")
                    elif optmenu is not None:
                        optmenu.configure(style="White.PLT.TMenubutton")
                    item.configure(background='white')
                item.grid(row=r, column=c, sticky='nsew')
                c += 1
            r += 1
            f += 1

        for c in range(len(columns)):
            self._preview_frame.grid_columnconfigure(c, weight=1, uniform=columns[c])

    def get_include_settings(self):
        """Get the current include/exclude settings for each feature in the original objects data.

        :return: a dict containing the feature names as the dict's keys and booleans indicating whether the
            corresponding feature is to be included in (True) or excluded from (False) the experiment as
            the dict's values.
        :rtype: dict of bool
        """
        return self._include_settings

    def get_norm_settings(self):
        """Get the normalization settings for each feature in the original objects data.

        :return: a dict containing the feature names as the dict's keys and enumerated constants of type
            :class:`pyplt.util.enums.NormalizationType` indicating how the corresponding feature is to be normalized
            as the dict's values.
        :rtype: dict of :class:`pyplt.util.enums.NormalizationType`
        """
        return self._norm_settings

    def get_shuffle_settings(self):
        """Get the settings chosen by the user with respect to shuffling the dataset.

        :return:
            * shuffle -- specifies whether or not to shuffle the dataset at the start of the experiment execution.
            * random_seed -- optional seed used to shuffle the dataset.
        :rtype:
            * shuffle -- bool
            * random_seed -- int or None
        """
        shuffle = self._shuffle.get()
        random_seed = self._random_seed.get()
        if random_seed == "":
            random_seed = None
        else:
            random_seed = int(random_seed)
        return shuffle, random_seed

    def _on_validate(self, action, index, value_if_allowed,
                     prior_value, text, validation_type, trigger_type, widget_name, val_type):
        """Validate the text input by the user in a `ttk.Entry` widget.

        All but the last of the method arguments refer to callback substitution codes describing the type of
        change made to the text of the `ttk.Entry widget`. Their description is based on existing documentation at
        http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/entry-validation.html.

        :param action: Action code: 0 for an attempted deletion, 1 for an attempted insertion, or -1 if the
            callback was called for focus in, focus out, or a change to the textvariable.
        :param index: When the user attempts to insert or delete text, this argument will be the index of the
            beginning of the insertion or deletion. If the callback was due to focus in, focus out, or a change to
            the textvariable, the argument will be -1.
        :param value_if_allowed: The value that the text will have if the change is allowed.
        :param prior_value: The text in the entry before the change.
        :param text: If the call was due to an insertion or deletion, this argument will be the text being
            inserted or deleted.
        :param validation_type: The current value of the widget's validate option.
        :param trigger_type: The reason for this callback: one of 'focusin', 'focusout', 'key', or 'forced' if
            the textvariable was changed.
        :param widget_name: The name of the widget.
        :param val_type: specifies the type of validation to carry out on the value/s entered by the user in the
            text entry widget for the given parameter.
        :type val_type: :class:`pyplt.util.enums.ParamType`
        :return: a boolean specifying whether the text is valid (True) or not (False).
        :rtype: bool
        """
        # print("Validating... " + val_type)
        # print(prior_value)
        # print(text)
        # print(value_if_allowed)
        if ParamType[val_type] == ParamType.FLOAT:
            try:
                float(text)
                return True
            except ValueError:
                try:
                    float(value_if_allowed)
                    return True
                except ValueError:
                    return False
        else:
            if text.isnumeric():
                return True
            else:
                return False
