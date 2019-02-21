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

from pyplt.gui.experiment.preprocessing.data_compression import AutoencoderSettings
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

    def get_autoencoder_menu(self):
        """Get the autoencoder GUI menu widget through which the parameter values selected by the user may be read.

        :return: the autoencoder menu widget.
        :rtype: `:class:pyplt.gui.experiment.preprocessing.data_compression.AutoencoderSettings`
        """
        return self._frame.get_autoencoder_menu()

    def auto_extract_enabled(self):
        """Indicate whether or not automatic feature selection (via autoencoder) has been chosen.

        :return: specifies whether or not automatic feature selection (via autoencder) was chosen.
        :rtype: bool
        """
        return self._frame.auto_extract_enabled()


class PreProcessingFrame(tk.Frame):
    """Frame widget that is visible whenever the `Data Pre-Processing tab` is in the 'unlocked' state.

    Extends the class `tkinter.Frame`.
    """

    _settings_canvas = None
    _settings_frame = None
    _autoencoder_menu = None

    def _on_settings_canvas_config(self, event):
        """Update the canvas `scrollregion` to account for its entire bounding box.

        This method is bound to all <Configure> events with respect to :attr:`self._settings_frame`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("__ on_config called __")
        self._settings_canvas.configure(scrollregion=self._settings_canvas.bbox("all"))

    def _on_main_canvas_config(self, event):
        """Update the canvas `scrollregion` to account for the entire area of the :attr:`self._main_sub_frame` widget.

        This method is bound to all <Configure> events with respect to :attr:`self._main_sub_frame`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("__ on_config called __")
        self._main_canvas.configure(scrollregion=(0, 0, self._main_sub_frame.winfo_reqwidth(),
                                                  self._main_sub_frame.winfo_reqheight()))

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

        self._settings_v_scroll = None

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

        # menu for choosing between manual and automatic feature selection
        self._features_area = tk.Frame(self._main_sub_sub_frame)
        self._features_area.grid(row=0, column=0, pady=(20, 15), sticky='ew')

        extract_frame = tk.Frame(self._features_area, padx=25)
        extract_frame.grid(row=0, column=1)

        self._extract = tk.BooleanVar()
        self._extract.set(False)  # false by default

        extract_label = tk.Label(extract_frame, text="Feature Extraction")
        extract_label.grid(row=0, column=0, sticky='w')
        extract_manual = ttk.Radiobutton(extract_frame, variable=self._extract,
                                         # Use the columns in the dataset as features
                                         # Use the pre-extracted features included in the dataset
                                         text="Use the features included in the dataset",
                                         value=False,
                                         command=self._toggle_autoencoder_menu,
                                         style='PLT.TRadiobutton')
        extract_manual.grid(row=1, column=0, sticky='w')
        extract_auto = ttk.Radiobutton(extract_frame, variable=self._extract,
                                       text="Extract features automatically (via an autoencoder)",
                                       value=True,
                                       command=self._toggle_autoencoder_menu,
                                       style='PLT.TRadiobutton')
        extract_auto.grid(row=2, column=0, sticky='w')

        self._settings_area = tk.Frame(self._main_sub_sub_frame, bd=2, relief='groove', bg=colours.PREPROC_BACK)
        # self._settings_area.grid(row=0, column=0, sticky='nsew', padx=(20, 10), pady=20)
        self._settings_area.grid(row=2, column=0, pady=(10, 5), sticky='ew')  # , fill=tk.BOTH, expand=True
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

        shuffle_menu = tk.Frame(self._main_sub_sub_frame, bd=2, relief='groove', bg=colours.PREPROC_BACK,
                                padx=15, pady=5)
        shuffle_menu.grid(row=3, column=0, pady=(5, 20), sticky='ew')

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

        # self.grid_columnconfigure(0, weight=1)  # make settings_area and shuffle_menu equal in width

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

        self._main_sub_frame.bind('<Configure>', self._on_main_canvas_config)

        # self._main_sub_sub_frame.grid_rowconfigure(0, weight=0)
        # self._main_sub_sub_frame.grid_rowconfigure(1, weight=1)
        # self._main_sub_sub_frame.grid_rowconfigure(2, weight=0)

    def _toggle_autoencoder_menu(self):
        """Show or hide autoencoder menu for feature extraction."""
        if self._extract.get():
            if self._autoencoder_menu is None:
                self._autoencoder_menu = AutoencoderSettings(self._main_sub_sub_frame, len(self._features)-1,
                                                             self._on_resize)
                # ^ -1 to exclude ID column
                self._autoencoder_menu.grid(row=1, column=0, sticky='ns')
                # call self._redraw_settings to update features each time code size param value changes!
                self._autoencoder_menu.trace_code_size(self._redraw_settings)
            else:
                self._autoencoder_menu.grid(row=1, column=0, sticky='ns')
            # force updates for canvas and scrollbar stuff
            self.update_idletasks()
            self._on_resize(None)
            self._on_main_canvas_config(None)
        else:
            if self._autoencoder_menu is not None:
                self._autoencoder_menu.grid_forget()
        # call self._redraw_settings to update features each time automatic feature extraction
        # via autoencoder is toggled on/off!
        self._redraw_settings()

    def _toggle_seed_menu(self):
        """Show or hide shuffle submenu for inputting random seed parameter."""
        if self._shuffle.get():
            self._seed_menu.pack()
            # force updates for canvas and scrollbar stuff
            self.update_idletasks()
            self._on_resize(None)
            self._on_main_canvas_config(None)
        else:
            self._seed_menu.pack_forget()

    def _toggle_include_all(self):
        """Toggle the include/exclude setting on or off for all features at one go.

        The method is called whenever the 'Include all' checkbox is toggled on or off. If the checkbox is toggled on,
        all features are included; otherwise they are all excluded (and each of their corresponding checkboxes in the
        GUI are updated accordingly).
        """
        if self._extract.get():
            # if autoencoder is used, keep all features included
            # keep on (True) all the time via binding
            self._include_all.set(True)
        else:
            for f_id in self._include_settings:
                self._include_settings[f_id].set(self._include_all.get())

    def _toggle_normalize_all(self):
        """Set the normalization to the given method for all features at one go.

        The method is called whenever a normalization method is selected via the 'Normalize all' drop-down menu.
        All features are then normalized using this method (and each of their corresponding drop-down menus in the
        GUI are updated accordingly).
        """
        sel = self._normalize_all.get()
        for f_id in self._norm_settings:
            self._norm_settings[f_id].set(sel)

    def destroy_tab(self):
        """Destroy the contents of the tab."""
        # print("Destroyin' preproc tab")
        self._data = None
        self._features = None
        self._include_settings = {}
        self._norm_settings = {}
        self._settings_frame = None
        self._settings_canvas = None
        # destroy any remaining children e.g. scrollbars!
        for child in self._settings_pane.winfo_children():
            child.destroy()

    def init_tab(self):
        """Initialize (or re-initialize) the contents of the tab."""
        # print("init_tab called (preproc tab)")
        if self._settings_canvas is None:
            data = self._files_tab.get_data()
            if isinstance(data, tuple):
                self._data = data[0].copy()  # objects only
            else:
                self._data = data.copy()  # single file
                self._data = self._data.iloc[:, :-1]  # ignore the ratings column!
            self._features = self._data.columns

            self._settings_canvas = tk.Canvas(self._settings_pane, bg=colours.CANVAS_BACK)
            # self._settings_canvas.bind('<Enter>', self._bind_mousewheel_settings)
            # self._settings_canvas.bind('<Leave>', self._unbind_mousewheel_settings)

            # set scrollbars
            self._settings_v_scroll = ttk.Scrollbar(self._settings_pane, orient="vertical",
                                                    command=self._settings_canvas.yview,
                                                    style="Yellow.PLT.Vertical.TScrollbar")
            self._settings_v_scroll.pack(side='right', fill='y')
            self._settings_canvas.configure(yscrollcommand=self._settings_v_scroll.set)
            h_scroll = ttk.Scrollbar(self._settings_pane, orient="horizontal", command=self._settings_canvas.xview,
                                     style="Yellow.PLT.Horizontal.TScrollbar")
            h_scroll.pack(side='bottom', fill='x')
            self._settings_canvas.configure(xscrollcommand=h_scroll.set)

            # reset input_size parameter of autoencoder GUI menu according to potentially new data!
            if self._extract.get() and (self._autoencoder_menu is not None):
                self._autoencoder_menu.set_input_size(len(self._features)-1)
                # ^ -1 to exclude ID column

            self._redraw_settings()

            self._settings_canvas.pack(side='left', expand=True, fill=tk.BOTH)

            # force updates for canvas and scrollbar stuff
            self.update_idletasks()
            self._on_resize(None)
            self._on_main_canvas_config(None)

    def _redraw_settings(self, *args):
        """Draw the GUI area containing the include/exclude and normalization settings for each of the given features.

        The include/exclude settings consist of a checkbox (`ttk.Checkbutton` widgets)
        for each feature indicating whether or not the feature is to be included in the experiment. The
        normalization settings consist of a drop-down menu (`ttk.OptionMenu` widgets)
        for each feature indicating how it is to be normalized.

        :param args: additional arguments for when the method is called as a callback function
            via the tk.IntVar.trace method.
        """
        # print("redraw settings...")
        self._include_settings = {}
        self._norm_settings = {}

        # Initialize settings for all features (include all with no normalization)
        if self._extract.get():  # if using automatic feature extraction via autoencoder
            code_size = self._autoencoder_menu.get_code_size()
            n_feats = code_size
            feat_names = ["ExtractedFeature" + str(f+1) for f in range(n_feats)]
        else:  # if using predetermined features
            n_feats = len(self._features)-1  # -1 to exclude ID column
            feat_names = self._features[1:]  # V.Imp.: excluding first feature (i.e., ID)!
        self._include_settings = {f: tk.BooleanVar(value=True) for f in range(n_feats)}
        self._norm_settings = {f: tk.StringVar(value=NormalizationType.NONE.name) for f in range(n_feats)}
        # n.b. can then go back to enum via NormalizationType[given str name]

        # Destroy old stuff first
        if self._settings_frame is not None:
            # print("Destroying frame and canvas...")
            self._settings_frame.destroy()
            self._settings_canvas.delete("all")

        # Now redraw
        self._settings_frame = tk.Frame(self._settings_canvas, relief='groove', bg=colours.CANVAS_BACK)

        self._settings_frame.bind('<Configure>', self._on_settings_canvas_config)

        self._settings_canvas.create_window((0, 0), window=self._settings_frame, anchor='nw')

        r = 0
        cn = 0
        columns = ['Include?', 'Feature', 'Normalization']
        for col_name in columns:
            item = tk.Label(self._settings_frame, text=col_name, background=colours.PREVIEW_DEFAULT, fg='white',
                            borderwidth=2, relief='raised')
            if len(str(col_name)) < 10:
                item.configure(width=10)
            item.grid(row=r, column=cn, sticky='nsew')
            cn += 1
        r += 1
        for f in range(n_feats):
            feat = feat_names[f]
            c = 0
            for col in columns:
                checkbtn = None
                optmenu = None
                if col == 'Include?':
                    item = tk.Frame(self._settings_frame, borderwidth=1, relief='groove')
                    checkbtn = ttk.Checkbutton(item, variable=self._include_settings[f], onvalue=True,
                                               offvalue=False)
                    checkbtn.pack()
                    # item.invoke()  # True by default
                    if self._extract.get():
                        # if autoencoder is used, keep all features included
                        # keep on (True) all the time via binding
                        checkbtn.configure(command=lambda f_id=f: self._disable_include_settings(f_id))
                        # ^ since we use bind, it will remain disabled after closing Help so no need for extra method
                elif col == 'Feature':
                    item = tk.Label(self._settings_frame, text=str(feat), borderwidth=1, relief='groove', padx=5)
                else:
                    options = [NormalizationType.NONE.name,
                               NormalizationType.MIN_MAX.name,
                               NormalizationType.Z_SCORE.name]
                    item = tk.Frame(self._settings_frame, borderwidth=1, relief='groove')
                    optmenu = ttk.OptionMenu(item, self._norm_settings[f], options[0], *options)
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

        for c in range(len(columns)):
            self._settings_frame.grid_columnconfigure(c, weight=1, uniform=columns[c])

    def _disable_include_settings(self, f_id):
        """Keep the include/exclude checkbutton widget of a given feature on when autoencoder is enabled.

        :param f_id: the index of the feature to keep included.
        """
        self._include_settings[f_id].set(True)  # just always keep true

    def get_include_settings(self):
        """Get the current include/exclude settings for each feature in the original objects data.

        :return: a dict containing the feature indices as the dict's keys and booleans indicating whether the
            corresponding feature is to be included in (True) or excluded from (False) the experiment as
            the dict's values.
        :rtype: dict of bool
        """
        return self._include_settings

    def get_norm_settings(self):
        """Get the normalization settings for each feature in the original objects data.

        :return: a dict containing the feature indices as the dict's keys and enumerated constants of type
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
        # differentiate between self._settings_canvas and and the rest
        # if mouse is on self._settings_canvas or its scrollbar, scroll w.r.t. the self._settings_canvas
        # otherwise scroll w.r.t. the whole window/canvas (i.e., self._main_canvas)
        x, y = self.winfo_pointerxy()
        w = self.winfo_containing(x, y)

        if w == self._settings_canvas or w == self._settings_v_scroll:
            widget = self._settings_canvas
        else:
            widget = self._main_canvas

        # print("Scrolling FEATURE SELECTION TAB........................")
        if self._OS == 'Linux':
            if event.num == 4:
                widget.yview_scroll(-1, "units")
            elif event.num == 5:
                widget.yview_scroll(1, "units")
        else:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def get_autoencoder_menu(self):
        """Get the autoencoder GUI menu widget through which the parameter values selected by the user may be read.

        :return: the autoencoder menu widget.
        :rtype: `:class:pyplt.gui.experiment.preprocessing.data_compression.AutoencoderSettings`
        """
        return self._autoencoder_menu

    def auto_extract_enabled(self):
        """Indicate whether or not automatic feature selection (via autoencoder) has been chosen.

        :return: specifies whether or not automatic feature selection (via autoencder) was chosen.
        :rtype: bool
        """
        return self._extract.get()
