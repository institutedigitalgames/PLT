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
from tkinter import font
from enum import Enum
import os

from pyplt import ROOT_PATH
from pyplt import experiment as exp
from pyplt.gui.util.help import RankDerivationHelpDialog
from pyplt.util.enums import FileType, ParamType
from pyplt.gui.experiment.dataset.preview import DataSetPreviewFrame
from pyplt.gui.util import windowstacking as ws, colours


class Confirmation(Enum):
    """Class specifying enumerated constants for loading confirmation states.

    Extends `enum.Enum`.
    """
    CANCEL = 0
    DONE = 1


class LoadingParamsWindow(tk.Toplevel):
    """GUI window for specifying the parameters for loading data from a file.

     Extends the class `tkinter.Toplevel`.
    """

    def __init__(self, parent, file_path, parent_window, file_type):
        """Populates the window widget with widgets to specify loading parameters and a data set preview frame.

        :param parent: the parent widget of this window widget.
        :type parent: `tkinter widget`
        :param file_path: the path of the file to be loaded.
        :type file_path: str
        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        :param file_type: the type of file to be loaded.
        :type file_type: :class:`pyplt.util.enums.FileType`.
        """
        self._prev = None
        self._parent = parent
        self._separator = None
        self._has_id = None
        self._has_fnames = None
        self._preview_frame = None
        self._file_path = file_path
        self._objects_confirmation = None
        self._ranks_confirmation = None
        self._single_confirmation = None
        self._parent_window = parent_window
        self._file_type = file_type
        self._data = None
        self._mdm = tk.DoubleVar(value=0.0)
        self._memory = tk.IntVar()
        self._memory_str = tk.StringVar()

        tk.Toplevel.__init__(self, parent)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, self._parent_window, ws.Mode.OPEN_ONLY)
        ws.stack_window(self, self._parent_window)

        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        self.title("Processing " + str(self._file_type.name).lower() + " file")
        self.resizable(width=False, height=False)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        setup_frame = tk.Frame(self, relief='groove', borderwidth=2)
        setup_frame.pack(pady=15, padx=20)
        params_frame = tk.Frame(setup_frame)
        params_frame.pack()

        # Separator
        self._separator = tk.StringVar()
        self._separator.set(',')  # comma by default
        sep_frame = tk.Frame(params_frame, padx=25)
        sep_frame.grid(row=0, column=0, sticky='nw')
        sep_label = tk.Label(sep_frame, text="Separator")
        sep_label.grid(row=0, column=0, sticky='w')
        sep_opt1 = ttk.Radiobutton(sep_frame, variable=self._separator, text="Comma", value=',',
                                   command=self._update_data, style='PLT.TRadiobutton')
        sep_opt1.grid(row=1, column=0, sticky='w')
        sep_opt2 = ttk.Radiobutton(sep_frame, variable=self._separator, text="Tab", value='\t',
                                   command=self._update_data, style='PLT.TRadiobutton')
        sep_opt2.grid(row=2, column=0, sticky='w')
        sep_opt3 = ttk.Radiobutton(sep_frame, variable=self._separator, text="Space", value=' ',
                                   command=self._update_data, style='PLT.TRadiobutton')
        sep_opt3.grid(row=3, column=0, sticky='w')

        # TODO: add 'Other' option which toggles textbox which returns its value to self.objects_separator
        # sep_opt4 = ttk.Radiobutton(sep_frame, command=toggle_text_box, text="Other")

        # Check if 1st column contains IDs and if 1st row contains Feature Names
        self._has_id = tk.BooleanVar()
        self._has_id.set(True)  # true by default
        self._has_fnames = tk.BooleanVar()
        self._has_fnames.set(True)  # true by default

        other_frame = tk.Frame(params_frame, padx=25)
        other_frame.grid(row=0, column=1)
        # Check ID
        has_id_label = tk.Label(other_frame, text="Does first column contain ID?")
        has_id_label.grid(row=0, column=0, sticky='w')
        has_id_opt1 = ttk.Radiobutton(other_frame, variable=self._has_id, text="Yes", value=True,
                                      command=self._update_data, style='PLT.TRadiobutton')
        has_id_opt1.grid(row=1, column=0, sticky='w')
        has_id_opt2 = ttk.Radiobutton(other_frame, variable=self._has_id, text="No", value=False,
                                      command=self._update_data, style='PLT.TRadiobutton')
        has_id_opt2.grid(row=2, column=0, sticky='w')

        # Check Feature Names
        fnames_text = "Does first row contain feature names?"
        if self._file_type == FileType.RANKS:
            fnames_text = "Does first row contain column names?"

        has_fnames_label = tk.Label(other_frame, text=fnames_text)
        has_fnames_label.grid(row=3, column=0, sticky='w', pady=(15, 0))
        has_fnames_opt1 = ttk.Radiobutton(other_frame, variable=self._has_fnames, text="Yes", value=True,
                                          command=self._update_data, style='PLT.TRadiobutton')
        has_fnames_opt1.grid(row=4, column=0, sticky='w')
        has_fnames_opt2 = ttk.Radiobutton(other_frame, variable=self._has_fnames, text="No", value=False,
                                          command=self._update_data, style='PLT.TRadiobutton')
        has_fnames_opt2.grid(row=5, column=0, sticky='w')

        # TODO: Add 'Skip Columns/Rows' feature

        # Data set preview
        preview_label = tk.Label(self, text="Preview", font=font.Font(family='Ebrima', size=14))
        preview_label.pack(pady=(0, 5))
        self._preview_frame = tk.Frame(self)
        self._preview_frame.pack(fill=tk.BOTH, expand=True)

        self._update_data()

        # after data has been loaded (since we need self._data here)...
        if self._file_type == FileType.SINGLE:
            single_frame = tk.Frame(setup_frame, bg=colours.PL_OUTER)
            single_frame.pack(pady=(10, 0), fill=tk.BOTH, expand=True)  # .grid(row=1, column=0, sticky='ew')
            header = tk.Frame(single_frame, bg=colours.PL_OUTER)
            header.pack(pady=(10, 0))
            single_label = tk.Label(header, text="Preference Derivation Parameters", font='Ebrima 12 normal',
                                    fg='white', bg=colours.PL_OUTER)
            single_label.pack(side=tk.LEFT, padx=5)
            self._help_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/help.png"))
            help_btn = tk.Button(header, command=self._help_dialog, image=self._help_img, relief='flat', bd=0,
                                 highlightbackground=colours.PL_OUTER, highlightcolor=colours.PL_OUTER,
                                 highlightthickness=0,
                                 background=colours.PL_OUTER, activebackground=colours.PL_OUTER)
            help_btn.pack(side=tk.RIGHT, padx=5)
            self._single_params = tk.Frame(single_frame, padx=20, pady=5, relief=tk.RAISED, bd=2)
            self._single_params.pack(side=tk.BOTTOM, pady=(10, 15))
            self._add_entry_param(self._single_params, "Minimum Distance Margin", self._mdm, 0,
                                  ParamType.FLOAT_POSITIVE.name)
            memory_label = tk.Label(self._single_params, text="Memory: ", bg=colours.BACKGROUND)
            memory_label.grid(row=1, column=0, sticky='w', padx=(0, 10), pady=5)
            memory_frame = tk.Frame(self._single_params)
            memory_frame.grid(row=1, column=1, pady=5, sticky='w')
            n_samples = self._data.shape[0]
            max_ = n_samples-1
            self._memory.set(max_)
            self._memory_str.set(max_)
            self._memory_scale = ttk.Scale(memory_frame, from_=1, to=max_, value=max_, variable=self._memory,
                                           orient=tk.HORIZONTAL, command=self._accept_int_only,
                                           style='PLT.Horizontal.TScale')
            self._memory_scale.pack(side=tk.TOP)
            tk.Label(memory_frame, text="1").pack(side=tk.LEFT)
            tk.Label(memory_frame, text=str(max_) + " (ALL)").pack(side=tk.RIGHT)
            memory_frame2 = tk.Frame(self._single_params)
            memory_frame2.grid(row=2, column=1, pady=5, sticky='ew')
            memory_entry = ttk.Entry(memory_frame2, width=7, textvariable=self._memory_str,
                                     validate="all", validatecommand=self._vcmd + ('ranged_int',),
                                     style='PLT.TEntry')
            memory_entry.pack()

        self._load_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/load_76_30_03.png"))
        self._cancel_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/cancel_76_30_04.png"))

        # Final loading stuff
        final_frame = tk.Frame(self, background=colours.NAV_BAR)
        final_frame.pack(fill=tk.BOTH)
        final_btns_frame = tk.Frame(final_frame, background=colours.NAV_BAR)
        final_btns_frame.pack(anchor=tk.CENTER)
        final_btn = tk.Button(final_btns_frame, command=self._final_load, image=self._load_img, relief='flat', bd=0,
                              highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                              background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        final_btn.grid(row=0, column=0, padx=5, pady=5)
        final_btn = tk.Button(final_btns_frame, command=self._on_closing, image=self._cancel_img, relief='flat', bd=0,
                              highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                              background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        final_btn.grid(row=0, column=1, padx=5, pady=5)

    def _update_data(self):
        """Extract the data specified by the file path and update the preview accordingly.

        The extracted data is stored in a `pandas.DataFrame`.
        """
        sep = self._separator.get()
        has_id = self._has_id.get()
        has_fnames = self._has_fnames.get()

        # finally, actually load the data according to the chosen params

        # print("separator: " + str(sep))
        # print("has_id: " + str(has_id))
        # print("has_fnames: " + str(has_fnames))
        # print("updating dataset...")

        self._data = exp._load_data(self._file_type, self._file_path, has_fnames, has_id, sep)

        if self._prev is None:
            self._prev = DataSetPreviewFrame(self._preview_frame, self._data)
        else:
            self._prev.update(self._data)

    def _final_load(self):
        """Order the window to close and set confirmation variables to indicate that the file load is complete."""
        ws.on_close(self, self._parent_window)  # self.destroy() is covered within this method
        self._prev = None
        if self._file_type == FileType.OBJECTS:
            self._objects_confirmation = Confirmation.DONE
            # print("OBJECTS FILE LOADED.")
        elif self._file_type == FileType.RANKS:
            self._ranks_confirmation = Confirmation.DONE
            # print("RANKS FILE LOADED.")
        elif self._file_type == FileType.SINGLE:
            self._single_confirmation = Confirmation.DONE
            # print("SINGLE FILE LOADED.")
        # print(self.data)

    def _on_closing(self):
        """Order the window to close and set confirmation variables to indicate that the file load was cancelled."""
        ws.on_close(self, self._parent_window)  # self.destroy() is covered within this method
        self._prev = None
        if self._file_type == FileType.OBJECTS:
            self._objects_confirmation = Confirmation.CANCEL
            print("Cancelled objects file load.")
        elif self._file_type == FileType.RANKS:
            self._ranks_confirmation = Confirmation.CANCEL
            print("Cancelled ranks file load.")
        else:  # FileType.SINGLE
            self._single_confirmation = Confirmation.CANCEL
            print("Cancelled single file load.")
        # print(self.data)

    def get_confirmation(self):
        """Get the confirmation variable indicating whether or not the file load has been completed or cancelled.

        :return: the confirmation variable for the file that was attempted to be loaded.
        :rtype: :class:`pyplt.gui.experiment.dataset.params.Confirmation`
        """
        self.wait_window()
        if self._file_type == FileType.OBJECTS:
            return self._objects_confirmation
        elif self._file_type == FileType.RANKS:
            return self._ranks_confirmation
        else:  # FileType.SINGLE
            return self._single_confirmation

    def get_data(self):
        """Get the data extracted and processed from the file being loaded.

        :return: the extracted data.
        :rtype: `pandas.DataFrame`
        """
        # N.B. splitting of single file into objects and ranks was moved from here to experiment execution
        # (see Experiment.run() method)
        return self._data

    def _add_entry_param(self, parent, name, var, row, val_type):
        """Generic method for adding parameter labels and text entries to the GUI menu.

        The method constructs a `ttk.Entry` widget preceded by a `tkinter.Label` widget displaying the
        given parameter name.

        :param parent: the parent widget of these widgets.
        :type parent: `tkinter widget`
        :param name: the name of the parameter to be used as the label of the text entry widget.
        :type name: str
        :param var: the variable in which the parameter value entered by the user is to be stored.
        :type var: `tkinter.StringVar` or `tkinter.IntVar` or `tkinter.DoubleVar` or `tkinter.BooleanVar`
        :param row: the number of the row in the parent widget's grid layout where these widgets are to be drawn.
        :type row: int
        :param val_type: specifies the type of validation to carry out on the value/s entered by the user in the
            text entry widget for the given parameter.
        :type val_type: :class:`pyplt.util.enums.ParamType`
        """
        label = tk.Label(parent, text=str(name)+": ", bg=colours.BACKGROUND)
        label.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=5)
        entry = ttk.Entry(parent, width=7, textvariable=var,
                          validate="all", validatecommand=self._vcmd + (val_type,), style='PLT.TEntry')
        entry.grid(row=row, column=1, pady=5, sticky='w')
        parent.after_idle(lambda: entry.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values
        return label, entry

    def _accept_int_only(self, event=None):
        """Convert memory Scale values to integer.

        :param event: the event that triggered this method to be called.
        :type event: `tkinter Event`
        """
        value = self._memory_scale.get()
        int_value = int(round(value))
        self._memory.set(int_value)
        self._memory_str.set(int_value)

    def _memory_entry_update(self, str_value):
        """Update the memory Scale value when the memory Entry widget value has been updated.

        The update to the Scale widget only occurs if the value entered in the Entry widget is valid.

        :param str_value: the value entered in the Entry widget.
        :type str_value: str
        """
        try:
            int_value = int(str_value)
            round_value = int(round(int_value))
            self._memory.set(round_value)
            # print("entry is " + str(str_value) + " (valid) so self._memory has been changed to: "
            #       + str(self._memory.get()))
        except ValueError:  # i.e. if str_value == ''
            # ignore, do not update anything
            # print("entry is '' but self._memory remains: " + str(self._memory.get()))
            return

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
        if val_type == 'ranged_int':  # 1 to n_samples-1
            try:
                if (((int(text) > 0) and (int(text) < self._data.shape[0]))
                    and ((int(value_if_allowed) > 0) and (int(value_if_allowed) < self._data.shape[0])))\
                        or ((int(value_if_allowed) > 0) and (int(value_if_allowed) < self._data.shape[0])):
                    self._memory_entry_update(value_if_allowed)
                    return True
            except ValueError:
                if value_if_allowed == '':
                    # print("entry is '' but self._memory is: " + str(self._memory.get()))
                    self._memory_entry_update(value_if_allowed)
                    return True
        elif ParamType[val_type] == ParamType.FLOAT:
            try:
                float(text)
                if (ParamType[val_type] == ParamType.FLOAT_POSITIVE) and (float(text) < 0.0):
                    return False
                return True
            except ValueError:
                try:
                    float(value_if_allowed)
                    if (ParamType[val_type] == ParamType.FLOAT_POSITIVE) and (float(value_if_allowed) < 0.0):
                        return False
                    return True
                except ValueError:
                    return False
        else:
            if text.isnumeric():
                return True
            else:
                return False
        return False

    def _help_dialog(self):
        """Open a help dialog window to assist the user with the rank derivation parameters."""
        RankDerivationHelpDialog(self)

    def get_rank_derivation_params(self):
        """Get the values of the rank derivation parameters (minimum distance margin and memory).

        These only apply when a single file format is used.

        :return: tuple containing the values of both parameters.
        :rtype: tuple (size 2)
        """
        mdm = self._mdm.get()
        memory = self._memory.get()
        return mdm, memory


class LoadingFoldsWindow(tk.Toplevel):
    """GUI window for specifying the parameters for loading data from a file.

     Extends the class `tkinter.Toplevel`.
    """

    def __init__(self, parent, file_path, parent_window, is_dual_format):
        """Populates the window widget with widgets to specify loading parameters and a data set preview frame.

        :param parent: the parent widget of this window widget.
        :type parent: `tkinter widget`
        :param file_path: the path of the file to be loaded.
        :type file_path: str
        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        :param is_dual_format: specifies whether the dual file format was used to load the dataset or not (single
            file format).
        :type is_dual_format: bool
        """
        self._prev = None
        self._parent = parent
        self._separator = None
        self._has_id = None
        self._has_fnames = None
        self._preview_frame = None
        self._file_path = file_path
        self._confirmation = None
        self._parent_window = parent_window
        self._data = None

        tk.Toplevel.__init__(self, parent)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, self._parent_window, ws.Mode.OPEN_ONLY)
        ws.stack_window(self, self._parent_window)

        self.title("Processing folds file")
        self.resizable(width=False, height=False)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        setup_frame = tk.Frame(self, relief='groove', borderwidth=2)
        setup_frame.pack(pady=15, padx=20)
        params_frame = tk.Frame(setup_frame)
        params_frame.pack()

        # Separator
        self._separator = tk.StringVar()
        self._separator.set(',')  # comma by default
        sep_frame = tk.Frame(params_frame, padx=25)
        sep_frame.grid(row=0, column=0, sticky='nw')
        sep_label = tk.Label(sep_frame, text="Separator")
        sep_label.grid(row=0, column=0, sticky='w')
        sep_opt1 = ttk.Radiobutton(sep_frame, variable=self._separator, text="Comma", value=',',
                                   command=self._update_data, style='PLT.TRadiobutton')
        sep_opt1.grid(row=1, column=0, sticky='w')
        sep_opt2 = ttk.Radiobutton(sep_frame, variable=self._separator, text="Tab", value='\t',
                                   command=self._update_data, style='PLT.TRadiobutton')
        sep_opt2.grid(row=2, column=0, sticky='w')
        sep_opt3 = ttk.Radiobutton(sep_frame, variable=self._separator, text="Space", value=' ',
                                   command=self._update_data, style='PLT.TRadiobutton')
        sep_opt3.grid(row=3, column=0, sticky='w')

        # TODO: add 'Other' option which toggles textbox which returns its value to self.objects_separator
        # sep_opt4 = ttk.Radiobutton(sep_frame, command=toggle_text_box, text="Other")

        # Check if 1st column contains IDs and if 1st row contains Feature Names
        self._has_id = tk.BooleanVar()
        self._has_id.set(True)  # true by default
        self._has_fnames = tk.BooleanVar()
        self._has_fnames.set(True)  # true by default

        other_frame = tk.Frame(params_frame, padx=25)
        other_frame.grid(row=0, column=1)
        # Check ID
        if is_dual_format:  # dual file format
            has_id_label = tk.Label(other_frame, text="Does first column contain rank ID?")
        else:  # single file format
            has_id_label = tk.Label(other_frame, text="Does first column contain object ID?")
        has_id_label.grid(row=0, column=0, sticky='w')
        has_id_opt1 = ttk.Radiobutton(other_frame, variable=self._has_id, text="Yes", value=True,
                                      command=self._update_data, style='PLT.TRadiobutton')
        has_id_opt1.grid(row=1, column=0, sticky='w')
        has_id_opt2 = ttk.Radiobutton(other_frame, variable=self._has_id, text="No", value=False,
                                      command=self._update_data, style='PLT.TRadiobutton')
        has_id_opt2.grid(row=2, column=0, sticky='w')

        # Check Feature Names
        fnames_text = "Does first row contain column names?"

        has_fnames_label = tk.Label(other_frame, text=fnames_text)
        has_fnames_label.grid(row=3, column=0, sticky='w', pady=(15, 0))
        has_fnames_opt1 = ttk.Radiobutton(other_frame, variable=self._has_fnames, text="Yes", value=True,
                                          command=self._update_data, style='PLT.TRadiobutton')
        has_fnames_opt1.grid(row=4, column=0, sticky='w')
        has_fnames_opt2 = ttk.Radiobutton(other_frame, variable=self._has_fnames, text="No", value=False,
                                          command=self._update_data, style='PLT.TRadiobutton')
        has_fnames_opt2.grid(row=5, column=0, sticky='w')

        # TODO: Add 'Skip Columns/Rows' feature

        # Data set preview
        preview_label = tk.Label(self, text="Preview", font=font.Font(family='Ebrima', size=14))
        preview_label.pack(pady=(0, 5))
        self._preview_frame = tk.Frame(self)
        self._preview_frame.pack(fill=tk.BOTH, expand=True)

        self._update_data()

        self._load_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/load_76_30_03.png"))
        self._cancel_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/cancel_76_30_04.png"))

        # Final loading stuff
        final_frame = tk.Frame(self, background=colours.NAV_BAR)
        final_frame.pack(fill=tk.BOTH)
        final_btns_frame = tk.Frame(final_frame, background=colours.NAV_BAR)
        final_btns_frame.pack(anchor=tk.CENTER)
        final_btn = tk.Button(final_btns_frame, command=self._final_load, image=self._load_img, relief='flat', bd=0,
                              highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                              background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        final_btn.grid(row=0, column=0, padx=5, pady=5)
        final_btn = tk.Button(final_btns_frame, command=self._on_closing, image=self._cancel_img, relief='flat', bd=0,
                              highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                              background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        final_btn.grid(row=0, column=1, padx=5, pady=5)

    def _update_data(self):
        """Extract the data specified by the file path and update the preview accordingly.

        The extracted data is stored in a `pandas.DataFrame`.
        """
        sep = self._separator.get()
        has_id = self._has_id.get()
        has_fnames = self._has_fnames.get()

        # finally, actually load the data according to the chosen params

        # print("separator: " + str(sep))
        # print("has_id: " + str(has_id))
        # print("has_fnames: " + str(has_fnames))
        # print("updating dataset...")

        self._data = exp._load_data(FileType.OBJECTS, self._file_path, has_fnames, has_id, sep)
        # ^ use same processing as objects file

        if self._prev is None:
            self._prev = DataSetPreviewFrame(self._preview_frame, self._data)
        else:
            self._prev.update(self._data)

    def _final_load(self):
        """Order the window to close and set confirmation variables to indicate that the file load is complete."""
        ws.on_close(self, self._parent_window)  # self.destroy() is covered within this method
        self._prev = None
        self._confirmation = Confirmation.DONE

    def _on_closing(self):
        """Order the window to close and set confirmation variables to indicate that the file load was cancelled."""
        ws.on_close(self, self._parent_window)  # self.destroy() is covered within this method
        self._prev = None
        self._confirmation = Confirmation.CANCEL
        print("Cancelled file load.")

    def get_confirmation(self):
        """Get the confirmation variable indicating whether or not the file load has been completed or cancelled.

        :return: the confirmation variable for the file that was attempted to be loaded.
        :rtype: :class:`pyplt.gui.experiment.dataset.params.Confirmation`
        """
        self.wait_window()
        return self._confirmation

    def get_data(self):
        """Get the data extracted and processed from the file being loaded.

        :return: the extracted data.
        :rtype: `pandas.DataFrame`
        """
        # N.B. splitting of single file into objects and ranks was moved from here to experiment execution
        # (see Experiment.run() method)
        return self._data
