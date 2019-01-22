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
from tkinter import filedialog, messagebox
import os
import numpy as np

from pyplt import ROOT_PATH
from pyplt.gui.experiment.dataset.params import LoadingParamsWindow, Confirmation
from pyplt.exceptions import ObjectsFirstException, RanksFormatException, IDsException, ObjectIDsFormatException, \
    NonNumericFeatureException, NoRanksDerivedError
from pyplt.util.enums import DataSetType, FileType


class DataLoadingTab(tk.Frame):
    """GUI tab for the data set loading stage of setting up an experiment.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent, parent_window):
        """Populates the frame widget.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this frame widget.
        :type parent_window: `tkinter.Toplevel`
        """
        self._parent = parent
        self._objects_confirm = None
        self._ranks_confirm = None
        self._single_confirm = None
        self._objects = None
        self._ranks = None
        self._exp_type = None
        self._parent_window = parent_window
        self._new_data = False
        self._objects_path = ""
        self._ranks_path = ""
        self._single_path = ""
        self._was_data_loaded = False
        self._init_dir = os.path.join(ROOT_PATH, "sample data sets")  # "../sample data sets"
        self._data = None

        tk.Frame.__init__(self, parent)

        self._objects_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/load_objects_128_30_01.png"))
        self._ranks_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/load_ranks_128_30_01.png"))
        self._single_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/load_data_128_30_01.png"))

        # Tab 1: Load Data

        # align all to center
        self.columnconfigure(0, weight=1)

        # Dual file format

        dual_frame = tk.Frame(self)
        dual_frame.grid(row=0, column=0, sticky='nsew', pady=(20, 10), padx=50)

        dual_label_frame = tk.Frame(dual_frame)
        dual_label_frame.pack(fill=tk.X, expand=True)

        dual_label = tk.Label(dual_label_frame, text="Dual file format")
        dual_label.grid(row=0, column=0, sticky='w')

        dual_sub_frame = tk.Frame(dual_frame, bd=2, relief='groove', pady=5)
        dual_sub_frame.pack(fill=tk.BOTH, expand=True)

        dual_btn_frame = tk.Frame(dual_sub_frame)
        load_objects_btn = tk.Button(dual_btn_frame, command=self._load_objects, image=self._objects_img,
                                     relief='flat', bd=0)
        load_objects_btn.grid(row=0, column=0, padx=(10, 5), sticky='w')
        load_ranks_btn = tk.Button(dual_btn_frame, command=self._load_ranks, image=self._ranks_img, relief='flat', bd=0)
        load_ranks_btn.grid(row=0, column=1, padx=(5, 10), sticky='w')
        dual_btn_frame.pack()

        # Single file format

        single_frame = tk.Frame(self)
        single_frame.grid(row=1, column=0, sticky='nsew', pady=(10, 10), padx=50)

        single_label_frame = tk.Frame(single_frame)
        single_label_frame.pack(fill=tk.X, expand=True)

        single_label = tk.Label(single_label_frame, text="Single file format")
        single_label.grid(row=0, column=0, sticky='w')

        single_sub_frame = tk.Frame(single_frame, bd=2, relief='groove', pady=5)
        single_sub_frame.pack(fill=tk.BOTH, expand=True)

        single_btn_frame = tk.Frame(single_sub_frame)
        load_single_btn = tk.Button(single_btn_frame, command=self._load_single, image=self._single_img,
                                    relief='flat', bd=0)
        load_single_btn.grid(row=0, column=0, sticky='w')
        single_btn_frame.pack()

        # Confirm labels

        confirm_frame = tk.Frame(self)
        confirm_frame.grid(row=2, column=0, sticky='nsew', pady=(20, 20), padx=100)

        confirm_labels_frame = tk.Frame(confirm_frame, pady=5)
        confirm_labels_frame.grid(row=0, column=0, sticky='nsew')
        # align to center
        confirm_frame.columnconfigure(0, weight=1)
        confirm_frame.rowconfigure(0, weight=1)

        # File info
        info_frame = tk.Frame(self)
        info_frame.grid(row=3, column=0, sticky='nsew', pady=(0, 20), padx=100)
        info_sub_frame = tk.Frame(info_frame, pady=5)
        info_sub_frame.grid(row=0, column=0, sticky='nsew')
        self._objects_info = tk.Label(info_sub_frame, font='Ebrima 8 normal')
        self._ranks_info = tk.Label(info_sub_frame, font='Ebrima 8 normal')
        self._single_info = tk.Label(info_sub_frame, font='Ebrima 8 normal')

        # align to center
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        info_sub_frame.columnconfigure(0, weight=1)
        info_sub_frame.rowconfigure(0, weight=1)
        info_sub_frame.rowconfigure(1, weight=1)
        info_sub_frame.rowconfigure(2, weight=1)

        self._objects_confirm = tk.StringVar()
        self._objects_confirm.set("Objects file not loaded.")
        self._ranks_confirm = tk.StringVar()
        self._ranks_confirm.set("Ranks file not loaded.")
        self._single_confirm = tk.StringVar()
        self._single_confirm.set("Single data file not loaded.")

        self._objects_label = tk.Label(confirm_labels_frame, textvariable=self._objects_confirm, fg='#990000')
        self._objects_label.grid(row=0, column=0, sticky='ew')
        self._ranks_label = tk.Label(confirm_labels_frame, textvariable=self._ranks_confirm, fg='#990000')
        self._ranks_label.grid(row=1, column=0, sticky='ew')
        self._single_label = tk.Label(confirm_labels_frame, textvariable=self._single_confirm, fg='#990000')
        self._single_label.grid(row=2, column=0, sticky='ew')

        # align to center
        confirm_labels_frame.columnconfigure(0, weight=1)
        confirm_labels_frame.rowconfigure(0, weight=1)
        confirm_labels_frame.rowconfigure(1, weight=1)
        confirm_labels_frame.rowconfigure(2, weight=1)

    def _load_data(self, file_type):
        """Load the data file of type file_type specified by the user.

        The user specifies the name of the file to be opened via a `tkinter.filedialog` window. The parameters
        with which the file is to be processed are then specified via a new
        :class:`pyplt.gui.experiment.dataset.params.LoadingParamsWindow`.
        The data is then extracted from the file and returned. The file must be a CSV file with
        a .csv extension. The GUI is updated to reflect the status of loaded files accordingly.

        :param file_type: the type of file to be opened.
        :type file_type: :class:`pyplt.util.enums.FileType`.
        :return: the data extracted from the file.
        :rtype: `pandas.DataFrame`
        """
        print("Load " + str(file_type.name).lower() + "...")
        fpath = filedialog.askopenfilename(parent=self._parent_window, initialdir=self._init_dir,
                                           title="Select " + str(file_type.name).lower() + " file",
                                           filetypes=(("csv files", "*.csv"),))

        if fpath != "":
            # print(objects_path)
            self._load_params = LoadingParamsWindow(self, fpath, self._parent_window, file_type)
            if self._load_params.get_confirmation() == Confirmation.DONE:
                if file_type == FileType.OBJECTS:
                    # Delete old ranks always bc they might not match (need to be re-validated w.r.t. the new objects)!
                    self._ranks = None
                    self._ranks_confirm.set("Ranks file not loaded.")
                    self._ranks_label.config(fg='#990000')
                    self._objects_confirm.set("Objects file loaded.")
                    self._objects_label.config(fg='green')
                    self._single_confirm.set("")
                    self._exp_type = DataSetType.PREFERENCES
                    self._objects_path = fpath
                    self._single_path = ""
                    self._ranks_path = ""
                    self._objects_info.config(text="Objects file path: " + str(fpath))
                    self._objects_info.grid(row=0, column=0, sticky='ew')
                    self._ranks_info.grid_forget()
                    self._single_info.grid_forget()
                    return self._load_params.get_data()
                elif file_type == FileType.RANKS:
                    if self._was_data_loaded:
                        self._new_data = True
                    self._ranks_confirm.set("Ranks file loaded.")
                    self._ranks_label.config(fg='green')
                    self._single_confirm.set("")
                    self._exp_type = DataSetType.PREFERENCES
                    self._ranks_path = fpath
                    self._single_path = ""
                    self._ranks_info.grid(row=1, column=0, sticky='ew')
                    self._ranks_info.config(text="Ranks file path: " + str(fpath))
                    return self._load_params.get_data()
                elif file_type == FileType.SINGLE:
                    # a bit different from OBJECTS and RANKS since we have to account for NoRanksDerivedError exception
                    try:
                        data = self._load_params.get_data()
                    except NoRanksDerivedError as error:
                        # show error message!
                        messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                        self._reset_single_info()
                        self._reset_object_info()
                        self._reset_ranks_info()
                        return
                        # ^ automatically skips the rest of the _load_single() method!
                        # (besides generating the <<FileChange>> event)
                        # (as if the file load was cancelled or invalid)
                    # in the event that no error (NoRanksDerivedError) was caught:
                    if self._was_data_loaded:
                        self._new_data = True
                    self._single_confirm.set("Single data file loaded.")
                    self._single_label.config(fg='green')
                    self._objects_confirm.set("")
                    self._ranks_confirm.set("")
                    self._exp_type = DataSetType.ORDERED
                    self._single_path = fpath
                    self._objects_path = ""
                    self._ranks_path = ""
                    self._objects_info.grid_forget()
                    self._ranks_info.grid_forget()
                    self._single_info.grid(row=2, column=0, sticky='ew')
                    self._single_info.config(text="Single file path: " + str(fpath))
                    return data
        else:
            print("Cancelled file load")

    def is_new_data(self):  # one-time only check
        """One-time check to verify whether new data (which may override any previously loaded data) has been loaded.

        If :attr:`self._new_data` is true, it is reset to False before this method returns True.

        :return: boolean indicating whether or not new data has been loaded (i.e., True if self._new_data is
            True; False otherwise).
        :rtype: bool
        """
        if self._new_data:
            self._new_data = False  # once checked, reset to false
            return True
        else:
            return False

    def _reset_object_info(self):
        """Reset the GUI file status to indicate that no object data is loaded.

        If new data had just been loaded, the GUI status for the ranks and single files are also reset.
        """
        if self._new_data:
            self._new_data = False
            self._reset_ranks_info()
            self._reset_single_info()
        self._objects_confirm.set("Objects file not loaded.")
        self._objects_label.config(fg='#990000')
        self._exp_type = None
        self._objects_path = ""
        self._objects_info.grid_forget()

    def _reset_ranks_info(self):
        """Reset the GUI file status to indicate that no rank data is loaded.

        If new data had just been loaded or single data was previously loaded,
        the GUI status for the objects and single files are also reset.
        """
        # additional single check just in case _load_data() isn't even run in the case of ObjectsFirstException...
        if self._new_data or self._single_confirm.get() == "Single data file loaded.":
            self._new_data = False
            self._reset_object_info()
            self._reset_single_info()
        self._ranks_confirm.set("Ranks file not loaded.")
        self._ranks_label.config(fg='#990000')
        self._exp_type = None
        self._ranks_path = ""
        self._ranks_info.grid_forget()

    def _reset_single_info(self):
        """Reset the GUI file status to indicate that no single data is loaded.

        If new data had just been loaded, the GUI status for the objects and ranks files are also reset.
        """
        if self._new_data:
            self._new_data = False
            self._reset_object_info()
            self._reset_ranks_info()
        self._single_confirm.set("Single data file not loaded.")
        self._single_label.config(fg='#990000')
        self._exp_type = None
        self._single_path = ""
        self._single_info.grid_forget()

    def _load_objects(self):
        """Attempt to load an objects file as specified by the user and carries out validation checks.

        If the data fails a validation check, the GUI file status is reset via :meth:`self._reset_object_info()`
        and the method returns early.

        :return:
            * the objects data extracted and processed from the specified file -- if valid.
            * None -- otherwise.
        :rtype: `pandas.DataFrame`
        """
        data = self._load_data(FileType.OBJECTS)
        self.event_generate("<<FileChange>>")
        if data is not None:  # to make sure it wasn't cancelled
            objects = data
            features = list(objects.columns)

            # Validation: Check if objects file contains numeric-only IDs
            try:
                # try to convert to integers...
                test = objects.iloc[:, 0].values.astype(int)
            except ValueError:
                error = ObjectIDsFormatException(suppress=True)
                messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                self._reset_object_info()
                self.event_generate("<<FileChange>>")
                return  # automatically skips the rest of the method...

            # Validation: Check for non-numeric features (& convert them to binary)
            try:
                can_be_float = objects.values.astype(float)
            except (ValueError, TypeError):
                error = NonNumericFeatureException(suppress=True)
                messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                self._reset_object_info()
                self.event_generate("<<FileChange>>")
                return  # automatically skips the rest of the method...
            # ^ TODO: make binary (binarization?) as in Java PLT instead of raising exception

            # only when all validation checks are passed, set actual variables!
            self._data = None
            self._ranks = None
            self._objects = data  # or objects (same)

    def _load_ranks(self):
        """Attempt to load a ranks file as specified by the user and carries out validation checks.

        If the data fails a validation check, the GUI file status is reset via :meth:`self._reset_ranks_info()`
        and the method returns early.

        :return:
            * the ranks data extracted and processed from the specified file -- if valid.
            * None -- otherwise.
        :rtype: `pandas.DataFrame`
        """
        # Validation: Check if the objects have been loaded first
        confirm = self._objects_confirm.get()
        if confirm == "" or confirm == "Objects file not loaded.":
            error = ObjectsFirstException(suppress=True)
            messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
            self._reset_ranks_info()
            self.event_generate("<<FileChange>>")
            return  # automatically skips the rest of the method...

        data = self._load_data(FileType.RANKS)
        self.event_generate("<<FileChange>>")
        if data is not None:  # to make sure it wasn't cancelled

            ranks = data

            # Validation: Check if ranks file contains 3 columns: (ID), Preferred_ID, Other_ID
            if len(ranks.columns) != 3:
                error = RanksFormatException(suppress=True)
                messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                self._reset_ranks_info()
                self.event_generate("<<FileChange>>")
                return  # automatically skips the rest of the method...

            # Validation: Check if all entries in ranks file refer to some ID (col 0) in objects.
            ids = list(self._objects.iloc[:, 0])
            arein = ranks.isin(ids).values[:, 1:]
            invalid = not (np.sum(arein, axis=(0, 1)) == arein.size)
            if invalid:
                # throw error
                error = IDsException(suppress=True)
                messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                self._reset_ranks_info()
                self.event_generate("<<FileChange>>")
                return  # automatically skips the rest of the method...

            # only when all validation checks are passed, set actual variables!
            self._ranks = data
            self._was_data_loaded = True

    def _load_single(self):
        """Attempt to load a single file as specified by the user and carries out validation checks.

        If the data fails a validation check, the GUI file status is reset via :meth:`self._reset_single_info()`
        and the method returns early.

        :return:
            * a tuple containing the objects and ranks data extracted and processed from the specified file -- if valid.
            * None -- otherwise.
        :rtype: type of `pandas.DataFrame` (size 2)
        """
        data = self._load_data(FileType.SINGLE)
        self.event_generate("<<FileChange>>")

        # Check for None because NoneType is not iterable and will give an error
        if data is not None:  # to make sure it wasn't cancelled
            # Validation: Check if objects file contains numeric-only IDs
            try:
                # try to convert to integers...
                test = data.iloc[:, 0].values.astype(int)
            except ValueError:
                error = ObjectIDsFormatException(suppress=True)
                messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                self._reset_single_info()
                self.event_generate("<<FileChange>>")
                return  # automatically skips the rest of the method...

            # Validation: Check for non-numeric features (& convert them to binary)
            try:
                can_be_float = data.values.astype(float)
            except (ValueError, TypeError):
                error = NonNumericFeatureException(suppress=True)
                messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                self._reset_object_info()
                self.event_generate("<<FileChange>>")
                return  # automatically skips the rest of the method...
            # ^ TODO: make binary (binarization?) as in Java PLT instead of raising exception

            # only when all validation checks are passed, set actual variables!
            self._objects = None
            self._ranks = None
            self._data = data  # or objects, ranks (same)
            self._was_data_loaded = True

    def is_data_loaded(self):
        """Check if a full data set has been loaded.

         A full data set consists of either both a valid objects file and a valid and compatible ranks file
         or a valid single file.

         :return: boolean indicating whether or not a full data set is loaded.
         :rtype: bool
         """
        if (self._objects_confirm.get() == "Objects file loaded."
            and self._ranks_confirm.get() == "Ranks file loaded.") \
                or (self._single_confirm.get() == "Single data file loaded."):
            return True
        else:
            return False

    def get_objects_path(self):
        """Get the objects file path, if applicable.

        :return: the objects file path.
        :rtype: str
        """
        return self._objects_path

    def get_ranks_path(self):
        """Get the ranks file path, if applicable.

        :return: the ranks file path.
        :rtype: str
        """
        return self._ranks_path

    def get_single_path(self):
        """Get single file path, if applicable.

        :return: the single file path.
        :rtype: str
        """
        return self._single_path

    def get_rank_derivation_params(self):
        """Get the values of the rank derivation parameters (minimum distance margin and memory).

        These only apply when a single file format is used.

        :return: tuple containing the values of both parameters.
        :rtype: tuple (size 2)
        """
        return self._load_params.get_rank_derivation_params()

    def get_data(self):
        """Get the loaded data.

        If the single file format is used, a single `pandas.DataFrame` containing the data is returned. If the dual
        file format is used, a tuple containing both the objects and ranks (each a `pandas.DataFrame`) is returned.

        :return: the loaded data.
        :rtype: `pandas.DataFrame` or tuple of `pandas.DataFrame` (size 2)
        """
        if self._data is None:
            return self._objects, self._ranks
        else:
            return self._data
