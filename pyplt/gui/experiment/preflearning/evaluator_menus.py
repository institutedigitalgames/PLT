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
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

from pyplt import ROOT_PATH
from pyplt.exceptions import ManualFoldsFormatException, FoldsSampleIDsException, NonNumericValuesException, \
    FoldsRowsException, MissingManualFoldsException
from pyplt.gui.experiment.dataset.params import Confirmation, LoadingParamsWindow, LoadingFoldsWindow
from pyplt.gui.util import colours


class HoldoutMenu(tk.Frame):
    """GUI menu for specifying parameters of the `Holdout` evaluation method.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent):
        """Initializes the frame widget and its contents.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter window`
        """
        # variables (set to default values):
        self._train_proportion = tk.IntVar(value=70)
        self._test_proportion = 30

        self._parent = parent
        tk.Frame.__init__(self, parent, bg=colours.EVAL_INNER, pady=15, relief='groove', bd=2)

        sub_frame = tk.Frame(self, bg=colours.EVAL_INNER)
        sub_frame.pack()

        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        tk.Label(sub_frame, text="Train proportion:", bg=colours.EVAL_INNER).grid(row=0, column=0, sticky='w',
                                                                                  padx=(0, 10), pady=(0, 5))
        entry = ttk.Entry(sub_frame, width=5, textvariable=self._train_proportion,
                          validate="all", validatecommand=self._vcmd, style='Eval.PLT.TEntry')
        entry.grid(row=0, column=1, pady=(0, 5))
        self.after_idle(lambda: entry.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values

        tk.Label(sub_frame, text="Test proportion:", bg=colours.EVAL_INNER).grid(row=1, column=0,
                                                                                 sticky='w', padx=(0, 10))
        self._test_proportion_text = ttk.Entry(sub_frame, width=5, style='Eval.PLT.TEntry')
        self._test_proportion_text.grid(row=1, column=1)
        self._test_proportion_text.insert(0, str(self._test_proportion))
        self._test_proportion_text.configure(state='disabled')
        self._test_proportion_text.bind("<<PLTStateToggle>>", self._check_test_prop_entry)  # bind

    def _check_test_prop_entry(self, event):
        """Ensure that the test proportion Entry widget remains disabled when a change in its state is detected.

        :param event: the <<PLTStateToggle>> event that triggered the call to this method.
        :type event: `tkinter Event`
        """
        new_state = str(self._test_proportion_text.cget('state'))
        # print("new_state: " + str(new_state))
        if (new_state == 'disable') or (new_state == 'disabled'):
            # print("Output neurons Entry state was changed (disabled)! -- ignoring...")
            return
        # print("Output neurons Entry state was changed (activated)!")
        # ALWAYS re-disable Output neurons Entry!
        # print("Setting back to disabled - as always.")
        self._test_proportion_text.configure(state='disable')  # set back to disabled
        # ^ n.b. state name does not always apply - check for specific widget!

    def _on_validate(self, action, index, value_if_allowed,
                     prior_value, text, validation_type, trigger_type, widget_name):
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
        :return: a boolean specifying whether the text is valid (True) or not (False).
        :rtype: bool
        """
        if text.isnumeric():
            try:
                if value_if_allowed == "":
                    # since train prop can be "", use test prop when returning params (see get_params())
                    self._test_proportion = 100
                elif int(value_if_allowed) <= 100:
                    self._test_proportion = 100 - int(value_if_allowed)
                else:
                    return False
                self._test_proportion_text.configure(state='normal')
                self._test_proportion_text.delete(0, 'end')
                self._test_proportion_text.insert(0, str(self._test_proportion))
                self._test_proportion_text.configure(state='disabled')
                return True
            except ValueError:
                return False
        else:
            return False

    def get_params(self):
        """Get the values for the `Holdout` method parameters as specified by the user via the GUI.

        The parameter values are returned in the form of a dict where the keys match the keywords of the arguments
        that would be passed to the corresponding :class:`pyplt.util.enums.EvaluatorType` constructor.

        :return: a dict containing the values for the following parameters in order:

            * test_proportion - the fractional proportion of data to be used as test data (the rest is to be used
              as training data) (default 0.3).
        :rtype: dict (size 1) of float
        """
        # since train prop can be "", use test prop when returning params (see on_validate())
        params_dict = {'test_proportion': self._test_proportion/float(100)}
        return params_dict


class KFCVMenu(tk.Frame):
    """GUI menu for specifying parameters of the `KFoldCrossValidation` evaluation method.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent, parent_window, files_tab, on_resize_fn):
        """Initializes the frame widget and its contents.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this tab (frame) widget.
        :type parent_window: `tkinter.Toplevel`
        :param files_tab: the `Load Data` tab.
        :type files_tab: :class:`pyplt.gui.experiment.dataset.loading.DataLoadingTab`
        :param on_resize_fn: the function called when the parent window is resized by the user. This is required by
            this class so that the window is resized accordingly whenever widgets are added to or removed from
            the `KFCVMenu` procedurally.
        :type on_resize_fn: function
        """
        self._files_tab = files_tab
        self._parent_window = parent_window
        self._on_resize_fn = on_resize_fn

        # variables (set to default values):
        self._k = tk.IntVar(value=3)
        self._test_folds = []

        self._parent = parent
        tk.Frame.__init__(self, parent, bg=colours.EVAL_INNER, pady=15, relief='groove', bd=2)

        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        self._sub_frame = tk.Frame(self, bg=colours.EVAL_INNER)
        self._sub_frame.pack(pady=(10, 10), padx=50)  # .grid(row=0, column=0, sticky='nsew', pady=(10, 10), padx=50)

        sub_label_frame = tk.Frame(self._sub_frame, bg=colours.EVAL_INNER)
        sub_label_frame.pack(fill=tk.X, expand=True)

        sub_label = tk.Label(sub_label_frame, text="Choose how to split your data: ", bg=colours.EVAL_INNER)
        sub_label.grid(row=0, column=0, sticky='w')

        options = ["Split into k folds", "Specify folds manually"]
        self._is_manual = tk.StringVar(value=options[0])
        sub_menu = ttk.OptionMenu(sub_label_frame, self._is_manual, options[0], *options,
                                  command=lambda _: self._update_kfcv_menu(), style='PLT.TMenubutton')
        sub_menu.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=(0, 10))

        # SET UP AUTOMATIC MENU

        self._auto_menu = tk.Frame(self._sub_frame, bg=colours.EVAL_INNER, bd=2, relief='groove',
                                   pady=10, padx=20)
        # don't pack yet - leave it to self._update_kfcv_menu()

        auto_sub_menu = tk.Frame(self._auto_menu, bg=colours.EVAL_INNER)
        auto_sub_menu.pack()

        tk.Label(auto_sub_menu, text="k (number of folds): ", bg=colours.EVAL_INNER).grid(row=0, column=0,
                                                                                          sticky='w',
                                                                                          padx=(0, 10),
                                                                                          pady=(0, 5))
        entry = ttk.Entry(auto_sub_menu, width=5, textvariable=self._k,
                          validate="all", validatecommand=self._vcmd, style='Eval.PLT.TEntry')
        entry.grid(row=0, column=1)
        self.after_idle(lambda: entry.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values

        # SET UP MANUAL MENU

        self._manual_menu = tk.Frame(self._sub_frame, bg=colours.EVAL_INNER, bd=2, relief='groove',
                                     pady=10, padx=20)
        # don't pack yet - leave it to self._update_kfcv_menu()

        manual_sub_menu = tk.Frame(self._manual_menu)
        manual_sub_menu.pack()

        self._manual_confirm = tk.StringVar(value="")  # "Manual folds confirmed."
        self._manual_path = tk.StringVar(value="")

        self._folds_btn_img = tk.PhotoImage(file=os.path.join(ROOT_PATH,
                                                              "assets/buttons/load_fold_ids_128_30_01_blue.png"))

        tk.Button(manual_sub_menu, command=self._set_manual_folds,
                  image=self._folds_btn_img, relief='flat', bd=0).pack()
        confirm_manual = tk.Label(manual_sub_menu, textvariable=self._manual_confirm, fg='green')
        confirm_manual.pack()
        manual_path = tk.Label(manual_sub_menu, textvariable=self._manual_path, font='Ebrima 8 normal')
        manual_path.pack()

        # trigger first kfcv menu update
        self._update_kfcv_menu()

    def _update_kfcv_menu(self):
        print("Updating kfcv menu...")
        if self._is_manual.get() == "Split into k folds":
            # AUTOMATIC APPROACH

            # first remove manual menu if it was previously open
            if self._manual_menu is not None:
                self._manual_menu.pack_forget()

            self._auto_menu.pack(fill=tk.BOTH, expand=True)
        else:  # if self._is_manual == "Specify folds manually"
            # MANUAL APPROACH

            # first remove auto menu if it was previously open
            if self._auto_menu is not None:
                self._auto_menu.pack_forget()

            self._manual_menu.pack(fill=tk.BOTH, expand=True)
            # make them appear properly
            self.update_idletasks()
            self._on_resize_fn(None)

    def _set_manual_folds(self):
        print("Load folds...")
        init_dir = os.path.join(ROOT_PATH, "sample data sets")  # "../sample data sets"
        fpath = filedialog.askopenfilename(parent=self._parent_window, initialdir=init_dir,
                                           title="Select folds file",
                                           filetypes=(("csv files", "*.csv"),))

        if fpath != "":
            main_data = self._files_tab.get_data()
            if isinstance(main_data, tuple):  # dual file format
                is_dual_format = True
            else:  # single file format
                is_dual_format = False

            self._load_params = LoadingFoldsWindow(self, fpath, self._parent_window,
                                                   is_dual_format=is_dual_format)
            if self._load_params.get_confirmation() == Confirmation.DONE:
                data = self._load_params.get_data()  # should have IDs col and column names by now

                # Validation: Check if manual folds file contains 2 columns: (ID), Test Fold ID
                if len(data.columns) != 2:
                    error = ManualFoldsFormatException(suppress=True)
                    messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                    # reset stuff
                    print("Failed file load")
                    self._manual_confirm.set("")
                    self._manual_path.set("")
                    self._test_folds = None
                    return  # automatically skips the rest of the method...

                # Validation: Check for non-numeric values
                try:
                    can_be_float = data.values.astype(float)
                except (ValueError, TypeError):
                    error = NonNumericValuesException(suppress=True)
                    messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                    # reset stuff
                    print("Failed file load")
                    self._manual_confirm.set("")
                    self._manual_path.set("")
                    self._test_folds = None
                    return  # automatically skips the rest of the method...

                # Validation: Check that all object/rank IDs in the manual file refer to real objects/ranks in the data
                # (even if IDs was just added by PLT)
                if is_dual_format:  # dual file format
                    ranks = main_data[1]
                    real_ids = ranks.iloc[:, 0]  # get real rank IDs from ID col of ranks
                else:  # single file format
                    real_ids = main_data.iloc[:, 0]  # get real sample/object IDs from ID col of (main) data
                sample_ids = data.iloc[:, 0]  # sample IDs in manual folds file
                # print("real_ids: ")
                # print(real_ids)
                # print("sample_ids: ")
                # print(sample_ids)
                if len(sample_ids) != len(real_ids):
                    error = FoldsRowsException(suppress=True)
                    messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                    # reset stuff
                    print("Failed file load")
                    self._manual_confirm.set("")
                    self._manual_path.set("")
                    self._test_folds = None
                    return  # automatically skips the rest of the method...
                result_vs_real_ids = np.where(sample_ids == real_ids, True, False)
                result_vs_real_indices = np.where(sample_ids == np.arange(len(real_ids)), True, False)
                # print("result (vs real IDs): ")
                # print(result_vs_real_ids)
                # print("result (vs real integer-based indices): ")
                # print(result_vs_real_indices)
                # raise error if not all sample IDs in the file match the real sample IDs or their integer-based indices
                if (np.sum(result_vs_real_ids) != len(sample_ids)) and \
                        (np.sum(result_vs_real_indices) != len(sample_ids)):
                    error = FoldsSampleIDsException(suppress=True)
                    messagebox.showerror(error.get_summary(), error.get_message(), parent=self._parent_window)
                    # reset stuff
                    print("Failed file load")
                    self._manual_confirm.set("")
                    self._manual_path.set("")
                    self._test_folds = None
                    return  # automatically skips the rest of the method...

                # only when all validation checks are passed, set variables n stuff!
                print("Folds file successfully loaded.")
                self._manual_confirm.set("Manual folds successfully loaded.")
                self._manual_path.set(str(fpath))
                self._test_folds = data.iloc[:, 1].values
                # ^ use second column (Test Fold ID column) bc first col is the object/rank ID column
                # make them appear properly
                self.update_idletasks()
                self._on_resize_fn(None)
        else:
            # reset stuff
            print("Cancelled file load")
            self._manual_confirm.set("")
            self._manual_path.set("")
            self._test_folds = None

    def _on_validate(self, action, index, value_if_allowed,
                     prior_value, text, validation_type, trigger_type, widget_name):
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
        :return: a boolean specifying whether the text is valid (True) or not (False).
        :rtype: bool
        """
        if text.isnumeric():
            return True
        else:
            return False

    def get_params(self):
        """Get the values for the `K-Fold Cross Validation` method parameters as specified by the user via the GUI.

        The parameter values are returned in the form of a dict where the keys match the keywords of the arguments
        that would be passed to the corresponding :class:`pyplt.util.enums.EvaluatorType` constructor.

        :return: a dict containing the values for the following parameters in order:

            * k - the number of folds to uniformly split the data into when using the automatic approach (default 3).
            * test_folds - an array specifying the fold index for each sample in the dataset when using
              the manual approach (default None). The entry test_folds[i] specifies the index of the test set that
              sample i belongs to. It is also possible to exclude sample i from any test set (i.e., include sample i
              in every training set) by setting test_folds[i] to -1.
              If `test_folds` is None, the automatic approach is assumed and only the `k` parameter is to be considered.
              Otherwise, the manual approach is to be assumed and only the `test_folds` parameter is to be considered.
        :rtype: dict (size 2) containing:
            * k - int or None
            * test_folds - `numpy.ndarray` or None
        :raises MissingManualFoldsException: if the user chooses to specify folds manually for cross validation but
            fails to load the required file containing the fold IDs.
        """
        if self._is_manual.get() == "Split into k folds":
            # AUTOMATIC APPROACH
            k = self._k.get()
            test_folds = None
        else:  # if self._is_manual == "Specify folds manually"
            # MANUAL APPROACH
            k = None
            test_folds = self._test_folds

        if (k is None) and (test_folds == []):
            raise MissingManualFoldsException

        params_dict = {'k': k,
                       'test_folds': test_folds}
        return params_dict
