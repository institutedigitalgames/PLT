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
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

from pyplt import ROOT_PATH
from pyplt.util.enums import FSMethod
from pyplt.gui.util import windowstacking as ws, colours
from pyplt.gui.util.help import ResultsHelpDialog
from pyplt.gui.util import text


class ResultsWindow(tk.Toplevel):
    """GUI window displaying the results of an experiment.

     The window widget extends the class `tkinter.Toplevel`.
    """

    def __init__(self, parent, parent_window, experiment, time_info, data_info, preproc_info,
                 pl_algo_info, eval_metrics, shuffle_info,  # mandatory
                 fs_info=None, fs_algo_info=None, fs_eval_info=None, pl_eval_info=None, fold_metrics=None):  # optional
        """Initializes the window widget with all of the information about and results obtained from the given experiment.

        :param parent: the parent widget of this window widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        :param experiment: the given experiment.
        :type experiment: :class:`pyplt.experiment.Experiment`
        :param time_info: a list containing meta-data about the experiment related to time (the start timestamp (UTC),
            the end timestamp (UTC), and the duration).
        :type time_info: list of float (size 3)
        :param data_info: a list containing the number of objects, the number of ranks, and the list of data file paths.
        :type data_info: list (size 3)
        :param preproc_info: a list containing the include settings dict and the normalization settings dict.
        :type preproc_info: list of dict (size 2)
        :param pl_algo_info: a list containing the algorithm type (:class:`pyplt.util.enums.PLAlgo`)
            and the string representation of its parameters.
        :type pl_algo_info: list (size 2)
        :param eval_metrics: the evaluation/training results in the form of a dict with keys:

            * '`Training Accuracy`'
            * '`Test Accuracy`' (if applicable)
        :type eval_metrics: dict
        :param shuffle_info: list containing the chosen settings related to shuffling the dataset:

            * shuffle -- bool specifying whether or not the dataset was shuffled at the start of the experiment
              execution.
            * random_seed -- optional seed (int or None) used to shuffle the dataset.
        :type shuffle_info: list (size 2)
        :param fs_info: a list containing the chosen feature selection method type
            (:class:`pyplt.util.enums.FSMethod`), the string representation of its parameters, and the list of
            features selected by the feature selection method.
        :type fs_info: list (size 3) or None, optional
        :param fs_algo_info: a list containing the chosen algorithm type
            (:class:`pyplt.util.enums.PLAlgo`) for the feature selection stage and the string
            representation of its parameters.
        :type fs_algo_info: list (size 2) or None, optional
        :param fs_eval_info: a list containing the evaluation method type
            (:class:`pyplt.util.enums.EvaluatorType`) for the feature selection stage and the
            string representation of its parameters.
        :type fs_eval_info: list (size 2) or None, optional
        :param pl_eval_info: a list containing the evaluation method type
            (:class:`pyplt.util.enums.EvaluatorType`) and the string representation of its parameters.
        :type pl_eval_info: list (size 2) or None, optional
        :param fold_metrics: optional fold-specific information (default None) in the form of list of tuples, each
            containing the start timestamp, end timestamp, evaluation metrics, and a `pandas.DataFrame`
            representation of the trained model as follows:

                * start_time -- `datetime` timestamp (UTC timezone)
                * end_time -- `datetime` timestamp (UTC timezone)
                * eval_metrics -- dict with keys:

                  * '`Training Accuracy`'
                  * '`Test Accuracy`' (if applicable)
                * model -- `pandas.DataFrame`
        :type fold_metrics: list of tuple, optional
        """
        self._parent = parent
        self._parent_window = parent_window
        self._exp = experiment
        tk.Toplevel.__init__(self, self._parent)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, self._parent_window, ws.Mode.OPEN_ONLY)
        ws.stack_window(self, self._parent_window)
        self.protocol("WM_DELETE_WINDOW", self._on_close_safe)

        self.title("Experiment Report")  # Results

        self._OS = platform.system()

        # (for scrollbars/)
        self._results_canvas = tk.Canvas(self)
        self._results_canvas.bind("<Configure>", self._on_resize)
        self._results_canvas.bind('<Enter>', self._bind_mousewheel)
        self._results_canvas.bind('<Leave>', self._unbind_mousewheel)
        self._canvas_height = self._results_canvas.winfo_reqheight()
        self._canvas_width = self._results_canvas.winfo_reqwidth()
        self._results_sub_frame = tk.Frame(self._results_canvas, padx=10, pady=10)
        # (/for scrollbars)

        n_obj = data_info[0]
        n_ranks = data_info[1]
        obj_path, ranks_path, single_path = data_info[2]
        f_include = preproc_info[0]  # dict
        f_norm = preproc_info[1]  # dict
        n_feats = len(f_include)  # no need for -1 as ID column not included
        fs_method = fs_info[0]
        fs_method_params = fs_info[1]
        sel_feats = fs_info[2]
        fs_algo = fs_algo_info[0]
        fs_algo_params = fs_algo_info[1]
        fs_eval = fs_eval_info[0]
        fs_eval_params = fs_eval_info[1]
        pl_algo = pl_algo_info[0]
        pl_algo_params = pl_algo_info[1]
        pl_eval = pl_eval_info[0]
        pl_eval_params = pl_eval_info[1]
        s_time, e_time, duration = time_info
        shuffle, random_seed = shuffle_info

        self._pl_algo = self._exp.get_pl_algorithm()  # get the actual algorithm PLAlgo object

        #############
        # META DATA
        #############

        exp_frame = tk.Frame(self._results_sub_frame, bd=2, relief='groove', padx=50, pady=5)
        exp_frame.grid(row=0, column=0, pady=5, sticky='ew')
        exp_title_frame = tk.Frame(exp_frame)
        exp_title_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(exp_title_frame, text="Experiment Times", justify='center', font='Ebrima 14 bold').pack()
        exp_sub_frame = tk.Frame(exp_frame)
        exp_sub_frame.pack()
        tk.Label(exp_sub_frame, text="Start timestamp (UTC):").grid(row=0, column=0, sticky='w')
        tk.Label(exp_sub_frame, text=str(s_time)).grid(row=0, column=1, sticky='w')
        tk.Label(exp_sub_frame, text="End timestamp (UTC):").grid(row=1, column=0, sticky='w')
        tk.Label(exp_sub_frame, text=str(e_time)).grid(row=1, column=1, sticky='w')
        tk.Label(exp_sub_frame, text="Duration:").grid(row=2, column=0, sticky='w')
        tk.Label(exp_sub_frame, text=str(duration)).grid(row=2, column=1, sticky='w')

        #############
        # DATA
        #############

        data_frame = tk.Frame(self._results_sub_frame, bd=2, relief='groove', padx=50, pady=5)
        data_frame.grid(row=1, column=0, pady=5, sticky='ew')
        data_title_frame = tk.Frame(data_frame)
        data_title_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(data_title_frame, text="Data", justify='center', font='Ebrima 14 bold').pack()
        data_sub_frame = tk.Frame(data_frame)
        data_sub_frame.pack()
        data_paths_frame = tk.Frame(data_sub_frame)
        data_paths_frame.grid(row=0, column=0, sticky='w')
        if not (single_path == ""):
            tk.Label(data_paths_frame, text="Single file path:").grid(row=0, column=0, sticky='w')
            tk.Label(data_paths_frame, text=str(single_path)).grid(row=0, column=1, sticky='w')
        else:
            tk.Label(data_paths_frame, text="Objects file path:").grid(row=0, column=0, sticky='w')
            tk.Label(data_paths_frame, text=str(obj_path)).grid(row=0, column=1, sticky='w')
            tk.Label(data_paths_frame, text="Ranks file path:").grid(row=1, column=0, sticky='w')
            tk.Label(data_paths_frame, text=str(ranks_path)).grid(row=1, column=1, sticky='w')
        data_other_frame = tk.Frame(data_sub_frame)
        data_other_frame.grid(row=1, column=0, sticky='ew')
        data_other_sub_frame = tk.Frame(data_other_frame)
        data_other_sub_frame.pack()
        tk.Label(data_other_sub_frame, text="Number of objects/samples:").grid(row=0, column=0, sticky='w')
        tk.Label(data_other_sub_frame, text=str(n_obj)).grid(row=0, column=1, sticky='e')
        tk.Label(data_other_sub_frame, text="Number of original features:").grid(row=1, column=0, sticky='w')
        tk.Label(data_other_sub_frame, text=str(n_feats)).grid(row=1, column=1, sticky='e')
        tk.Label(data_other_sub_frame, text="Number of ranks/preferences:").grid(row=2, column=0, sticky='w')
        tk.Label(data_other_sub_frame, text=str(n_ranks)).grid(row=2, column=1, sticky='e')

        ####################################
        # FEATURES INCLUDED & PRE-PROCESSING
        ####################################

        preproc_frame = tk.Frame(self._results_sub_frame, bd=2, relief='groove', padx=50, pady=5)
        preproc_frame.grid(row=2, column=0, pady=5, sticky='ew')
        tk.Label(preproc_frame, text="Features Included & Pre-processing",
                 justify='center', font='Ebrima 14 bold').pack()
        pf = tk.Frame(preproc_frame)
        pf.pack(pady=(5, 0))
        r = 0
        tk.Label(pf, text="Feature", font='Ebrima 11 normal').grid(row=r, column=0, sticky='w', padx=(0, 5))
        tk.Label(pf, text="Normalization", font='Ebrima 11 normal').grid(row=r, column=1, padx=(5, 0))
        for f in f_include:
            r += 1
            # no need to skip first row as ID is excluded in f_include and f_norm
            if f_include[f].get():  # only if feature included (==True)
                tk.Label(pf, text=str(f)).grid(row=r, column=0, sticky='w')  # TODO: need orig_feats[f] here for label
                tk.Label(pf, text=text.real_type_name(str(f_norm[f].get()))).grid(row=r, column=1)

        ttk.Separator(preproc_frame, orient=tk.HORIZONTAL).pack(fill=tk.BOTH, expand=True, pady=5)

        shuffle_frame = tk.Frame(preproc_frame)
        shuffle_frame.pack()
        shuffle_applied = "No"
        seed = "N/A"
        if shuffle:
            shuffle_applied = "Yes"
            seed = random_seed
        tk.Label(shuffle_frame, text="Dataset shuffle enabled?").grid(row=0, column=0, sticky='e', padx=(0, 5))
        tk.Label(shuffle_frame, text=str(shuffle_applied)).grid(row=0, column=1, sticky='w', padx=(5, 0))
        tk.Label(shuffle_frame, text="Random seed:").grid(row=1, column=0, sticky='e', padx=(0, 5))
        tk.Label(shuffle_frame, text=str(seed)).grid(row=1, column=1, sticky='w', padx=(5, 0))
        shuffle_frame.grid_columnconfigure(0, weight=1)
        shuffle_frame.grid_columnconfigure(1, weight=1)

        ###################
        # FEATURE SELECTION
        ###################

        fs_frame = tk.Frame(self._results_sub_frame, bd=2, relief='groove', padx=50, pady=5)
        # fs_frame.pack(pady=5)
        fs_frame.grid(row=3, column=0, pady=5, sticky='ew')
        tk.Label(fs_frame, text="Feature Selection", justify='center', font='Ebrima 14 bold').pack()
        fs_sub_frame = tk.Frame(fs_frame)
        fs_sub_frame.pack()
        tk.Label(fs_sub_frame, text="Method: " + text.real_type_name(str(fs_method.name))).grid(row=0, column=0)
        if (fs_method_params is not None) and (not fs_method_params == "{}"):
            fs_params_frame = tk.Frame(fs_sub_frame)
            fs_params_frame.grid(row=1, column=0)
            tk.Label(fs_params_frame, text="Parameters:", font='Ebrima 10 italic').pack()
            fs_params_sub_frame = tk.Frame(fs_params_frame)
            fs_params_sub_frame.pack()
            self._print_params(fs_params_sub_frame, fs_method_params)
        if fs_algo is not None:
            fs_a_frame = tk.Frame(fs_frame)
            fs_a_frame.pack(pady=10)
            tk.Label(fs_a_frame, text="Algorithm: " + text.real_type_name(str(fs_algo.name))).grid(row=0, column=0)
            if (fs_algo_params is not None) and (not fs_algo_params == "{}"):
                fs_algo_params_frame = tk.Frame(fs_a_frame)
                fs_algo_params_frame.grid(row=1, column=0)
                tk.Label(fs_algo_params_frame, text="Parameters:", font='Ebrima 10 italic').pack()
                fs_algo_params_sub_frame = tk.Frame(fs_algo_params_frame)
                fs_algo_params_sub_frame.pack()
                self._print_params(fs_algo_params_sub_frame, fs_algo_params)

        if fs_eval is not None:
            fs_e_frame = tk.Frame(fs_frame)
            fs_e_frame.pack()
            tk.Label(fs_e_frame, text="Evaluation: " + text.real_type_name(str(fs_eval.name))).grid(row=0, column=0)
            if (fs_eval_params is not None) and (not fs_eval_params == "{}"):
                fs_eval_params_frame = tk.Frame(fs_e_frame)
                fs_eval_params_frame.grid(row=1, column=0)
                tk.Label(fs_eval_params_frame, text="Parameters:", font='Ebrima 10 italic').pack()
                fs_eval_params_sub_frame = tk.Frame(fs_eval_params_frame)
                fs_eval_params_sub_frame.pack()
                self._print_params(fs_eval_params_sub_frame, fs_eval_params)
        sf = tk.Frame(fs_frame)
        sf.pack(pady=(10, 0))
        if not (fs_method == FSMethod.NONE):
            tk.Label(sf, text="Selected Features:", font='Ebrima 10 bold').pack()
            sf_sub_frame = tk.Frame(sf)
            sf_sub_frame.pack()
            r = 0
            for f in sel_feats:
                tk.Label(sf_sub_frame, text=str(f)).grid(row=r, column=0, sticky='w')
                r += 1

        #####################
        # PREFERENCE LEARNING
        #####################

        pl_frame = tk.Frame(self._results_sub_frame, bd=2, relief='groove', padx=50, pady=5)
        pl_frame.grid(row=4, column=0, pady=5, sticky='ew')
        tk.Label(pl_frame, text="Preference Learning", justify='center', font='Ebrima 14 bold').pack()
        pl_a_frame = tk.Frame(pl_frame)
        pl_a_frame.pack(pady=10)
        tk.Label(pl_a_frame, text="Algorithm: " + text.real_type_name(str(pl_algo.name))).grid(row=0, column=0)
        if (pl_algo_params is not None) and (not pl_algo_params == "{}"):
            pl_algo_params_frame = tk.Frame(pl_a_frame)
            pl_algo_params_frame.grid(row=1, column=0)
            tk.Label(pl_algo_params_frame, text="Parameters:", font='Ebrima 10 italic').pack()
            pl_algo_params_sub_frame = tk.Frame(pl_algo_params_frame)
            pl_algo_params_sub_frame.pack()
            self._print_params(pl_algo_params_sub_frame, pl_algo_params)

        pl_e_frame = tk.Frame(pl_frame)
        pl_e_frame.pack()
        tk.Label(pl_e_frame, text="Evaluation: " + text.real_type_name(str(pl_eval.name))).grid(row=0, column=0)
        if (pl_eval_params is not None) and (not pl_eval_params == "{}"):
            pl_eval_params_frame = tk.Frame(pl_e_frame)
            pl_eval_params_frame.grid(row=1, column=0)
            tk.Label(pl_eval_params_frame, text="Parameters:", font='Ebrima 10 italic').pack()
            pl_eval_params_sub_frame = tk.Frame(pl_eval_params_frame)
            pl_eval_params_sub_frame.pack()
            self._print_params(pl_eval_params_sub_frame, pl_eval_params)

        ###################
        # MODEL PERFORMANCE
        ###################

        perf_frame = tk.Frame(self._results_sub_frame, bd=2, relief='groove', padx=50, pady=5)
        # perf_frame.pack(pady=5)
        perf_frame.grid(row=5, column=0, pady=5, sticky='ew')
        tk.Label(perf_frame, text="Model Performance", justify='center', font='Ebrima 14 bold').pack()

        self._tree = None
        self._tree_scroll = None
        self._fold_metrics = fold_metrics
        if len(self._fold_metrics) > 1:
            fold_metrics_frame = tk.Frame(perf_frame, bd=2, relief='sunken')
            fold_metrics_frame.pack(pady=10)

            eval_metric_names = tuple(self._fold_metrics[0][2].keys())  # keys of eval_metrics of first fold in fold_metrics
            cols = ('Fold', )+eval_metric_names

            self._tree = ttk.Treeview(fold_metrics_frame, columns=cols, height=4)
            for c in cols:
                self._tree.heading(c, text=c)
                self._tree.column(c, width=150)
            self._tree['show'] = 'headings'

            f = 0
            for fold in self._fold_metrics:
                f = f + 1
                fold_name = "Fold "+str(f)
                self._tree.insert("", tk.END, iid=f, values=(fold_name, )+tuple(fold[2].values()))

            # add vertical scrollbar beside Treeview
            self._tree_scroll = ttk.Scrollbar(fold_metrics_frame, orient="vertical", command=self._tree.yview)
            self._tree_scroll.pack(side='right', fill='y')
            self._tree.configure(yscrollcommand=self._tree_scroll.set)

            # finally, pack the Treeview itself
            self._tree.pack()

        avg_metrics_frame = tk.Frame(perf_frame)
        avg_metrics_frame.pack()
        r = 0
        for e in eval_metrics:
            tk.Label(avg_metrics_frame, text=str(e) + ": ").grid(row=r, column=0, sticky='w')
            tk.Label(avg_metrics_frame, text=str(eval_metrics[e]) + " %").grid(row=r, column=1, sticky='w')
            r += 1

        # Send additional data to Experiment for log saving
        f_norm_clean = dict()
        fi = 0
        for f in f_norm:
            # no need to ignore 0 since ID column is not included
            f_norm_clean[f] = f_norm[f].get()  # get StringVar value
            fi += 1
        self._exp._set_file_meta_data(obj_path, ranks_path, single_path, f_norm_clean)

        # Buttons to save model and/or save experiment log
        self._model_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/save_model_128_30_01_white.png"))
        self._report_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/save_report_128_30_01_white.png"))
        self._help_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/help.png"))
        self._fake_btn_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/fake_btn.png"))

        self._algo = self._exp.get_pl_algorithm()
        save_buttons_frame = tk.Frame(self, bd=2, relief='groove', background=colours.NAV_BAR)
        save_buttons_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)

        help_btn = tk.Button(save_buttons_frame, command=self._help_dialog, image=self._help_img, relief='flat', bd=0,
                             highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                             background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        help_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # add fake button to center save buttons
        fake_btn = tk.Button(save_buttons_frame, image=self._fake_btn_img, relief='flat', bd=0,
                             highlightbackground=colours.NAV_BAR, highlightcolor=colours.NAV_BAR, highlightthickness=0,
                             background=colours.NAV_BAR, activebackground=colours.NAV_BAR)
        fake_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        sbf = tk.Frame(save_buttons_frame, background=colours.NAV_BAR)
        sbf.pack()
        save_model_btn = tk.Button(sbf, command=lambda t=time.time(): self._save_model(t),
                                   image=self._model_img, relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                                   highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                                   activebackground=colours.NAV_BAR)
        save_model_btn.grid(row=0, column=0, padx=10, pady=10)
        save_exp_log_btn = tk.Button(sbf, command=lambda t=time.time(): self._save_exp_log_with_dialog(t),
                                     image=self._report_img, relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                                     highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                                     activebackground=colours.NAV_BAR)
        save_exp_log_btn.grid(row=0, column=1, padx=10, pady=10)

        # add scrollbars

        v_scroll = ttk.Scrollbar(self, orient="vertical", command=self._results_canvas.yview)
        v_scroll.pack(side='right', fill='y')
        self._results_canvas.configure(yscrollcommand=v_scroll.set)
        h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self._results_canvas.xview)
        h_scroll.pack(side='bottom', fill='x')
        self._results_canvas.configure(xscrollcommand=h_scroll.set)

        # pack everything
        self._results_sub_frame.pack(fill=tk.BOTH, expand=True)
        self._results_canvas.pack(side='left', expand=True, fill=tk.BOTH)

        self._results_canvas.create_window((0, 0), window=self._results_sub_frame, anchor='n')
        self._results_canvas.config(scrollregion=self._results_canvas.bbox("all"))

        self._results_sub_frame.bind('<Configure>', self._on_canvas_config)

        # scroll to bottom (to show accuracy) for the first time
        self.update()
        self._results_canvas.yview(tk.MOVETO, 1)
        self.update_idletasks()

    def _save_model(self, timestamp):
        if self._tree is None:  # only one or two folds so only one model to save
            success = self._algo.save_model_with_dialog(timestamp, self)
            if success:
                # show info box confirmation
                messagebox.showinfo("Model successfully saved", "The model was successfully saved!", parent=self)
        else:  # more than two folds, so more than one model to save
            # save the model of the selected fold in the tree!
            fold = self._tree.focus()
            if fold == '':
                # show error in messagebox - no model selected - please select the fold for which to save the model
                messagebox.showerror("Cannot save model", "No fold was specified. Please select the fold for which "
                                                          "to save the model.", parent=self)
            else:
                # algo = self._fold_metrics[int(fold)-1][3]  # i.e. the algo_copy of the given row in fold_metrics
                # success = algo.save_model_with_dialog(timestamp, self, suffix="_fold"+str(fold))
                model = self._fold_metrics[int(fold)-1][3]  # i.e. the algo_copy of the given row in fold_metrics
                success = self._save_model_dialog(model, timestamp, self, suffix="_fold"+str(fold))
                if success:
                    # show info box confirmation
                    messagebox.showinfo("Model successfully saved", "The model for fold " + fold +
                                        " was successfully saved!", parent=self)

    def _save_model_dialog(self, model, timestamp, parent_window, suffix=""):
        """Open a file dialog window (GUI) and save the given model to file at the path indicated by the user.

        The model file must be a Comma Separated Value (CSV)-type file with the extension '.csv'.

        :param model: a `pandas.DataFrame` representation of the model to be saved.
        :type model: `pandas.DataFrame`
        :param timestamp: the timestamp to be included in the file name.
        :type timestamp: float
        :param parent_window: the window widget which the file dialog window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        :param suffix: an additional string to add at the end of the file name (default "").
        :type suffix: str, optional
        :return: specifies whether or not the file was successfully saved.
        :rtype: bool
        """
        # similar to get_params_string()
        init_dir = os.path.join(ROOT_PATH, "logs")
        init_filename = "model_" + str(timestamp) + suffix + ".csv"
        fpath = filedialog.asksaveasfilename(parent=parent_window, initialdir=init_dir,
                                             title="Save model file", initialfile=init_filename,
                                             defaultextension=".csv", filetypes=(("CSV files", "*.csv"),))
        if fpath != "":
            # Finally, save to file!
            model.to_csv(fpath, index=False)
            return True
        else:
            print("Cancelled save model.")
            return False

    def _help_dialog(self):
        """Open a help dialog window to assist the user."""
        ResultsHelpDialog(self)

    def _on_canvas_config(self, event):
        """Update the canvas scrollregion to account for its entire bounding box.

        This method is bound to all <Configure> events with respect to :attr:`self._results_sub_frame`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("__ on_config called __")
        self._results_canvas.configure(scrollregion=self._results_canvas.bbox("all"))

    def _on_resize(self, event):
        """Resize the canvas widget according to the user's specification via the mouse.

        This method is called whenever a <Configure> event occurs with respect to :attr:`self._results_canvas`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        old_x = self._results_canvas.xview()
        old_y = self._results_canvas.yview()

        # determine the ratio of old width/height to new width/height
        w_scale = float(event.width) / self._canvas_width
        h_scale = float(event.height) / self._canvas_height
        self._canvas_width = event.width
        self._canvas_height = event.height
        # resize the canvas
        self.config(width=self._canvas_width, height=self._canvas_height)
        # rescale all the objects tagged with the "all" tag
        self._results_canvas.scale("all", 0, 0, w_scale, h_scale)

        # also scale frame!
        self._results_sub_frame.config(width=self._canvas_width, height=self._canvas_height)

        # force update of scroll region for when scroll position is at the bottom!
        self._results_canvas.config(scrollregion=self._results_canvas.bbox("all"))

        # re-set x and y positions
        # print("old x: " + str(old_x))
        # print("old y: " + str(old_y))
        self._results_canvas.bind('<Expose>', lambda _, x=old_x, y=old_y: self._move_canvas(_, x, y))

    def _move_canvas(self, event, x, y):
        """Scroll the vertical and horizontal scrollbars of the canvas to a particular position.

        This method is called whenever an <Expose> event occurs with respect to :attr:`self._results_canvas`.

        :param event: the <Expose> event that triggered the method call.
        :type event: `tkinter Event`
        :param x: specifies the view position (top & bottom) to set the horizontal scrollbar to from interval
            [0.0, 1.0].
        :type x: tuple of float (size 2)
        :param y: specifies the view position (top & bottom) to set the vertical scrollbar to from interval
            [0.0, 1.0].
        :type y: tuple of float (size 2)
        """
        if not(x[0] == 0.0 and x[1] == 1.0):  # only do so when xview is actually scrollable (not greyed out)
            self._results_canvas.xview(tk.MOVETO, x[0])
        self._results_canvas.yview(tk.MOVETO, y[0])
        self._results_canvas.unbind('<Expose>')

    def _bind_mousewheel(self, event):
        """Bind all mouse wheel events with respect to the canvas to a canvas-scrolling function.

        This method is called whenever an <Enter> event occurs with respect to :attr:`self._results_canvas`.

        :param event: the <Enter> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # for Windows OS and MacOS
        self._results_canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        # for Linux OS
        self._results_canvas.bind_all("<Button-4>", self._on_mouse_scroll)
        self._results_canvas.bind_all("<Button-5>", self._on_mouse_scroll)

    def _unbind_mousewheel(self, event):
        """Unbind all mouse wheel events with respect to the canvas from any function.

        This method is called whenever a <Leave> event occurs with respect to :attr:`self._results_canvas`.

        :param event: the <Leave> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # for Windows OS and MacOS
        self._results_canvas.unbind_all("<MouseWheel>")
        # for Linux OS
        self._results_canvas.unbind_all("<Button-4>")
        self._results_canvas.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        """Vertically scroll through the canvas by an amount derived from the given <MouseWheel> event.

        :param event: the <MouseWheel> event that triggered the method call.
        :type event: `tkinter Event`
        """
        x, y = self.winfo_pointerxy()
        w = self.winfo_containing(x, y)

        # differentiate between Treeview and the rest
        # if mouse is on Treeview or its scrollbar, scroll w.r.t. the Treeview
        # otherwise scroll w.r.t. the whole window/canvas
        if w == self._tree or w == self._tree_scroll:
            widget = self._tree
        else:
            widget = self._results_canvas

        # print("Scrolling RESULTS SCREEN........................")
        if self._OS == 'Linux':
            if event.num == 4:
                widget.yview_scroll(-1, "units")
            elif event.num == 5:
                widget.yview_scroll(1, "units")
        else:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_close_safe(self):
        """Safely close the window by making sure to call :meth:`self._unbind_mousewheel()` before unstacking."""
        # self._pl_algo.clean_up() # do any final clean ups required by the pl algorithm class
        # ^ now handled by Experiment.run()
        self._unbind_mousewheel(None)
        ws.on_close(self, self._parent_window)

    def _print_params(self, frame, params_string):
        """Neatly display the parameters of an algorithm or method in a given frame widget.

        :param frame: the Frame widget to display the parameters in.
        :type frame: `tkinter.Frame`
        :param params_string: a string specifying the parameters. Parameters in the string is separated by a
            semicolon character (';'). Each parameter is defined by its name and value which are separated
            by a colon character (':').
        :type params_string: str
        """
        params_string = params_string.strip("{")  # trim opening bracket
        params_string = params_string.strip("}")  # trim closing bracket
        params = params_string.split(";")  # extract list of params
        r = 0
        for param in params:
            print(param)
            # extract the name/key and value of the given parameter
            key, value = param.split(":")
            # trim whitespaces from left & right
            key = key.strip()
            value = value.strip()
            # add to grid
            tk.Label(frame, text=text.real_param_name(str(key))+":").grid(row=r, column=0, sticky='w')
            tk.Label(frame, text=text.real_type_name(str(value))).grid(row=r, column=1, sticky='w')
            r += 1

    def _save_exp_log_with_dialog(self, timestamp):
        """Open a file dialog window (GUI) and save a log of the experiment to file at the path indicated by the user.

        The model file must be a Comma Separated Value (CSV)-type file with the extension '.csv'.

        :param timestamp: the timestamp to be included in the file name.
        :type timestamp: float
        :return: specifies whether or not the file was successfully saved.
        :rtype: bool
        """
        init_dir = os.path.join(ROOT_PATH, "logs")
        init_filename = "exp_" + str(timestamp) + ".csv"
        fpath = filedialog.asksaveasfilename(parent=self, initialdir=init_dir,
                                             title="Save experiment report", initialfile=init_filename,
                                             defaultextension=".csv", filetypes=(("CSV files", "*.csv"),))
        if fpath != "":
            # only allow .csv files!
            self._exp.save_exp_log(timestamp, path=fpath)
            # show info box confirmation
            messagebox.showinfo("Save success", "The experiment report was successfully saved!", parent=self)
            return True
        else:
            print("Cancelled save experiment log/report.")
            return False
