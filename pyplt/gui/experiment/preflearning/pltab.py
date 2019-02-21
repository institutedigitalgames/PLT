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
from tkinter import messagebox
from threading import Thread
from queue import Queue

from pyplt.autoencoder import Autoencoder
from pyplt.evaluation.cross_validation import KFoldCrossValidation
from pyplt.gui.experiment.preflearning.ranksvm_menu import RankSVMMenu
from pyplt.gui.experiment.preflearning.backprop_menu import BackpropMenu
from pyplt.util.enums import PLAlgo, EvaluatorType, FSMethod, KernelType
from pyplt.gui.experiment.preflearning.evaluator_menus import HoldoutMenu, KFCVMenu
from pyplt.gui.experiment.progresswindow import ProgressWindow
from pyplt.gui.util import colours, supported_methods
from pyplt.gui.util.tab_locking import LockableTab
from pyplt.evaluation.holdout import HoldOut
from pyplt.exceptions import NoFeaturesError, InvalidParameterValueException, NoRanksDerivedError, \
    NormalizationValueError, IncompatibleFoldIndicesException, MissingManualFoldsException, \
    AutoencoderNormalizationValueError
from pyplt.experiment import Experiment
from pyplt.fsmethods.sfs import SFS
from pyplt.plalgorithms.backprop_tf import BackpropagationTF
from pyplt.plalgorithms.ranksvm import RankSVM
from pyplt.util import AbortFlag


class PLTab(LockableTab):
    """GUI tab for the preference learning and evaluation stage of setting up an experiment.

    Extends the class :class:`pyplt.gui.util.tab_locking.LockableTab` which, in turn, extends the
    `tkinter.Frame` class.
    """

    def __init__(self, parent, parent_window, files_tab, preproc_tab, fs_tab):
        """Initializes the `PLTab` object.

        :param parent: the parent widget of this tab (frame) widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this tab (frame) widget.
        :type parent_window: `tkinter.Toplevel`
        :param files_tab: the `Load Data` tab.
        :type files_tab: :class:`pyplt.gui.experiment.dataset.loading.DataLoadingTab`
        :param preproc_tab: the `Preprocessing` tab.
        :type preproc_tab: :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingTab`
        :param fs_tab: the `Feature Selection` tab.
        :type fs_tab: :class:`pyplt.gui.experiment.featureselection.featselectiontab.FeatureSelectionTab`
        """
        self._files_tab = files_tab
        self._preproc_tab = preproc_tab
        self._fs_tab = fs_tab

        self._parent = parent
        self._parent_window = parent_window
        self._frame = None
        LockableTab.__init__(self, self._parent, self._parent_window)

    def get_normal_frame(self):
        """Return a `PLFrame` widget for when the tab is in the 'unlocked' state.

        The `PLFrame` widget is instantiated only once on the first occasion that the tab is 'unlocked'.

        :return: the `PLFrame` widget that is visible whenever the tab is in the 'unlocked' state.
        :rtype: :class:`pyplt.gui.experiment.preflearning.pltab.PLFrame`
        """
        if self._frame is None:
            self._frame = PLFrame(self.get_base_frame(), self._parent_window,
                                  self._files_tab, self._preproc_tab, self._fs_tab)
        return self._frame

    def get_pl_algorithm(self):
        """Get the preference learning algorithm type chosen by the user via the `PLFrame`.

        :return: the preference learning algorithm type chosen by the user.
        :rtype: :class:`pyplt.util.enums.PLAlgo`
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_algorithm()

    def get_pl_algorithm_params(self):
        """Get the parameters of the preference learning algorithm chosen by the user (if applicable).

        :return: the parameters of the preference learning algorithm chosen by the user (if applicable).
        :rtype: list
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_algorithm_params()

    def get_evaluator(self):
        """Get the evaluation method type chosen by the user via the `PLFrame`.

        :return: the evaluation method type chosen by the user.
        :rtype: :class:`pyplt.util.enums.EvaluatorType`
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_evaluator()

    def get_evaluator_params(self):
        """Get the parameters of the evaluation method chosen by the user (if applicable).

        :return: the parameters of the evaluation method chosen by the user (if applicable).
        :rtype: list
        """
        if self._frame is None:
            return None
        else:
            return self._frame.get_evaluator_params()

    def run_experiment(self):
        """Call the method which triggers the execution of the experiment."""
        if self._frame is not None:
            self._frame.run_exp()


class PLFrame(tk.Frame):
    """Frame widget that is visible whenever the `Preference Learning` tab is in the 'unlocked' state.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent, parent_window, files_tab, preproc_tab, fs_tab):
        """Initializes the frame widget and its contents.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter widget`
        :param parent_window: the window which will contain this frame widget.
        :type parent_window: `tkinter.Toplevel`
        :param files_tab: the `Load Data` tab.
        :type files_tab: :class:`pyplt.gui.experiment.dataset.loading.DataLoadingTab`
        :param preproc_tab: the `Preprocessing` tab.
        :type preproc_tab: :class:`pyplt.gui.experiment.preprocessing.preproctab.PreProcessingTab`
        :param fs_tab: the `Feature Selection` tab.
        :type fs_tab: :class:`pyplt.gui.experiment.featureselection.featselectiontab.FeatureSelectionTab`
        """
        self._files_tab = files_tab
        self._preproc_tab = preproc_tab
        self._fs_tab = fs_tab

        self._parent = parent
        self._parent_window = parent_window
        self._algorithm_name = tk.StringVar(value=PLAlgo.RANKSVM.name)
        self._algorithm_sub_menus = {}
        self._evaluator_name = tk.StringVar(value=EvaluatorType.NONE.name)
        self._evaluator_sub_menus = {}

        self._OS = platform.system()

        tk.Frame.__init__(self, parent)  # , width=1000, height=500

        self._main_canvas = tk.Canvas(self)  # , background='yellow'
        self._main_canvas.bind("<Configure>", self._on_resize)
        self._main_canvas.bind('<Enter>', self._bind_mousewheel)
        self._main_canvas.bind('<Leave>', self._unbind_mousewheel)
        self._canvas_width = self._main_canvas.winfo_reqwidth()
        self._canvas_height = self._main_canvas.winfo_reqheight()
        self._main_sub_frame = tk.Frame(self._main_canvas)  # , background='green'
        self._main_sub_sub_frame = tk.Frame(self._main_sub_frame)

        self._algo_menu = tk.Frame(self._main_sub_sub_frame, bg=colours.PL_OUTER)
        self._algo_menu.pack(fill=tk.X, expand=True)
        a_select = tk.Frame(self._algo_menu, bg=colours.PL_OUTER)
        a_select.pack(padx=75, pady=10)
        options = [key.name for key in supported_methods.supported_algorithms.keys()]
        tk.Label(a_select, text="Choose Preference Learning Algorithm",
                 bg=colours.PL_OUTER, fg='white').grid(row=0, column=0)
        ttk.OptionMenu(a_select, self._algorithm_name, options[0], *options,
                       command=lambda sel: self._update_algo_menu(sel),
                       style='PL.PLT.TMenubutton').grid(row=0, column=1)
        # force algo menu update for default algo
        self._update_algo_menu(self._algorithm_name.get())

        self._evaluator_menu = tk.Frame(self._main_sub_sub_frame, bg=colours.EVAL_OUTER)
        self._evaluator_menu.pack(fill=tk.X, expand=True)  # would remain empty if fs method is not a wrapper method
        # show algorithm selection menu
        e_select = tk.Frame(self._evaluator_menu, bg=colours.EVAL_OUTER)
        e_select.pack(padx=75, pady=10)
        tk.Label(e_select, text="Choose Evaluation Method", bg=colours.EVAL_OUTER, fg='white').grid(row=0, column=0)
        options = [key.name for key in supported_methods.supported_evaluation_methods.keys()]
        ttk.OptionMenu(e_select, self._evaluator_name, options[0], *options,
                       command=lambda _: self._update_eval_menu(), style='Eval.PLT.TMenubutton').grid(row=0, column=1)

        # add scrollbars
        v_scroll = ttk.Scrollbar(self, orient="vertical", command=self._main_canvas.yview,
                                 style="PLT.Vertical.TScrollbar")  # self._results_frame
        v_scroll.pack(side='right', fill='y')
        self._main_canvas.configure(yscrollcommand=v_scroll.set)
        h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self._main_canvas.xview,
                                 style="PLT.Horizontal.TScrollbar")  # self._results_frame
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
        # print("__ on_config called __ ")
        self._main_canvas.configure(scrollregion=(0, 0, self._main_sub_frame.winfo_reqwidth(),
                                                  self._main_sub_frame.winfo_reqheight()))

    def get_algorithm(self):
        """Get the preference learning algorithm type chosen by the user.

        :return: the preference learning algorithm chosen by the user.
        :rtype: :class:`pyplt.util.enums.PLAlgo`
        """
        return PLAlgo[self._algorithm_name.get()]

    def get_algorithm_params(self):
        """Get the parameters of the preference learning algorithm chosen by the user (if applicable).

        :return: the parameters of the preference learning algorithm chosen by the user (if applicable).
        :rtype: list
        """
        try:
            algo_name = self._algorithm_name.get()
            return self._algorithm_sub_menus[algo_name].get_params()
        except AttributeError:  # in case the given algo menu does not have get_params() method
            return None

    def get_evaluator(self):
        """Get the evaluation method type chosen by the user.

        :return: the evaluation method type chosen by the user.
        :rtype: :class:`pyplt.util.enums.EvaluatorType`
        """
        return EvaluatorType[self._evaluator_name.get()]

    def get_evaluator_params(self):
        """Get the parameters of the evaluation method chosen by the user (if applicable).

        :return: the parameters of the evaluation method chosen by the user (if applicable).
        :rtype: list
        """
        try:
            eval_name = self._evaluator_name.get()
            if eval_name != EvaluatorType.NONE.name:
                return self._evaluator_sub_menus[eval_name].get_params()
        except AttributeError:  # in case the given eval menu does not have get_params() method
            return None
        return None  # in case NONE is chosen

    def _update_algo_menu(self, *args):
        """Display the algorithm parameter menu corresponding to the algorithm chosen by the user.

        The menu allows the user to specify the parameters of the preference learning algorithm (if applicable).

        :param args: optional parameter to pass the user selection from the algorithm drop-down menu.
        """
        # this method is exactly the same as that in fstab!  # TODO: keep up to date with corresponding method in fstab
        sel = args[0]
        sel_type = PLAlgo[sel]

        exists = False
        for algo_name, algo_menu in self._algorithm_sub_menus.items():
            if algo_name == sel:
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
            self._algorithm_sub_menus[sel] = new_menu

        # force updates for canvas and scrollbar stuff
        self.update_idletasks()
        self._on_resize(None)
        self._on_canvas_config(None)

    def _update_eval_menu(self):
        """Display the evaluation method parameter menu corresponding to the evaluation method chosen by the user.

        The menu allows the user to specify the parameters of the evaluation method (if applicable).
        """
        # this method is exactly the same as that in fstab!!  # TODO: keep up to date with corresponding method in fstab
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

    def run_exp(self):
        """Trigger the execution of the experiment in a separate thread from the main (GUI) thread.

        A :class:`pyplt.gui.experiment.progresswindow.ProgressWindow` widget is also initialized to keep a
        progress log and progress bar of the experiment execution process.

        Threading is carried out using the `threading.Thread` class.
        """
        # wait_variable
        data = self._files_tab.get_data()
        mdm, memory = self._files_tab.get_rank_derivation_params()
        feat_include_settings = self._preproc_tab.get_include_settings()
        feat_norm_settings = self._preproc_tab.get_norm_settings()
        # shuffle settings
        shuffle, random_state = self._preproc_tab.get_shuffle_settings()
        # feature selection
        fs_method = self._fs_tab.get_fs_method()
        fs_params = self._fs_tab.get_fs_method_params()
        fs_algo = self._fs_tab.get_fs_algorithm()
        fs_algo_params = self._fs_tab.get_fs_algorithm_params()
        fs_eval = self._fs_tab.get_fs_evaluator()
        fs_eval_params = self._fs_tab.get_fs_evaluator_params()
        # preference learning
        pl_algo = self.get_algorithm()
        pl_algo_params = self.get_algorithm_params()
        pl_eval = self.get_evaluator()
        pl_eval_params = self.get_evaluator_params()

        obj_path = self._files_tab.get_objects_path()
        ranks_path = self._files_tab.get_ranks_path()
        single_path = self._files_tab.get_single_path()

        file_paths = [obj_path, ranks_path, single_path]

        print("Running Experiment...")
        print("---------------- DATA ----------------")
        if isinstance(data, tuple):
            _objects, _ranks = data
            print("Objects:")
            print(_objects)
            print("Ranks:")
            print(_ranks)
            print("Details:")
            print(str(len(_objects)) + " objects.")
            print(str(len(_ranks)) + " ranks.")
        else:
            print("Data:")
            print(data)
            print("Details: ")
            print(str(len(data)) + " samples.")
            print("Minimum Distance Margin: " + str(mdm))
            print("Memory: " + str(memory))
        print("---------------- FS ----------------")
        # fs
        print("FS Method: " + str(fs_method))
        print("FS Params:")
        print(fs_params)
        print("FS algorithm: " + str(fs_algo))
        print("FS algorithm params:")
        print(fs_algo_params)
        print("FS evaluator: " + str(fs_eval))
        print("FS evaluator params:")
        print(fs_eval_params)
        print("---------------- PL ----------------")
        # pl
        print("PL Algorithm: " + str(pl_algo))
        print("PL Algorithm params:")
        print(pl_algo_params)
        print("PL evaluator: " + str(pl_eval))
        print("PL evaluator params:")
        print(pl_eval_params)
        print("--------------------------------------")

        auto_extract_enabled = self._preproc_tab.auto_extract_enabled()

        use_feats = [key for key, val in feat_include_settings.items() if val.get()]  # if val.get() is True
        # ^ feature ids not names/labels

        # +1 each feat ID and add a 0 to use_feats to make up for ID column which hasn't yet been removed
        # no need if autoencoder is enabled bc use_feats are only used for input size
        if not auto_extract_enabled:
            new_use_feats = [f+1 for f in use_feats]
            use_feats = [0] + new_use_feats

        # Get actual objects and set up Experiment variables
        exp = Experiment()
        if isinstance(data, tuple):
            _objects, _ranks = data
            if auto_extract_enabled:
                # N.B. ignore use_feats if autoencoder is enabled bc we need all original features
                # use only for input size
                use_feats = list(_objects.columns)[:-1]  # exclude last column to account for ID column
            else:
                _objects = _objects.iloc[:, use_feats]  # include/exclude features
                # ^ ID col still included bc we took care of it above
            exp._set_objects(_objects, has_ids=True)
            # ^ has_ids=True bc _load_data() adds ID column if there isn't already
            exp._set_ranks(_ranks, has_ids=True)  # has_ids=True bc _load_data() adds ID column if there isn't already
        else:
            last_col_id = len(data.columns)-1
            use_feats.append(last_col_id)  # include last column (ratings)
            if auto_extract_enabled:
                # N.B. ignore use_feats if autoencoder is enabled bc we need all original features
                # use only for input size
                use_feats = list(data.columns)[:-2]  # exclude last 2 columns to account for ID column and ratings col
            else:
                data = data.iloc[:, use_feats]  # include/exclude features
                # ^ ID col still included bc we took care of it above
            exp._set_single_data(data, has_ids=True)
            # ^ has_ids=True bc _load_data() adds ID column if there isn't already
            exp.set_rank_derivation_params(mdm=mdm, memory=memory)

        if auto_extract_enabled:
            # set autoencoder
            ae_menu = self._preproc_tab.get_autoencoder_menu()
            input_size = len(use_feats)
            code_size = ae_menu.get_code_size()
            encoder_top = ae_menu.get_encoder_neurons()
            # encoder_actf = ae_menu.get_encoder_actfs()
            decoder_top = ae_menu.get_decoder_neurons()
            # decoder_actf = ae_menu.get_decoder_actfs()
            lr = ae_menu.get_learn_rate()
            error_thresh = ae_menu.get_error_thresh()
            e = ae_menu.get_epochs()
            ae = Autoencoder(input_size, code_size, encoder_top, decoder_top, lr, error_thresh, e)
            exp.set_autoencoder(ae)

        # set normalization methods (but first convert to dict of NormalizationType-names values
        # rather than StringVar-of-NormalizationType-names values)
        feat_norm_settings = {key: val.get() for key, val in feat_norm_settings.items()}
        exp._set_norm_settings(feat_norm_settings)

        if not (fs_method == FSMethod.NONE):
            fs_method_obj = self._get_fs_instance(fs_method, fs_params)
            exp.set_fs_method(fs_method_obj)
            fs_m_params = fs_method_obj.get_params_string()
        else:
            fs_m_params = None

        if fs_algo is not None:
            print(fs_algo)
            try:
                fs_algo_obj = self._get_algo_instance(fs_algo, fs_algo_params)
            except InvalidParameterValueException as error:
                messagebox.showerror("Feature Selection: " + error.get_summary(),
                                     error.get_message(), parent=self._parent_window)
                return
            exp.set_fs_algorithm(fs_algo_obj)
            fs_a_params = fs_algo_obj.get_params_string()
        else:
            fs_a_params = None

        if fs_eval is not None and not (fs_eval == EvaluatorType.NONE):
            try:
                fs_eval_obj = self._get_eval_instance(fs_eval, fs_eval_params)
            except (InvalidParameterValueException, MissingManualFoldsException) as ex:
                messagebox.showerror("Feature Selection: " + ex.get_summary(),
                                     ex.get_message(), parent=self._parent_window)
                return  # stop before starting experiment
            exp.set_fs_evaluator(fs_eval_obj)
            fs_e_params = fs_eval_obj.get_params_string()
        else:
            fs_e_params = None

        # we always have a pl algorithm so no need to check for None
        try:
            pl_algo_obj = self._get_algo_instance(pl_algo, pl_algo_params)
        except InvalidParameterValueException as error:
            messagebox.showerror("Preference Learning: " + error.get_summary(),
                                 error.get_message(), parent=self._parent_window)
            return
        exp.set_pl_algorithm(pl_algo_obj)
        pl_a_params = pl_algo_obj.get_params_string()

        if not (pl_eval == EvaluatorType.NONE):
            try:
                pl_eval_obj = self._get_eval_instance(pl_eval, pl_eval_params)
            except (InvalidParameterValueException, MissingManualFoldsException) as ex:
                messagebox.showerror("Preference Learning: " + ex.get_summary(),
                                     ex.get_message(), parent=self._parent_window)
                return  # stop before starting experiment
            exp.set_pl_evaluator(pl_eval_obj)
            pl_e_params = pl_eval_obj.get_params_string()
        else:
            pl_e_params = None

        q = Queue()

        exec_stopper = AbortFlag()

        pw = ProgressWindow(self._parent_window, q, exec_stopper)  # instantiate a progress window

        te = Thread(target=self._run_exp_, args=(exp, shuffle, random_state, pw, exec_stopper, data, file_paths,
                                                 fs_method, fs_m_params, fs_algo, fs_a_params, fs_eval, fs_e_params,
                                                 pl_algo, pl_a_params, pl_eval, pl_e_params))
        pw.set_exec_thread(te)
        te.start()

    @staticmethod
    def _get_fs_instance(method, method_params=None):
        """Create an instance of the feature selection method class represented by the given enum constant.

        Each enumerated constant of type :class:`pyplt.util.enums.FSMethod` corresponds to a class of type (extending)
        :class:`pyplt.fsmethods.base.FeatureSelectionMethod`.

        The instance is initialized with the method parameter values specified by the user in the `Feature Selection`
        tab (if applicable).

        :param method: the feature selection method type (enum).
        :type method: :class:`pyplt.util.enums.FSMethod`
        :param method_params: the method parameter values specified by the user in the `Feature Selection` tab
            (if applicable) (default None). The keys of the dict should match the keywords of the arguments
            that would be passed to the corresponding :class:`pyplt.fsmethods.base.FeatureSelectionMethod` constructor.
        :type method_params: dict or None, optional
        :return: an instance of the class corresponding to the given feature selection method.
        :rtype: :class:`pyplt.fsmethods.base.FeatureSelectionMethod`
        """
        return supported_methods.get_fs_method_instance(method, method_params)

    @staticmethod
    def _get_algo_instance(algo, algo_params=None):
        """Create an instance of the preference learning algorithm class represented by the given enum constant.

        Each enumerated constant of type :class:`pyplt.util.enums.PLAlgo` corresponds to a class of type (extending)
        :class:`pyplt.plalgorithms.base.PLAlgorithm`.

        The instance is initialized with the algorithm parameter values specified by the user in the
        `Preference Learning` tab (if applicable).

        :param algo: the algorithm type (enum).
        :type algo: :class:`pyplt.util.enums.PLAlgo`
        :param algo_params: the algorithm parameter values specified by the user in the `Preference Learning` tab
            (if applicable) (default None). The keys of the dict should match the keywords of the arguments
            that would be passed to the corresponding :class:`pyplt.plalgorithms.base.PLAlgorithm` constructor.
            For example, for the `Backpropagation` algorithm the dict should contain the following items:

            * ann_topology: the topology of the neurons in the network
            * learn_rate: the learning rate
            * error_threshold: the error threshold
            * epochs: the number of epochs
            * activation_functions: the activation functions for each neuron layer in the network

            On the other hand, for the `RankSVM` algorithm the dict should contain the following items:

            * kernel: the kernel name
            * gamma: the gamma kernel parameter value
            * degree: the degree kernel parameter value

        :type algo_params: dict or None, optional
        :return: an instance of the class corresponding to the given algorithm.
        :rtype: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        :raises InvalidParameterValueException: if the user attempted to use a value smaller or equal to 0.0
            for the `gamma` parameter of the `RankSVM` algorithm.
        """
        try:
            # get algorithm object and instantiate with these params (converted from dict to kwargs)
            return supported_methods.get_algorithm_instance(algo, algo_params)
        except InvalidParameterValueException as exception:
            raise exception  # raise again for upper method to deal with!

    @staticmethod
    def _get_eval_instance(eval_, eval_params=None):
        """Create an instance of the evaluation method class represented by the given enum constant.

        Each enumerated constant of type :class:`pyplt.util.enums.EvaluatorType` corresponds to a class of type
        (extending) :class:`pyplt.evaluation.base.Evaluator`.

        The instance is initialized with the method parameter values specified by the user in the `Preference Learning`
        tab (if applicable).

        :param eval_: the evaluation method type (enum).
        :type eval_: :class:`pyplt.util.enums.EvaluatorType`
        :param eval_params: the method parameter values specified by the user in the `Preference Learning` tab
            (if applicable) (default None). The keys of the dict should match the keywords of the arguments
            that would be passed to the corresponding :class:`pyplt.evaluation.base.Evaluator` constructor.
            For example, for the `Holdout` method, the dict should contain the following items:

            * test_proportion: a float specifying the proportion of data to be used as training data (the rest
              is to be used as test data) or None

            On the other hand, for the `KFoldCrossValidation` method, the dict should contain the following items:

            * k: the number of folds to uniformly split the data into when using the automatic approach or None
            * test_folds: an array specifying the fold index for each sample in the dataset when using
              the manual approach or None

        :type eval_params: dict or None, optional
        :return: an instance of the class corresponding to the given evaluation method.
        :rtype: :class:`pyplt.evaluation.base.Evaluator`
        :raises InvalidParameterValueException: if the user attempts to use a value smaller than 2 for
            the `k` parameter of K-Fold Cross Validation.
        :raises MissingManualFoldsException: if the user chooses to specify folds manually for cross validation but
            fails to load the required file containing the fold IDs.
        """
        try:
            # get algorithm object and instantiate with these params (converted from dict to kwargs)
            return supported_methods.get_eval_method_instance(eval_, eval_params)
        except MissingManualFoldsException as exception:
            raise exception  # raise again for upper method to deal with!

    def _on_resize(self, event):
        """Resize the canvas widget according to the user's specification via the mouse.

        This method is called whenever a <Configure> event occurs with respect to :attr:`self._main_canvas`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("_on_resize() has been called!")
        if event is not None:  # otherwise use latest values of self._canvas_width and self._canvas_height
            # for forcing updates for canvas/scrollbars
            self._canvas_width = event.width
            self._canvas_height = event.height
        # print("event/canvas width = " + str(self._canvas_width))
        # print("event/canvas height = " + str(self._canvas_height))
        try:
            if self._canvas_width > self._main_sub_frame.winfo_reqwidth():
                self._main_canvas.itemconfig(self.c_win, width=self._canvas_width)
            else:
                self._main_canvas.itemconfig(self.c_win, width=self._main_sub_frame.winfo_reqwidth())
            if self._canvas_height > self._main_sub_frame.winfo_reqheight():
                self._main_canvas.itemconfig(self.c_win, height=self._canvas_height)
            else:
                self._main_canvas.itemconfig(self.c_win, height=self._main_sub_frame.winfo_reqheight())
        except AttributeError:
            print("Canvas contents have not been drawn yet.")

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
        # print("Scrolling RESULTS SCREEN........................")
        if self._OS == 'Linux':
            if event.num == 4:
                self._main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._main_canvas.yview_scroll(1, "units")
        else:
            self._main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _run_exp_(self, exp, shuffle, random_state, pw, exec_stopper, data, file_paths,
                  fs_method, fs_m_params, fs_algo, fs_a_params,
                  fs_eval, fs_e_params, pl_algo, pl_a_params, pl_eval, pl_e_params):
        """Begin execution of the experiment and pass on experiment details and results to the progress window.

        :param exp: the experiment to be run.
        :type exp: :class:`pyplt.experiment.Experiment`
        :param shuffle: specifies whether or not to shuffle the data (samples in the case of the single file format;
            ranks in the case of the dual file format) at the start of executing the experiment; i.e., prior to fold
            splitting, rank derivation, and normalization (if applicable) (default False).
        :type shuffle: bool, optional
        :param random_state: seed for the random number generator (if int), or numpy RandomState object, used to
            shuffle the dataset if `shuffle` is True (default None).
        :type random_state: int or `numpy.random.RandomState`, optional
        :param pw: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution.
        :type pw: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`
        :param exec_stopper: an abort flag object used to abort the execution before completion
            if so instructed by the user via the progress window.
        :type exec_stopper: :class:`pyplt.util.AbortFlag`
        :param data: the data used in the experiment to be passed on to the progress window. If the single file
            format is used, a single `pandas.DataFrame` containing the data should be passed. If the dual file
            format is used, a tuple containing both the objects and ranks (each a `pandas.DataFrame`) should be passed.
        :type data: `pandas.DataFrame` or tuple of `pandas.DataFrame` (size 2)
        :param file_paths: the file paths of the data used in the experiment to be passed on to the progress window.
        :type file_paths: list of str
        :param fs_method: the feature selection method used in the experiment to be passed on to the progress window
            (if applicable).
        :type fs_method: :class:`pyplt.util.enums.FSMethod`
        :param fs_m_params: the feature selection method parameters used in the experiment to be passed on to the
            progress window (if applicable).
        :type fs_m_params: str or None
        :param fs_algo: the feature selection algorithm used in the experiment to be passed on to the progress window
            (if applicable).
        :type fs_algo: :class:`pyplt.util.enums.PLAlgo` or None
        :param fs_a_params: the feature selection algorithm parameters used in the experiment to be passed on to the
            progress window (if applicable).
        :type fs_a_params: str or None
        :param fs_eval: the feature selection evaluation method used in the experiment to be passed on to the progress
            window (if applicable).
        :type fs_eval: :class:`pyplt.util.enums.EvaluatorType`
        :param fs_e_params: the feature selection evaluation method parameters used in the experiment to be passed on
            to the progress window (if applicable).
        :type fs_e_params: str or None
        :param pl_algo: the preference learning algorithm used in the experiment to be passed on to the progress window.
        :type pl_algo: :class:`pyplt.util.enums.PLAlgo`
        :param pl_a_params: the preference learning algorithm parameters used in the experiment to be passed on to the
            progress window (if applicable).
        :type pl_a_params: str or None
        :param pl_eval: the evaluation method used in the experiment to be passed on to the progress window
            (if applicable).
        :type pl_eval: :class:`pyplt.util.enums.EvaluatorType`
        :param pl_e_params: the evaluation method parameters used in the experiment to be passed on to the progress
            window (if applicable).
        :type pl_e_params: str or None
        :return: None -- if aborted before completion by `exec_stopper`.
        """
        # Actually run the experiment
        try:
            eval_metrics, fold_metrics = exp.run(shuffle=shuffle, random_state=random_state,
                                                 debug=True, progress_window=pw, exec_stopper=exec_stopper)
        except MemoryError as ex:
            pw.put(ex.__class__.__name__)  # let progress window know of the error so it can abort properly
            return  # terminate experiment thread by returning
        except (NoFeaturesError, NoRanksDerivedError, InvalidParameterValueException, NormalizationValueError,
                IncompatibleFoldIndicesException, AutoencoderNormalizationValueError) as ex:
            pw.put(str(ex.__class__.__name__) + "??" + ex.get_summary() + "??" + ex.get_message())
            # ^ let progress window know of the error so it can abort properly
            return  # terminate experiment thread by returning

        if eval_metrics is None:  # check if experiment was aborted
            # abort execution!
            print("Abort complete.")
            return

        # get the features selected by fs (if applicable)
        sel_feats = exp.get_features()

        preproc_info = [self._preproc_tab.get_include_settings(),  # TODO: change from f_names to f_ids in result screen
                        self._preproc_tab.get_norm_settings()]

        pw.put("DONE")

        d = exp.get_data()
        if isinstance(d, tuple):
            n_objects = d[0].shape[0]
            n_ranks = d[1].shape[0]
        else:
            n_objects = d.shape[0]
            n_ranks = 'N/A'

        # TODO: do something with data parameter...

        pw.done(experiment=exp,
                time_info=exp.get_time_meta_data(),
                data_info=[n_objects, n_ranks, file_paths], preproc_info=preproc_info,
                fs_info=[fs_method, fs_m_params, sel_feats],
                fs_algo_info=[fs_algo, fs_a_params],
                fs_eval_info=[fs_eval, fs_e_params],
                pl_algo_info=[pl_algo, pl_a_params],
                pl_eval_info=[pl_eval, pl_e_params],
                eval_metrics=eval_metrics,
                fold_metrics=fold_metrics,
                shuffle_info=[shuffle, random_state])
