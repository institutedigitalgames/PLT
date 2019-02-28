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
from tkinter import ttk, font
# from tkinter import messagebox
from threading import Thread
from queue import Queue
import os
import numpy as np
import platform

from pyplt import ROOT_PATH
import pyplt.gui.util.windowstacking as ws
from pyplt.autoencoder import Autoencoder
from pyplt.evaluation.holdout import HoldOut
from pyplt.exceptions import NoFeaturesError, NoRanksDerivedError, InvalidParameterValueException, \
    NormalizationValueError, IncompatibleFoldIndicesException, AutoencoderNormalizationValueError
from pyplt.experiment import Experiment
from pyplt.gui.experiment.preprocessing.data_compression import AutoencoderSettingsBeginner
from pyplt.util.enums import NormalizationType, ActivationType
from pyplt.fsmethods.sfs import SFS
from pyplt.gui.experiment.dataset.loading import DataLoadingTab
from pyplt.util.enums import PLAlgo, FSMethod, EvaluatorType, KernelType
from pyplt.gui.experiment.progresswindow import ProgressWindow
from pyplt.gui.util import colours, supported_methods
from pyplt.gui.util.help import BeginnerStep1HelpDialog, BeginnerStep2HelpDialog, \
    BeginnerStep3HelpDialog, BeginnerStep4HelpDialog, BeginnerStep5HelpDialog
from pyplt.plalgorithms.backprop_tf import BackpropagationTF
from pyplt.plalgorithms.ranksvm import RankSVM
from pyplt.util import AbortFlag


class BeginnerMenu(tk.Toplevel):
    """GUI window containing a simplified PLT experiment set-up menu for beginners.

    Extends the class `tkinter.Toplevel`.
    """

    def __init__(self, parent):
        """Initializes the `BeginnerMenu` object.

        :param parent: the parent window on top of which this `tkinter.Toplevel` window will be stacked.
        :type parent: `tkinter.Toplevel`
        """
        self._apply_fs = tk.BooleanVar(value=False)
        self._apply_ae = tk.BooleanVar(value=False)
        self._autoencoder_menu = None
        self._algorithm = tk.StringVar(value=PLAlgo.RANKSVM.name)
        self._include_settings = {}
        self._norm_settings = {}

        self._parent = parent

        tk.Toplevel.__init__(self, parent, height=250)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, parent, ws.Mode.WITH_CLOSE)
        ws.stack_window(self, parent)

        self.title("Experiment Setup (Beginner)")

        # self._main_frame = tk.Frame(self)
        # self._main_frame.pack()

        self._OS = platform.system()

        self._main_canvas = tk.Canvas(self, width=500, height=500)  # , background='yellow'
        self._main_canvas.bind("<Configure>", self._on_resize)
        self._main_canvas.bind('<Enter>', self._bind_mousewheel)
        self._main_canvas.bind('<Leave>', self._unbind_mousewheel)
        self._canvas_width = self._main_canvas.winfo_reqwidth()
        self._canvas_height = self._main_canvas.winfo_reqheight()
        self._main_sub_frame = tk.Frame(self._main_canvas)  # , background='green'
        self._main_sub_sub_frame = tk.Frame(self._main_sub_frame)

        ebrima_big = font.Font(family='Ebrima', size=12, weight=font.BOLD)
        # ebrima_small = font.Font(family='Ebrima', size=10, weight=font.NORMAL)

        self._run_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/run_128_30_02.png"))
        self._help_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/help.png"))

        # 1. LOAD DATA

        self._data_frame = tk.Frame(self._main_sub_sub_frame, bd=2, relief='groove')
        self._data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        title_frame = tk.Frame(self._data_frame, bg=colours.NAV_BAR)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(title_frame, text="Step 1: Load Data", font=ebrima_big, fg='white',
                 bg=colours.NAV_BAR).pack(padx=10, pady=5, side=tk.LEFT)
        help_btn = tk.Button(title_frame, command=lambda s=1: self._help_dialog(s), image=self._help_img,
                             relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                             highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                             activebackground=colours.NAV_BAR)
        help_btn.pack(padx=10, pady=5, side=tk.RIGHT)
        self._load_frame = DataLoadingTab(self._data_frame, self)
        self._load_frame.pack(side=tk.BOTTOM)

        # 2. PREPROCESSING -- FEATURE EXTRACTION

        self._others_frame = tk.Frame(self._main_sub_sub_frame)
        self._others_frame.pack(fill=tk.BOTH, expand=True)

        self._ae_frame = tk.Frame(self._others_frame, bd=2, relief='groove')
        self._ae_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        title_frame = tk.Frame(self._ae_frame, bg=colours.NAV_BAR)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(title_frame, text="Step 2: Preprocessing", font=ebrima_big, fg='white',
                 bg=colours.NAV_BAR).pack(padx=10, pady=5, side=tk.LEFT)
        help_btn = tk.Button(title_frame, command=lambda s=2: self._help_dialog(s), image=self._help_img,
                             relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                             highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                             activebackground=colours.NAV_BAR)
        help_btn.pack(padx=10, pady=5, side=tk.RIGHT)

        self._sub_ae_frame = tk.Frame(self._ae_frame)
        self._sub_ae_frame.pack(pady=10, side=tk.BOTTOM)

        self._sub_sub_ae_frame = tk.Frame(self._sub_ae_frame)
        self._sub_sub_ae_frame.pack()

        tk.Label(self._sub_sub_ae_frame, text="Enable Automatic Feature Extraction?").grid(row=0, column=0)
        ttk.Checkbutton(self._sub_sub_ae_frame, variable=self._apply_ae, onvalue=True, offvalue=False,
                        style="PLT.TCheckbutton").grid(row=0, column=1, padx=(10, 0))
        self._apply_ae.trace('w', self._toggle_autoencoder_menu)

        # 3. FEATURE SELECTION

        self._fs_frame = tk.Frame(self._others_frame, bd=2, relief='groove')
        self._fs_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        title_frame = tk.Frame(self._fs_frame, bg=colours.NAV_BAR)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(title_frame, text="Step 3: Feature Selection", font=ebrima_big, fg='white',
                 bg=colours.NAV_BAR).pack(padx=10, pady=5, side=tk.LEFT)
        help_btn = tk.Button(title_frame, command=lambda s=3: self._help_dialog(s), image=self._help_img,
                             relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                             highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                             activebackground=colours.NAV_BAR)
        help_btn.pack(padx=10, pady=5, side=tk.RIGHT)

        sub_fs_frame = tk.Frame(self._fs_frame)
        sub_fs_frame.pack(pady=10, side=tk.BOTTOM)

        tk.Label(sub_fs_frame, text="Enable Feature Selection?").grid(row=0, column=0)
        ttk.Checkbutton(sub_fs_frame, variable=self._apply_fs, onvalue=True, offvalue=False,
                        style="PLT.TCheckbutton").grid(row=0, column=1, padx=(10, 0))

        # 4. PREFERENCE LEARNING

        self._pl_frame = tk.Frame(self._others_frame, bd=2, relief='groove')
        self._pl_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        title_frame = tk.Frame(self._pl_frame, bg=colours.NAV_BAR)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(title_frame, text="Step 4: Preference Learning", font=ebrima_big, fg='white',
                 bg=colours.NAV_BAR).pack(padx=10, pady=5, side=tk.LEFT)
        help_btn = tk.Button(title_frame, command=lambda s=4: self._help_dialog(s), image=self._help_img,
                             relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                             highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                             activebackground=colours.NAV_BAR)
        help_btn.pack(padx=10, pady=5, side=tk.RIGHT)

        sub_pl_frame = tk.Frame(self._pl_frame)
        sub_pl_frame.pack(pady=10, side=tk.BOTTOM)

        tk.Label(sub_pl_frame, text="Choose Preference Learning Algorithm: ").grid(row=0, column=0, padx=(10, 0))
        options = [key.name for key in supported_methods.supported_algorithms_beginner.keys()]
        algo_menu = ttk.OptionMenu(sub_pl_frame, self._algorithm, options[0], *options, style='PLT.TMenubutton')
        algo_menu.grid(row=0, column=1, padx=(10, 10))
        algo_menu["menu"].config(bg="#e6e6e6")

        # 5. RUN EXPERIMENT

        self._run_frame = tk.Frame(self._others_frame, bd=2, relief='groove')
        self._run_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        title_frame = tk.Frame(self._run_frame, bg=colours.NAV_BAR)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(title_frame, text="Step 5: Run Experiment", font=ebrima_big, fg='white',
                 bg=colours.NAV_BAR).pack(padx=10, pady=5, side=tk.LEFT)
        help_btn = tk.Button(title_frame, command=lambda s=5: self._help_dialog(s), image=self._help_img,
                             relief='flat', bd=0, highlightbackground=colours.NAV_BAR,
                             highlightcolor=colours.NAV_BAR, highlightthickness=0, background=colours.NAV_BAR,
                             activebackground=colours.NAV_BAR)
        help_btn.pack(padx=10, pady=5, side=tk.RIGHT)

        sub_run_frame = tk.Frame(self._run_frame)
        sub_run_frame.pack(pady=10, side=tk.BOTTOM)

        self._run_btn = tk.Button(sub_run_frame, command=self._run_exp, image=self._run_img, relief='flat', bd=0)
        self._run_btn.pack()

        self._others_locked = False
        self._load_frame.bind('<<FileChange>>', self._toggle_lock)
        self._toggle_lock(None)  # force call self._toggle_lock() once to disable steps 2-4 the first time.
        self._run_btn.bind("<<PLTStateToggle>>", self._check_lock)  # bind

        # ^ n.b. ensured that 'Run Experiment' button, OutputLayer#nuerons checkbox and Steps 2-4 of BeginnerMenu
        # are re-disabled or re-enabled accordingly on close of stacked windows (help dialog or load params).
        # solution via binding state changes to method which ensures re-disable (or re-enable if appropriate time/case).

        ws.place_window(self, parent, position=ws.SIDE)

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

    def _help_dialog(self, step):
        """Open a help dialog window to assist the user in the given step of the BeginnerMenu.

        :param step: specifies for which step to load the help text.
        :type step: int from 1 to 4
        """
        if step == 1:
            BeginnerStep1HelpDialog(self)
        elif step == 2:
            BeginnerStep2HelpDialog(self)
        elif step == 3:
            BeginnerStep3HelpDialog(self)
        elif step == 4:
            BeginnerStep4HelpDialog(self)
        elif step == 5:
            BeginnerStep5HelpDialog(self)

    def _toggle_lock(self, event):
        """Unlock widgets for Steps 2-5 when a complete data set is loaded; keep them locked (disabled) otherwise.

        This method is called every time a <Configure> event occurs with respect to the data
        loading frame widget of Step 1 (i.e., :meth:`self._load_frame`).

        Widgets are 'locked' by setting their state to 'disable' and 'unlocked' by setting their state to 'normal'.

        :param event: the <Configure> event that triggered this method to be called.
        :type event: `tkinter Event`
        """
        # print(".............. self._toggle_lock() called ...............")
        # print("Checking for new full data...")
        if self._load_frame.is_data_loaded():  # if data is fully loaded
            # print("Full data is loaded.")
            if self._others_locked:  # and widgets are currently locked
                # print("Unlocking all widgets in other frame...")
                ws._toggle_state(self._others_frame, None, 'normal')  # enable all widgets in other frame
                self._others_locked = False
            # update autoencoder menu input size and output size each time new data has been loaded
            if self._autoencoder_menu is not None:
                data = self._load_frame.get_data()
                if isinstance(data, tuple):
                    objects, ranks = data
                    input_size = len(objects.columns) - 1  # -1 to account for ID column
                else:
                    input_size = len(data.columns) - 2  # -2 to account for ID column and ratings column
                # print("updating autoencoder input_size !!! " + str(input_size))
                self._autoencoder_menu.set_input_size(input_size)
            # resize to show everything properly (file paths)
            self.update_idletasks()
            self._on_resize(None)
        else:
            # print("Full data is NOT loaded.")
            # always lock since closing LoadingParamsWindow enables all widgets in this window (due to window stacking)
            # print("Locking all widgets in other frame...")
            ws._toggle_state(self._others_frame, None, 'disable')  # disable all widgets in other frame
            self._others_locked = True

    def _check_lock(self, event):
        """Unlock widgets for Steps 2-5 when a complete data set is loaded; keeps them locked (disabled) otherwise.

        This method is called every time a <<PLTStateToggle>> event occurs with respect to the 'Run Experiment' button
        widget of Step 5 (i.e., self._run_btn).

        Widgets are 'locked' by setting their state to 'disable' and 'unlocked' by setting their state to 'normal'.

        :param event: the <<PLTStateToggle>> event that triggered this method to be called.
        :type event: `tkinter Event`
        """
        # print(".............. self._check_lock() called ...............")
        new_state = str(self._run_btn.cget('state'))
        # print("new_state: " + str(new_state))
        if (new_state == 'disable') or (new_state == 'disabled'):
            # print("Run button state was changed (disabled) !!! -- ignoring.")
            return
        # print("Run button state was changed (activated) !!!")

        if self._load_frame.is_data_loaded():  # if data is fully loaded
            # print("Full data is loaded.")
            # print("Letting self._toggle_lock() do its thing...")
            return  # this stuff is handled by self._load_frame <Configure> events!
        else:
            # print("Full data is NOT loaded.")
            # always lock since closing LoadingParamsWindow enables all widgets in this window (due to window stacking)
            # print("Re-Locking all widgets in other frame...")
            ws._toggle_state(self._others_frame, None, 'disable')  # disable all widgets in other frame
        # print("done.")

    def _toggle_autoencoder_menu(self, *args):
        """Show or hide topology section of autoencoder menu.

        :param args: additional arguments for when the method is called as a callback function
            via the tk.IntVar.trace method.
        """
        if self._apply_ae.get():
            if self._autoencoder_menu is None:
                data = self._load_frame.get_data()
                if isinstance(data, tuple):
                    objects, ranks = data
                    input_size = len(objects.columns)-1  # -1 to account for ID column
                else:
                    input_size = len(data.columns)-2  # -2 to account for ID column and ratings column
                self._autoencoder_menu = AutoencoderSettingsBeginner(self._sub_ae_frame, input_size, self._on_resize)
                self._autoencoder_menu.pack(padx=5, pady=5)
            else:
                self._autoencoder_menu.pack(padx=5, pady=5)
        else:
            self._autoencoder_menu.pack_forget()

        # resize to show everything properly
        self.update_idletasks()
        self._on_resize(None)

    def _run_exp(self):
        """Trigger the execution of the experiment in a separate thread from the main (GUI) thread.

        A ProgressWindow widget is also initialized to keep a progress log and progress bar of the experiment
        execution process.

        Threading is carried out using the threading.Thread class.
        """
        print("Running Experiment...")

        data = self._load_frame.get_data()
        mdm, memory = self._load_frame.get_rank_derivation_params()

        # Get actual objects and set up Experiment variables
        self._exp = Experiment()
        if isinstance(data, tuple):
            _objects, _ranks = data
            objects = _objects
            self._exp._set_objects(_objects, has_ids=True)
            # ^ has_ids=True bc _load_data() adds ID column if there isn't already
            self._exp._set_ranks(_ranks, has_ids=True)
            # ^ has_ids=True bc _load_data() adds ID column if there isn't already
        else:
            objects = data.iloc[:, :-1]
            self._exp._set_single_data(data, has_ids=True)
            # ^ has_ids=True bc _load_data() adds ID column if there isn't already
            self._exp.set_rank_derivation_params(mdm=mdm, memory=memory)

        if self._apply_ae.get():
            # set autoencoder
            input_size = self._autoencoder_menu.get_input_size()
            code_size = self._autoencoder_menu.get_code_size()
            code_actf = ActivationType.RELU
            encoder_top = self._autoencoder_menu.get_encoder_neurons()
            print("AUTOENCODER encoder topology: " + str(encoder_top))
            encoder_actf = [ActivationType.RELU for _ in encoder_top]
            decoder_top = self._autoencoder_menu.get_decoder_neurons()
            print("AUTOENCODER decoder topology: " + str(decoder_top))
            decoder_actf = [ActivationType.RELU for _ in decoder_top]
            output_actf = ActivationType.RELU
            activation_functions = encoder_actf + [code_actf] + decoder_actf + [output_actf]
            print("AUTOENCODER activation_functions: " + str(activation_functions))
            lr = 0.001
            error_thresh = 0.001
            e = 10
            ae = Autoencoder(input_size, code_size, encoder_top, decoder_top, activation_functions, lr, error_thresh, e)
            self._exp.set_autoencoder(ae)

        # Normalize all feature values to the range [0,1] via the Min-Max method
        if self._apply_ae.get():
            n_cols = self._autoencoder_menu.get_code_size()
        else:
            n_cols = len(objects.columns)-1
        col_ids = np.arange(n_cols)
        # ^ remove last id to avoid out of range exception since col_ids includes ID col

        for c_id in col_ids:  # just for debugging purposes...
            print("Setting normalization for feature " + str(c_id))
        self._exp.set_normalization(list(col_ids), NormalizationType.MIN_MAX)
        # ^ (min_val=0, max_val=1 by default)
        # don't forget to convert norm_settings values to tk.StringVars in order to work with ResultsWindow!!
        self._norm_settings = {c: tk.StringVar(value=NormalizationType.MIN_MAX.name) for c in col_ids}
        # also in the process, list all features as included in the experiment
        self._include_settings = {c: tk.BooleanVar(value=True) for c in col_ids}

        # fixed shuffle settings
        shuffle = False
        random_state = None  # random seed

        # feature selection
        if self._apply_fs.get():
            fs_method = FSMethod.SFS
            fs_method_obj = SFS(verbose=False)  # do not show detailed SFS info in progress log
            fs_params = None
            fs_algo = PLAlgo.RANKSVM
            fs_algo_obj = RankSVM(kernel=KernelType.LINEAR)
            fs_algo_params = fs_algo_obj.get_params_string()
            fs_eval = EvaluatorType.HOLDOUT
            fs_eval_obj = HoldOut()  # test_proportion: (default: 0.3)
            fs_eval_params = fs_eval_obj.get_params_string()
            self._exp.set_fs_method(fs_method_obj)
            self._exp.set_fs_algorithm(fs_algo_obj)
            self._exp.set_fs_evaluator(fs_eval_obj)
        else:
            fs_method = FSMethod.NONE
            fs_params = None
            fs_algo = None
            fs_algo_params = None
            fs_eval = EvaluatorType.NONE
            fs_eval_params = None

        # preference learning
        pl_algo_name = self._algorithm.get()
        pl_algo = PLAlgo[pl_algo_name]
        # TODO: for special cases where we want parameter values other than the defaults, add an if statement here
        if pl_algo == PLAlgo.BACKPROPAGATION:
            pl_algo_obj = BackpropagationTF(ann_topology=[5, 1],
                                            activation_functions=[ActivationType.RELU, ActivationType.RELU])
            # ^ learn_rate, error_threshold, epochs: (default: 0.1, 0.1, 10)
        elif pl_algo == PLAlgo.RANKSVM:
            pl_algo_obj = RankSVM(kernel=KernelType.RBF, gamma=1)
        else:
            # get algorithm object and instantiate with its default parameter values
            return supported_methods.get_algorithm_instance(pl_algo, beginner_mode=True)

        pl_algo_params = pl_algo_obj.get_params_string()
        pl_eval = EvaluatorType.HOLDOUT
        pl_eval_obj = HoldOut()  # test_proportion: (default: 0.3)
        pl_eval_params = pl_eval_obj.get_params_string()
        self._exp.set_pl_algorithm(pl_algo_obj)
        self._exp.set_pl_evaluator(pl_eval_obj)

        obj_path = self._load_frame.get_objects_path()
        ranks_path = self._load_frame.get_ranks_path()
        single_path = self._load_frame.get_single_path()

        file_paths = [obj_path, ranks_path, single_path]

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

        if self._apply_fs.get():
            print("---------------- FS ----------------")
            # fs
            print("FS Method: SFS")
            print("FS algorithm: Backpropagation")
            print("FS algorithm params:")
            print(fs_algo_params)
            print("FS evaluator: HoldOut")
            print("FS evaluator params:")
            print(fs_eval_params)

        print("---------------- PL ----------------")
        # pl
        print("PL Algorithm: " + str(self._algorithm.get()))
        print("PL Algorithm params:")
        print(pl_algo_params)
        print("PL evaluator: HoldOut")
        print("PL evaluator params:")
        print(pl_eval_params)
        print("--------------------------------------")

        q = Queue()

        exec_stopper = AbortFlag()

        pw = ProgressWindow(self, q, exec_stopper)  # instantiate a progress window

        te = Thread(target=self._run_exp_, args=(self._exp, shuffle, random_state, pw, exec_stopper, data, file_paths,
                                                 fs_method, fs_params, fs_algo, fs_algo_params, fs_eval, fs_eval_params,
                                                 pl_algo, pl_algo_params, pl_eval, pl_eval_params))
        pw.set_exec_thread(te)
        te.start()

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

        preproc_info = [self._include_settings,
                        self._norm_settings]

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
