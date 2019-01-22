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
from tkinter import messagebox
from tkinter import ttk
import queue

import pyplt.gui.util.windowstacking as ws
from pyplt import ROOT_PATH
from pyplt.exceptions import NoFeaturesError
from pyplt.gui.experiment.results.resultsscreen import ResultsWindow
from pyplt.gui.util import colours

GWL_EXSTYLE=-20
WS_EX_APPWINDOW=0x00040000
WS_EX_TOOLWINDOW=0x00000080


class ProgressWindow(tk.Toplevel):
    """Window widget displaying the execution progress of an experiment via a progress bar and progress log.

    Extends the class `tkinter.Toplevel`.
    """

    def __init__(self, parent_window, q, exec_stopper):
        """Initializes the ProgressWindow widget object.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        :param q: the queue to be used for communication between the thread (`threading.Thread`) carrying
            out the execution of the experiment and the progress log (`tkinter.Listbox`) of this class.
        :type q: `queue.Queue`
        :param exec_stopper: an abort flag object used to abort the execution before completion
            if so instructed by the user.
        :type exec_stopper: :class:`pyplt.util.AbortFlag`
        """
        self._wait = False  # hack
        self._completed = False
        self._parent_window = parent_window
        self._queue = q
        self._exec_stopper = exec_stopper
        self._exec_thread = None

        tk.Toplevel.__init__(self, self._parent_window)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, self._parent_window, ws.Mode.OPEN_ONLY)
        ws.stack_window(self, self._parent_window)

        # set window size & position the window in centre of screen
        width = 600
        height = 450
        # get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

        # remove window decorations (only for Windows OS)
        if os.name == 'nt':
            self.overrideredirect(True)
            self.after(10, self._setup_window)

        self.protocol("WM_DELETE_WINDOW", self._on_abort)
        self.title("Experiment Execution Progress")

        # ----- populate with remaining widgets -----
        self._main_frame = tk.Frame(self, bg=colours.PROGRESS_BACK)
        self._main_frame.pack(expand=True, fill=tk.BOTH)

        # title
        title_frame = tk.Frame(self._main_frame, bg=colours.PROGRESS_BACK)
        title_frame.grid(row=0, column=0, sticky='ew')
        tk.Label(title_frame, text="Experiment Execution Progress", font='Ebrima 12 bold',  fg='white',
                 bg=colours.PROGRESS_BACK).grid(row=0, column=0, sticky='w', padx=20, pady=20)

        # progress log Listbox
        log_frame = tk.Frame(self._main_frame, bg=colours.PROGRESS_BACK, padx=50)
        log_frame.grid(row=1, column=0, sticky='nsew')

        lg_frame = tk.Frame(log_frame, bd=2, relief='sunken')
        lg_frame.pack(expand=True, fill=tk.BOTH)

        self._log = tk.Listbox(lg_frame, activestyle='none', bg='white', bd=0, highlightthickness=0)

        # Listbox vertical scrollbar
        v_scroll = ttk.Scrollbar(lg_frame, orient="vertical", command=self._log.yview,
                                 style="White.PLT.Vertical.TScrollbar")
        v_scroll.pack(side='right', fill='y')
        self._log.configure(yscrollcommand=v_scroll.set)

        # Listbox horizontal scrollbar
        h_scroll = ttk.Scrollbar(lg_frame, orient="horizontal", command=self._log.xview,
                                 style="White.PLT.Horizontal.TScrollbar")
        h_scroll.pack(side='bottom', fill='x')
        self._log.configure(xscrollcommand=h_scroll.set)

        self._log.pack(expand=True, fill=tk.BOTH, padx=(2, 0), pady=(1, 0))

        self._report_img = tk.PhotoImage(file=os.path.join(ROOT_PATH,
                                                           "assets/buttons/generate_report_128_30_01_white.png"))
        self._abort_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/abort_76_30_01.png"))
        self._close_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/close_76_30_01.png"))

        # bottom area
        self._bottom_frame = tk.Frame(self._main_frame, bg=colours.PROGRESS_BACK)
        self._bottom_frame.grid(row=2, column=0, sticky='ew')

        # progress bar
        self._progress_bar = ttk.Progressbar(self._bottom_frame, mode='indeterminate',
                                             style="PLT.Horizontal.TProgressbar")
        self._progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=20, pady=20)

        # abort button
        self._abort_btn = tk.Button(self._bottom_frame, command=self._on_abort, image=self._abort_img,
                                    relief='flat', bd=0, highlightbackground=colours.PROGRESS_BACK,
                                    highlightcolor=colours.PROGRESS_BACK, highlightthickness=0,
                                    background=colours.PROGRESS_BACK,
                                    activebackground=colours.PROGRESS_BACK)
        self._abort_btn.pack(side=tk.RIGHT, padx=20, pady=20)

        # configure sizes
        self._main_frame.grid_columnconfigure(0, weight=1)
        self._main_frame.grid_rowconfigure(1, weight=1)

        # start progress bar
        self._progress_bar.start()
        # initialize the progress log
        self.log("Commenced executing experiment.")

        # init report variables
        self._experiment = None
        self._time_info = None
        self._data_info = None
        self._preproc_info = None
        self._fs_info = None
        self._fs_algo_info = None
        self._fs_eval_info = None
        self._pl_algo_info = None
        self._pl_eval_info = None
        self._eval_metrics = None
        self._fold_metrics = None
        self._shuffle_info = None

        # start polling
        self._poll()

    def update_gui(self):
        """Update the GUI.

        Hack-y method called while :class:`pyplt.plalgorithms.ranksvm.RankSVM` precomputes kernels from the
        experiment execution thread to keep the GUI (main) thread going.

        The method also sets the flag variable :attr:`self._wait` to True immediately before calling
        :meth:`self.update_idletasks()` and sets it back to False immediately after. This is done in order to
        avoid a deadlock between threads when aborting experiment execution.
        """
        self._wait = True
        # print("updating idletasks")
        self.update_idletasks()
        self._wait = False
        # print("done updating idletasks")

    def put(self, item):
        """Add a given string item to the queue to be in turn displayed in the progress log.

        :param item: a string to be added to the queue and therefore displayed in the progress log.
        :type item: str
        """
        # print("New item added to queue...")
        self._queue.put(item)
        # print("current # items: " + str(len(self._queue)))

    def _poll(self):
        """Poll for new items in the queue, triggering their display in the progress log.

        On completion, the function calls itself again after a 2 second pause until a None or "DONE" item is
        encountered in the queue.
        """
        try:
            while 1:
                # print("polling...")
                line = self._queue.get_nowait()
                if ("NoFeaturesError" in line) or ("NoRanksDerivedError" in line) or \
                        ("InvalidParameterValueException" in line) or ("NormalizationValueError" in line) or \
                        ("IncompatibleFoldIndicesException" in line):
                    error = str(line).split("??")
                    # error_name = error[0]
                    error_summary = error[1]
                    error_message = error[2]
                    # print("progresswindow caught " + error_name + "!")
                    messagebox.showerror(error_summary, error_message, parent=self)
                    self._on_abort()  # simulate abort on messagebox close
                elif line == "MemoryError":
                    # print("progresswindow caught MemoryError!")
                    messagebox.showerror("Cannot run experiment - Memory Error", "Out of memory - dataset is too "
                                                                                 "large. The algorithm you chose does "
                                                                                 "not yet support datasets of this "
                                                                                 "size. Please try again with a "
                                                                                 "different algorithm or use a "
                                                                                 "smaller dataset.", parent=self)
                    self._on_abort()  # simulate abort on messagebox close
                elif (line is None) or (line == "DONE"):
                    print("TERMINATING POLLING...")
                    return  # break
                else:
                    # print("PROGRESS WINDOW GOT " + str(line) + " FROM POLL!")
                    self.log(line)
                self.update_idletasks()
        except queue.Empty:
            # print("queue empty.")
            pass
        # print("setting after 0.5 seconds")
        self.after(500, self._poll)

    def log(self, event):
        """Display the given string at the end of the progress log (`tkinter.Listbox` object).

        :param event: the string to be displayed in the progress log.
        :type event: str
        """
        state = str(self._log.cget('state'))
        if (state == 'disable') or (state == 'disabled'):
            # activate to still log things when disabled (e.g., when showing results screen!)
            self._log.config(state='normal')
        # add event string as new row in log Listbox
        # print("Adding the following log to Listbox: " + str(event))
        self._log.insert(tk.END, event)
        self._log.yview(tk.END)  # scroll to bottom

    def progress(self):
        """Increment the progress bar (`ttk.Progressbar` object)."""
        # increment progress bar
        self._progress_bar.step()

    def done(self, experiment, time_info, data_info, preproc_info, fs_info, fs_algo_info,
             fs_eval_info, pl_algo_info, pl_eval_info, eval_metrics, fold_metrics, shuffle_info):
        """Update the `ProgressWindow` widget on completion of experiment execution.

        Add a 'Success' label, adds a 'Generate Report' button (to open or re-open the results window),
        and converts the 'Abort' button into a 'Close' button. The window
        (:class:`pyplt.gui.experiment.results.resultsscreen.ResultsWindow`) containing the experiment details and
        results is opened automatically.

        :param experiment: the experiment to be passed on to the results window.
        :type experiment: :class:`pyplt.experiment.Experiment`
        :param time_info: the time meta-data to be passed on to the results window in the form of a list containing
            the start timestamp (UTC), end timestamp (UTC), and duration of the experiment.
        :type time_info: list of float (size 3)
        :param data_info: the data to be passed on to the results window in the form of a list containing the number
            of objects, the number of ranks, and the list of data file paths.
        :type data_info: list (size 3)
        :param preproc_info: the pre-processing information to be passed on to the results window in the form of
            a list containing the include settings dict and the normalization settings dict.
        :type preproc_info: list of dict (size 2)
        :param fs_info: feature selection method information to be passed on to the results window (if applicable) in
            the form of a list containing the chosen feature selection method type
            (:class:`pyplt.util.enums.FSMethod`), the string representation of its parameters, and the list of
            features selected by the feature selection method.
        :type fs_info: list (size 3)
        :param fs_algo_info: the feature selection algorithm information to be passed on to the results window
            (if applicable) in the form of a list containing the chosen algorithm type
            (:class:`pyplt.util.enums.PLAlgo`) and the string representation of its parameters.
        :type fs_algo_info: list (size 2)
        :param fs_eval_info: the feature selection evaluation method information to be passed on to the results window
            (if applicable) in the form of a list containing the evaluation method type
            (:class:`pyplt.util.enums.EvaluatorType`) and the string representation of its parameters.
        :type fs_eval_info: list (size 2)
        :param pl_algo_info: the preference learning algorithm information to be passed on to the results window in the
            form of a list containing the algorithm type (:class:`pyplt.util.enums.PLAlgo`) and the string
            representation of its parameters.
        :type pl_algo_info: list (size 2)
        :param pl_eval_info: the evaluation method information to be passed on to the results window (if applicable) in
            the form of a list containing the evaluation method type (:class:`pyplt.util.enums.EvaluatorType`)
            and the string representation of its parameters.
        :type pl_eval_info: list (size 2)
        :param eval_metrics: the evaluation/training results to be passed on to the results window in the form of a
            dict with keys:

            * '`Training Accuracy`'
            * '`Test Accuracy`' (if applicable)
        :type eval_metrics: dict
        :param fold_metrics: optional fold-specific information (default None) in the form of list of tuples, each
            containing the start timestamp, end timestamp, evaluation metrics, and a
            `pandas.DataFrame` representation of the trained model as follows:

                * start_time -- `datetime` timestamp (UTC timezone)
                * end_time -- `datetime` timestamp (UTC timezone)
                * eval_metrics -- dict with keys:

                  * '`Training Accuracy`'
                  * '`Test Accuracy`' (if applicable)
                * model -- `pandas.DataFrame`
        :type fold_metrics: list of tuple, optional
        :param shuffle_info: list containing the chosen settings related to shuffling the dataset:

            * shuffle -- bool specifying whether or not the dataset was shuffled at the start of the experiment
              execution.
            * random_seed -- optional seed (int or None) used to shuffle the dataset.
        :type shuffle_info: list (size 2)
        """
        # called when experiment execution is complete
        # convert 'abort' btn into 'success' label, 'generate report' btn (re-open results screen), and 'close' btn
        self._completed = True
        self._experiment = experiment
        self._time_info = time_info
        self._data_info = data_info
        self._preproc_info = preproc_info
        self._fs_info = fs_info
        self._fs_algo_info = fs_algo_info
        self._fs_eval_info = fs_eval_info
        self._pl_algo_info = pl_algo_info
        self._pl_eval_info = pl_eval_info
        self._eval_metrics = eval_metrics
        self._fold_metrics = fold_metrics
        self._shuffle_info = shuffle_info

        self.put("Experiment execution terminated.")
        self._progress_bar.stop()

        # hide progress bar & abort button
        self._progress_bar.destroy()
        self._abort_btn.pack_forget()

        # add 'success' label
        tk.Label(self._bottom_frame, text="Execution Success!", bg='#5cd65c', padx=10).pack(side=tk.LEFT,
                                                                                            padx=20, pady=20)

        # add 'generate report' button (re-open results screen)
        tk.Button(self._bottom_frame, command=self._gen_report, image=self._report_img, relief='flat', bd=0,
                  highlightbackground=colours.PROGRESS_BACK,
                  highlightcolor=colours.PROGRESS_BACK, highlightthickness=0,
                  background=colours.PROGRESS_BACK,
                  activebackground=colours.PROGRESS_BACK).pack(side=tk.LEFT, pady=20)

        # replace 'abort' button with 'close' button
        self._abort_btn.configure(image=self._close_img, relief='flat', bd=0)
        self._abort_btn.pack(side=tk.RIGHT, padx=20, pady=20)

        # force generate the report automatically the first time upon execution completion
        self._gen_report()

    def _gen_report(self):
        """Open the window containing the experiment details and results.

        The results window is a widget of type :class:`pyplt.gui.experiment.results.resultsscreen.ResultsWindow`.
        """
        r_win = ResultsWindow(parent=self, parent_window=self, experiment=self._experiment, time_info=self._time_info,
                              data_info=self._data_info, preproc_info=self._preproc_info,
                              fs_info=self._fs_info, fs_algo_info=self._fs_algo_info, fs_eval_info=self._fs_eval_info,
                              pl_algo_info=self._pl_algo_info, pl_eval_info=self._pl_eval_info,
                              eval_metrics=self._eval_metrics, fold_metrics=self._fold_metrics,
                              shuffle_info=self._shuffle_info)
        r_win.geometry('1000x500')

    def _on_abort(self):
        """Start aborting the experiment (if its execution is not yet complete) and trigger closing of `ProgressWindow`.

        After some delay, this method calls :meth:`self._last_things()` which always waits for the experiment execution
        thread to join the main (GUI) thread (an thus stop) before closing the `ProgressWindow`.
        """
        self._abort_btn.configure(state='disable')  # disable abort button to prevent further calls (and thus errors)
        print("ABORT/CLOSE.")
        if not self._completed:
            # say your last words (unless experiment was successfully completed)...
            self.put("Aborting experiment execution - please wait...")
            self.update_idletasks()
        # tell execution thread to stop running
        self._exec_stopper.stop()

        # delay call to join() to avoid deadlock by letting any prior call to update_gui() from exec_thread to finish
        self.after(1000, self._last_things)

    def _last_things(self):
        """Finish aborting the experiment (if its execution is not yet complete) and close the ProgressWindow

        This method always waits for the experiment execution thread to join the main (GUI) thread (an thus stop)
        before closing the ProgressWindow.

        In order to avoid deadlock between threads, this method first lets any prior call to :meth:`self.update_gui()`
        from :attr:`self._exec_thread` finish off by checking if :attr:`self._wait` is False; if not, it tries again
        after 1 second (and so on).
        """
        if self._wait:
            # delay again to avoid deadlock by letting any prior call to update_gui() from exec_thread to finish
            self.after(1000, self._last_things)
        else:
            # wait for exec thread to join us before being able to run a new experiment!
            if self._exec_thread is not None:
                print("Waiting to join thread to main thread...")
                self._exec_thread.join()
                print("Done joining thread.")

            # close this window and restore the parent window
            ws.on_close(self, self._parent_window)

    def set_exec_thread(self, thread):
        """Set the execution thread variable."""
        self._exec_thread = thread

    def _setup_window(self):
        """Hide the window's toolbar & border while otherwise retaining normal window properties (for Windows OS)."""
        # from https://stackoverflow.com/questions/30786337/
        # tkinter-windows-how-to-view-window-in-windows-task-bar-which-has-no-title-bar
        from ctypes import windll
        hwnd = windll.user32.GetParent(self.winfo_id())
        style = windll.user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
        style = style & ~WS_EX_TOOLWINDOW
        style = style | WS_EX_APPWINDOW
        res = windll.user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
        # re-assert the new window style
        self.wm_withdraw()
        self.after(10, lambda: self.wm_deiconify())
