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
from tkinter import filedialog

from pyplt import ROOT_PATH


class PLAlgorithm:
    """Base class for all preference learning algorithms."""

    _debug = False
    _description = ""
    _params = {}
    _name = ""

    _train_accuracy = None
    _result_model = None

    def __init__(self, description="A preference learning algorithm.", name="", debug=False, **kwargs):
        """Initializes the PLAlgorithm object.

        :param description: a description of the algorithm (default "A preference learning algorithm.").
        :type description: str, optional
        :param name: the name of the algorithm (default "").
        :type name: str, optional
        :param debug: specifies whether or not to print notes to console for debugging (default False).
        :type debug: bool, optional
        :param kwargs: any additional parameters for the algorithm.
        """
        self._name = name
        self._debug = debug
        self._description = description
        self._params = {}
        for key in kwargs:
            self._params[key] = kwargs[key]

    # Abstract methods

    def init_train(self, n_features):
        """Abstract method for carrying out any initializations prior to the training stage.

        All children classes must implement this method.

        :param n_features: the number of features to be used during the training process.
        :type n_features: int
        """
        pass

    def train(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Abstract method for the training stage in the machine learning process.

        All children classes must implement this method.

        :param train_objects: containing the objects data to train the model on.
        :type train_objects: `pandas.DataFrame`
        :param train_ranks: containing the pairwise rank data to train the model on.
        :type train_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used when training;
            if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * True or any other value -- if execution is completed successfully.
            * None -- if experiment is aborted before completion by `exec_stopper`.
        """
        pass

    def predict(self, input_object, progress_window=None, exec_stopper=None):
        """Abstract method for predicting the output of a given input by running it through the learned model.

        All children classes must implement this method.

        :param input_object: the input data corresponding to a single object.
        :type input_object: one row from a `pandas.DataFrame`
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return: a list containing the predicted output resulting from running the learned model using the given input.
        :rtype: list of float (size 1)
        """
        pass

    def save_model_with_dialog(self, timestamp, parent_window, suffix=""):
        """Open a file dialog window (GUI) and save the learned model to file at the path indicated by the user.

        The model file must be a Comma Separated Value (CSV)-type file with the extension '.csv'.

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
            # only allow .csv files!
            self.save_model(timestamp, path=fpath)
            return True
        else:
            print("Cancelled save model.")
            return False

    def save_model(self, timestamp, path="", suppress=False):
        """Abstract model to save the model to file.

        Optionally, the file creation may be suppressed and a `pandas.DataFrame` representation of the model
        returned instead.

        All children classes must implement this method.

        :param timestamp: the timestamp to be included in the file name.
        :type timestamp: float
        :param path: the path at which the file is to be saved (default ""). If "", the file is saved to a logs folder
            in the project root directory by default.
        :type path: str, optional
        :param suppress: specifies whether or not to suppress the file creation and return a `pandas.DataFrame`
            representation of the model instead (default False).
        :type suppress: bool, optional
        :return: a `pandas.DataFrame` representation of the model, if the `suppress` parameter was set to True,
            otherwise None.
        :rtype:
            * `pandas.DataFrame` -- if `suppress` is True
            * None -- otherwise
        """
        # similar to get_params_string()
        pass

    def load_model(self):
        """Abstract method for loading a model which was trained using this algorithm.

        All children classes must implement this method.
        """
        # a parsing version of get_params_string()
        pass

    @staticmethod
    def transform_data(object_):
        """Abstract method to transform a sample (object) into the format required by this particular algorithm implementation.

        All children classes must implement this method.

        :param object_: the data sample (object) to be transformed.
        :type object_: one row from a `pandas.DataFrame`
        :return: the transformed object.
        """
        pass

    # Common methods

    def calc_train_accuracy(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Base method for calculating the training accuracy of the learned model.

        The training accuracy is determined by calculating the percentage of how many of the training ranks
        the model is able to predict correctly.

        :param train_objects: the objects data the model was trained on.
        :type train_objects: `pandas.DataFrame`
        :param train_ranks: the pairwise rank data the model was trained on.
        :type train_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used when training;
            if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the training accuracy of the learned model -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: float
        """
        if self._debug:
            print("Calculating TRAINING accuracy...")

        # check if execution was aborted before updating progress window since we couldn't check right after train()
        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting training accuracy execution...")
            return

        if progress_window is not None:
            progress_window.put("Calculating accuracy.")

        good_predicts = 0
        bad_predicts = 0
        for idx, row in train_ranks.iterrows():
            # print("Predicting rank " + str(row[0]) + " > " + str(row[1]))
            if use_feats is None:
                pref_object = train_objects.loc[row[0], :]  # preferred object
                other_object = train_objects.loc[row[1], :]  # other object
            else:
                pref_object = train_objects.loc[row[0], use_feats]  # preferred object
                other_object = train_objects.loc[row[1], use_feats]  # other object
            pref_object = self.transform_data(pref_object)
            # print(pref_object)
            other_object = self.transform_data(other_object)
            # print(other_object)
            predict_pref = self.predict(pref_object, progress_window=progress_window, exec_stopper=exec_stopper)
            if predict_pref is None:  # check if experiment was aborted
                # abort execution!
                print("Aborting training accuracy execution...")
                return
            predict_other = self.predict(other_object, progress_window=progress_window, exec_stopper=exec_stopper)
            if predict_other is None:  # check if experiment was aborted
                # abort execution!
                print("Aborting training accuracy execution...")
                return
            # print("pref prediction: " + str(predict_pref))
            # print("other prediction: " + str(predict_other))
            if predict_pref[0] > predict_other[0]:
                # print("Correct.")
                good_predicts += 1
            else:
                # print("Wrong.")
                bad_predicts += 1

        if self._debug:
            print("Total good predicts (TRAINING) = " + str(good_predicts))
            print("Total bad predicts (TRAINING) = " + str(bad_predicts))
        accuracy = good_predicts / len(train_ranks) * 100
        if self._debug:
            print("Performance (TRAINING): " + str(accuracy) + "%")
        self._train_accuracy = accuracy

        if progress_window is not None:
            progress_window.put("Training accuracy: " + str(accuracy) + "%")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Backpropagation execution...")
            return

        return accuracy

    def test(self, objects, test_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Base method for calculating the prediction accuracy of the learned model on a given dataset (test set).

        The prediction accuracy is determined by calculating the percentage of how many of the test ranks
        the model is able to predict correctly.

        :param objects: the objects data to be predicted by the model.
        :type objects: `pandas.DataFrame`
        :param test_ranks: the pairwise rank data to be predicted by the model.
        :type test_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used during the prediction process;
            if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the prediction accuracy of the learned model -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        """
        if progress_window is not None:
            progress_window.put("Starting testing with Holdout method.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Holdout execution...")
            return

        print("running generic test() method.")
        good_predicts = 0
        bad_predicts = 0
        for idx, row in test_ranks.iterrows():
            if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
                # abort execution!
                print("Aborting Holdout execution...")
                return
            # print("Predicting rank " + str(row[0]) + " > " + str(row[1]))
            if use_feats is None:
                pref_object = objects.loc[row[0], :]  # preferred object
                other_object = objects.loc[row[1], :]  # other object
            else:
                pref_object = objects.loc[row[0], use_feats]  # preferred object
                other_object = objects.loc[row[1], use_feats]  # other object
            pref_object = self.transform_data(pref_object)
            # print(pref_object)
            other_object = self.transform_data(other_object)
            # print(other_object)
            predict_pref = self.predict(pref_object, progress_window=progress_window, exec_stopper=exec_stopper)
            if predict_pref is None:  # check if experiment was aborted
                # abort execution!
                print("Aborting training accuracy execution...")
                return
            predict_other = self.predict(other_object, progress_window=progress_window, exec_stopper=exec_stopper)
            if predict_other is None:  # check if experiment was aborted
                # abort execution!
                print("Aborting training accuracy execution...")
                return
            # print("pref prediction: " + str(predict_pref))
            # print("other prediction: " + str(predict_other))
            if predict_pref[0] > predict_other[0]:
                # print("Correct.")
                good_predicts += 1
            else:
                # print("Wrong.")
                bad_predicts += 1

        if self._debug:
            print("Total good predicts = " + str(good_predicts))
            print("Total bad predicts = " + str(bad_predicts))
        accuracy = good_predicts / len(test_ranks) * 100

        if progress_window is not None:
            progress_window.put("Testing complete.")
            progress_window.put("Testing accuracy: " + str(accuracy) + "%")

        if self._debug:
            print("Performance: " + str(accuracy) + "%")
        # self._eval_accuracy = accuracy

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Holdout execution...")
            return

        return accuracy

    # Getters and setters

    def get_name(self):
        """Get the name of the preference learning algorithm.

        :return: the name of the algorithm.
        :rtype: str
        """
        return self._name

    def get_description(self):
        """Get the preference learning algorithm.

        :return: the description of the algorithm.
        :rtype: str
        """
        return self._description

    def get_params(self):
        """Return all additional parameters of the preference learning algorithm (if applicable).

        :return: a dict containing all additional parameters of the algorithm with the parameter names as the
            dict's keys and the corresponding parameter values as the dict's values (if applicable).
        :rtype: dict
        """
        return self._params

    def get_params_string(self):
        """Return a string representation of all additional parameters of the preference learning algorithm (if applicable).

        :return: the string representation of all additional parameters of the algorithm (if applicable).
        :rtype: str
        """
        return "{" + self._get_param_string(self._params) + "}"

    def _get_param_string(self, params):
        """Internal recursive method for the construction of a string representation of additional method parameters.

        :param params: a parameter or list of parameters to be included in the string.
        :type params: dict
        :return: a string representation of the given parameters.
        :rtype: str
        """
        ret = ""
        for p in params:
            if len(ret) > 0:
                ret += "; "
            try:
                param_dict = params[p]
                for d in param_dict.keys():  # will only work if params[p] is a dict
                    ret += "{" + self._get_param_string(param_dict[d]) + "}"
            except AttributeError:  # if not dict
                ret += str(p) + ": " + str(params[p])  # even tuples work here
        return ret

    def get_train_accuracy(self):
        """Get the training accuracy of the learned model.

        :return: the training accuracy of the learned model.
        :rtype: float
        """
        return self._train_accuracy

    # def get_result_model(self):
    #     return self._result_model

    def clean_up(self):
        """Base method for any potential final clean up instructions to be carried out.

        Does nothing unless overriden in child class.
        """
        return
