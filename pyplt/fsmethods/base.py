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


class FeatureSelectionMethod:
    """Base class for all feature selection methods."""

    def __init__(self, description="A feature selection method.", name="", **kwargs):
        """Initializes the FeatureSelectionMethod object.

        :param description: a description of the feature selection method (default "A feature selection method.").
        :type description: str, optional
        :param name: the name of the feature selection method (default "").
        :type name: str, optional
        :param kwargs: any additional parameters for the feature selection method.
        """
        self._name = name
        self._description = description
        self._params = {}
        for key in kwargs:
            self._params[key] = kwargs[key]

        self._orig_features = []
        self._sel_features = []

    # Abstract methods

    def _evaluate(self, objects, ranks, feature_set, algorithm, test_objects=None, test_ranks=None,
                  preprocessed_folds=None, progress_window=None, exec_stopper=None, **kwargs):
        """Abstract method for measuring the effectiveness of a given feature subset.

        All children classes must implement this method.

        :param objects: the objects data to evaluate the feature set with (if applicable).
        :type objects: `pandas.DataFrame`
        :param ranks: the pairwise rank data to evaluate the feature set with (if applicable).
        :type ranks: `pandas.DataFrame`
        :param feature_set: the feature set being evaluated.
        :type feature_set: list of str
        :param algorithm: the algorithm used to train models to evaluate the given feature set.
        :type algorithm: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        :param test_objects: optional objects data used to test the models used to evaluate and select
            features (default None).
        :type test_objects: `pandas.DataFrame` or None, optional
        :param test_ranks: optional pairwise rank data used to test the models used to evaluate and select features
            (default None).
        :type test_ranks: `pandas.DataFrame` or None, optional
        :param preprocessed_folds: the data used to evaluate the feature set with in the form of pre-processed folds
            (default None). This is an alternative way to pass the data and is only considered if either of the
            `objects` and `ranks` parameters is None.
        :type preprocessed_folds: :class:`pyplt.evaluation.cross_validation.PreprocessedFolds` or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :param kwargs: any additional evaluation parameters.
        :return:
            * the evaluation measure with respect to the given feature set -- if execution is completed successfully.
            * None if aborted if aborted before completion by `exec_stopper`.
        :rtype: float
        """
        pass

    def select(self, objects, ranks, algorithm, test_objects=None, test_ranks=None,
               preprocessed_folds=None, progress_window=None, exec_stopper=None):
        """Abstract method for running the feature selection process.

        All children classes must implement this method.

        :param objects: the objects data to be used during the feature selection process. If None, the
            data is obtained via the `preprocessed_folds` parameter instead.
        :type objects: `pandas.DataFrame` or None
        :param ranks: the pairwise rank data to be used during the feature selection process. If None, the
            data is obtained via the `preprocessed_folds` parameter instead.
        :type ranks: `pandas.DataFrame` or None
        :param algorithm: the algorithm to be used for feature selection (if applicable).
        :type algorithm: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        :param test_objects: optional test objects data to be used during the feature selection process (default None).
        :type test_objects: `pandas.DataFrame` or None, optional
        :param test_ranks: optional test pairwise rank data to be used during the feature selection process
            (default None).
        :type test_ranks: `pandas.DataFrame` or None, optional
        :param preprocessed_folds: the data used to evaluate the feature set with in the form of pre-processed folds
            (default None). This is an alternative way to pass the data and is only considered if either of the
            `objects` and `ranks` parameters is None.
        :type preprocessed_folds: :class:`pyplt.evaluation.cross_validation.PreprocessedFolds` or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the subset of selected features -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: list of str
        """
        pass

    # Getters and setters

    def get_name(self):
        """Get the name of the feature selection method.

        :return: the name of the feature selection method.
        :rtype: str
        """
        return self._name

    def get_selected_features(self):
        """Get the subset of selected features.

        :return: the subset of selected features.
        :rtype: list of str
        """
        return self._sel_features

    def get_description(self):
        """Get the description of the feature selection method.

        :return: the description of the feature selection method.
        :rtype: str
        """
        return self._description

    def get_params(self):
        """Return all additional parameters of the feature selection method (if applicable).

        :return: a dict containing all additional parameters of the feature selection method with the parameter
            names as the dict's keys and the corresponding parameter values as the dict's values (if applicable).
        :rtype: dict
        """
        return self._params

    def get_params_string(self):
        """Return a string representation of all additional parameters of the feature selection method (if applicable).

        :return: the string representation of all additional parameters of the feature selection method (if applicable).
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
            if isinstance(params[p], tuple):
                ret += " {" + self._get_param_string(params[p]) + "} "
            else:
                ret += str(p) + ": " + str(params[p])
        return ret
