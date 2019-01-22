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

from sklearn import model_selection

from pyplt.evaluation.base import Evaluator
from pyplt.exceptions import InvalidParameterValueException
from pyplt.util.enums import EvaluatorType


class PreprocessedFolds:
    """Class for neatly storing and working with a dataset that has been split into two or more folds.

    The data in each fold is assumed to be pre-processed prior to instantiation of this class.
    """

    def __init__(self, folds):
        """Initializes the `PreprocessedFolds` instance and stores the given fold data.

        :param folds: a list of tuples containing the pre-processed training set and test set (if applicable)
            of each fold. Each tuple (fold) should contain:

            * train_objects: `pandas.DataFrame`
            * train_ranks: `pandas.DataFrame`
            * test_objects: `pandas.DataFrame` (if applicable) or None
            * test_ranks: `pandas.DataFrame` (if applicable) or None

            If either the `test_objects` or `test_ranks` of the first fold is None, it is assumed that only training
            will be carried out.
        :type folds: list of tuples (size 4):
        """
        self._folds = folds
        self._n_folds = len(self._folds)
        self._features = folds[0][0].columns.copy()
        # if either the test_objects or test_ranks of the first fold is None, assume training only
        if folds[0][2] is None or folds[0][3] is None:
            self._train_only = True
        else:
            self._train_only = False

    def next_fold(self):
        """Get the pre-processed training set and test set (if applicable) of the next fold.

        :return: **yields** the pre-processed training set and test set (if applicable) of the next fold.
        :rtype:
            * train_objects: `pandas.DataFrame`
            * train_ranks: `pandas.DataFrame`
            * test_objects: `pandas.DataFrame` (if applicable) or None
            * test_ranks: `pandas.DataFrame` (if applicable) or None
        """
        for train_objects, train_ranks, test_objects, test_ranks in self._folds:
            yield train_objects, train_ranks, test_objects, test_ranks

    def is_training_only(self):
        """Indicate whether or not training only is to be applied on the given data."""
        return self._train_only

    def get_n_folds(self):
        """Get the number of folds in the data."""
        return self._n_folds

    def get_features(self):
        """Get the features defining the objects in the data.

        These are determined by looking at the features of the training objects in the first fold.
        """
        return self._features


class KFoldCrossValidation(Evaluator):
    """K-Fold Cross Validation."""
    # TODO: add longer description

    def __init__(self, k=3, test_folds=None):
        """Initializes the KFoldCrossValidation object.

        The dataset may be split into folds in two ways: automatically or manually. If automatic,
        the `k` argument is to be used. If manual, the user may specify the fold index for each sample in the
        dataset via the `test_folds` argument.

        :param k: the number of folds to uniformly split the data into when using the automatic approach (default 3).
        :type k: int, optional
        :param test_folds: an array specifying the fold index for each sample in the dataset when using
            the manual approach (default None). The entry test_folds[i] specifies the index of the test set that
            sample i belongs to. It is also possible to exclude sample i from any test set (i.e., include sample i
            in every training set) by setting test_folds[i] to -1.
            If `test_folds` is None, the automatic approach is assumed and only the `k` parameter is considered.
            Otherwise, the manual approach is assumed and only the `test_folds` parameter is considered.
        :type test_folds: `numpy.ndarray` or None, optional
        :raises InvalidParameterValueException: if a `k` parameter value less than 2 is used.
        """
        if (k is not None) and (k < 2):
            raise InvalidParameterValueException(parameter="k", value=k,
                                                 method="K-Fold Cross Validation (K-FCV)", is_algorithm=False,
                                                 additional_msg="K-FCV requires at least two folds. "
                                                                "Please choose a value of k=2 or higher.")

        desc = "K-Fold Cross Validation."  # TODO: add longer description
        self._k = k
        self._test_folds = test_folds

        # self._objects = None  # full objects (prior to fold-splitting)
        # self._ranks = None  # full ranks (prior to fold-splitting)
        self._dual_format = None
        # self._folds = None  # only used for manual splitting (via k_col)
        self._kf = None

        if self._test_folds is not None:  # k_col is given
            self._k = None
            self._kf = model_selection.PredefinedSplit(self._test_folds)  # k_codes
        else:  # only k is given
            self._kf = model_selection.KFold(n_splits=self._k, random_state=0)

        # call base class constructor
        super().__init__(description=desc, name=EvaluatorType.KFCV.name, k=self._k, manual_test_folds=self._test_folds)

    def split(self, data):
        """Get the indices for the training set and test set of the next fold of the dataset.

        If the single file format is used, the indices are given with respect to objects. Otherwise (if the dual
        file format is used), the indices are given with respect to the ranks.

        :param data: the data to be split into folds. If the single file format is used, a single `pandas.DataFrame`
            containing the data should be passed. If the dual file format is used, a tuple containing both the
            objects and ranks (each a `pandas.DataFrame`) should be passed.
        :type data: `pandas.DataFrame` or tuple of `pandas.DataFrame` (size 2)
        :return: **yields** two arrays containing the integer-based indices for the training set and test set of the
            next fold.
        :rtype:
            * train: `numpy.ndarray`
            * test: `numpy.ndarray`
        """
        if isinstance(data, tuple):  # dual file format (objects & ranks)
            self._dual_format = True
            objects = data[0].copy(deep=True)
            ranks = data[1].copy(deep=True)
        else:  # single file format
            self._dual_format = False
            objects = data.copy(deep=True)
            ranks = None

        if self._test_folds is not None:  # k_col is given
                folds = [(train, test) for train, test in self._kf.split(None)]
                # ^ pass None because we're using PredefinedSplit in this case
        else:  # only k is given
            if self._dual_format:  # dual file format (objects & ranks)
                # split by ranks into k folds
                folds = [(train, test) for train, test in self._kf.split(ranks)]
            else:  # single file format
                # split by objects into k folds
                folds = [(train, test) for train, test in self._kf.split(objects)]
        return folds
