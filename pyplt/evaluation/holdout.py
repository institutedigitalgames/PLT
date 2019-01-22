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

import numpy as np

from pyplt.util.enums import EvaluatorType
from pyplt.evaluation.base import Evaluator


class HoldOut(Evaluator):
    """Holdout evaluator.

    This evaluation method splits the pairwise rank data into a training set and a test set. The training set
    is used to train the model via preference learning whereas the test set is used to estimate the prediction
    accuracy of the model. Often, 70% of the data is used as the training set while the remaining 30% is used
    as the test set (i.e., a test proportion of 0.3) however the user may choose a different proportion.
    """

    def __init__(self, test_proportion=0.3, debug=False):
        """Initializes the HoldOut object.

        :param test_proportion: the proportion of data to be used as the test set; the remaining data is used as
            the training data (default 0.3).
        :type test_proportion: float, optional
        :param debug: specifies whether or not to print notes to console for debugging (default False).
        :type debug: bool, optional
        """
        desc = "The Holdout evaluation method splits the pairwise rank data into a training set and a test set. " \
               "The training set is used to train the model via preference learning whereas the test set is used " \
               "to estimate the prediction accuracy of the model. Often, 70% of the data is used as the training " \
               "set while the remaining 30% is used as the test set (i.e., a test proportion of 0.3) however the " \
               "user may choose a different proportion."
        self._split_idx = 0
        self._test_proportion = test_proportion
        self._dual_format = None
        super().__init__(description=desc, name=EvaluatorType.HOLDOUT.name,
                         test_proportion=test_proportion, debug=debug)

    # Internal methods

    def split(self, data):
        """Split the given dataset into a training set and a test set according to the given proportion parameter.

        If the single file format is used, the indices are given with respect to objects. Otherwise (if the dual
        file format is used), the indices are given with respect to the ranks.

        :param data: the data to be split into folds. If the single file format is used, a single `pandas.DataFrame`
            containing the data should be passed. If the dual file format is used, a tuple containing both the
            objects and ranks (each a `pandas.DataFrame`) should be passed.
        :type data: `pandas.DataFrame` or tuple of `pandas.DataFrame` (size 2)
        :return: two arrays containing the indices for the training set and test set.
        :rtype:
            * train: `numpy.ndarray`
            * test: `numpy.ndarray`
        """

        if isinstance(data, tuple):  # dual file format (objects & ranks)
            data = data[1].copy(deep=True)  # ranks
            self._dual_format = True
        else:  # single file format
            data = data.copy(deep=True)  # samples/objects
            self._dual_format = False

        n = data.shape[0]  # len(data)

        self._split_idx = round((1.0-self._test_proportion) * n)
        # ^ ensure integer split point rather than float!!

        if self._debug:
            print("splitting at " + str(self._split_idx) + " of " + str(n))
        # train = data.iloc[:self._split_idx, :]  # 0 to split-1
        # test = data.iloc[self._split_idx:, :]  # split to len-1
        train = np.arange(0, self._split_idx)  # 0 to split-1
        test = np.arange(self._split_idx, n)  # split to len-1

        return train, test
