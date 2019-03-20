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

from pyplt.fsmethods.base import FeatureSelectionMethod


class WrapperFSMethod(FeatureSelectionMethod):
    """Parent class for all wrapper-type feature selection methods."""

    def __init__(self, description="A wrapper feature selection method.", **kwargs):
        """Initializes the wrapper-type feature selection method.

        :param description: a description of the feature selection method
            (default "A wrapper feature selection method.").
        :type description: str, optional
        :param kwargs: any additional parameters for the feature selection method.
        """
        super().__init__(description=description, **kwargs)

    def _evaluate(self, objects, ranks, feature_set, algorithm, test_objects=None, test_ranks=None,
                  preprocessed_folds=None, progress_window=None, exec_stopper=None, **kwargs):
        """Evaluate a given feature subset with respect to its predictive capacity in preference learning.

        Runs preference learning using a given algorithm and optionally a given evaluation method. The resulting
        test accuracy (if an evaluation method is also specified) or training accuracy (if no evaluation method
        is specified) comprises the metric by which the given feature set is evaluated.

        :param objects: the objects data used to train the models used to evaluate and select features. If None, the
            data is obtained via the `preprocessed_folds` parameter instead.
        :type objects: `pandas.DataFrame` or None
        :param ranks: the pairwise rank data used to train the models used to evaluate and select features. If None, the
            data is obtained via the `preprocessed_folds` parameter instead.
        :type ranks: `pandas.DataFrame` or None
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
        :return:
            * the prediction accuracy achieved using the given training algorithm / evaluation method over
              the given feature set -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: float
        """
        if objects is None or ranks is None:  # i.e. we have pre-processed fold data
            # use preprocessed_folds!
            tot_accuracy = 0.0
            if preprocessed_folds.is_training_only():  # train only
                for train_objects, train_ranks, test_objects, test_ranks in preprocessed_folds.next_fold():
                    # first train the model using the specified algorithm
                    success = algorithm.train(train_objects, train_ranks, use_feats=feature_set,
                                              progress_window=progress_window, exec_stopper=exec_stopper)
                    if success is None:  # check if execution was aborted!
                        print("Aborting feature selection execution...")
                        return
                    # no evaluation; training only
                    print("no eval")
                    # calculate & return training accuracy
                    train_acc = algorithm.calc_train_accuracy(train_objects, train_ranks, use_feats=feature_set,
                                                              progress_window=progress_window,
                                                              exec_stopper=exec_stopper)
                    if train_acc is None:  # check if execution was aborted!
                        print("Aborting feature selection execution...")
                        return
                    tot_accuracy = tot_accuracy + train_acc
            else:  # train & test
                for train_objects, train_ranks, test_objects, test_ranks in preprocessed_folds.next_fold():
                    # first train the model using the specified algorithm
                    success = algorithm.train(train_objects, train_ranks, use_feats=feature_set,
                                              progress_window=progress_window, exec_stopper=exec_stopper)
                    if success is None:  # check if execution was aborted!
                        print("Aborting feature selection execution...")
                        return
                    # test data specified; train & evaluate
                    print("eval")
                    # calculate & return test accuracy
                    test_acc = algorithm.test(test_objects, test_ranks, use_feats=feature_set,
                                              progress_window=progress_window, exec_stopper=exec_stopper)
                    if test_acc is None:  # check if execution was aborted!
                        print("Aborting feature selection execution...")
                        return
                    tot_accuracy = tot_accuracy + test_acc
            avg_accuracy = tot_accuracy / float(preprocessed_folds.get_n_folds())
            return avg_accuracy
        else:  # i.e. we have normal data
            # first train the model using the specified algorithm
            success = algorithm.train(objects, ranks, use_feats=feature_set,
                                      progress_window=progress_window, exec_stopper=exec_stopper)
            if success is None:  # check if execution was aborted!
                print("Aborting feature selection execution...")
                return
            # then see whether to test/evaluate or not
            if test_objects is None or test_ranks is None:
                # no evaluation; training only
                print("no eval")
                # calculate & return training accuracy
                return algorithm.calc_train_accuracy(objects, ranks, use_feats=feature_set,
                                                     progress_window=progress_window, exec_stopper=exec_stopper)
            else:
                # test data specified; train & evaluate
                print("eval")
                # calculate & return test accuracy
                result = algorithm.test(test_objects, test_ranks, use_feats=feature_set,
                                        progress_window=progress_window, exec_stopper=exec_stopper)
                if result is None:  # check if execution was aborted!
                    print("Aborting feature selection execution...")
                    return
                _, test_acc = result
                return test_acc
