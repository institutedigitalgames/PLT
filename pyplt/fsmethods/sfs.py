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

from pyplt.util.enums import FSMethod
from pyplt.fsmethods.wrappers import WrapperFSMethod


class SFS(WrapperFSMethod):
    """Sequential Forward Selection (SFS) method.

    SFS is a bottom-up hill-climbing algorithm where one feature is added at a time to the
    current feature set. The feature to be added is selected from the subset of the remaining
    features such that the new feature set generates the maximum value of the performance
    function over all candidate features for addition. The selection procedure begins with an
    empty feature set and terminates when an added feature yields equal or lower
    performance to the performance obtained without it. The performance of each subset of
    features considered is computed as the prediction accuracy of a model trained using
    that subset of features as input. All of the preference learning algorithms implemented
    in the tool can be used to train this model; i.e., RankSVM and Backpropagation.

    Extends the :class:`pyplt.fsmethods.wrappers.WrapperFSMethod` class which, in turn, extends the
    :class:`pyplt.fsmethods.base.FeatureSelectionMethod` class.
    """

    def __init__(self, verbose=True):
        """Initializes the feature selection method with the appropriate name and description.

        :param verbose: specifies whether or not to display detailed progress information to
            the `progress_window` if one is used (default True).
        :type verbose: bool
        """
        self._verbose = verbose
        desc = "Sequential Forward Selection method (SFS) is a bottom-up hill-climbing algorithm where one feature " \
               "is added at a time to the current feature set. The feature to be added is selected from the subset " \
               "of the remaining features such that the new feature set generates the maximum value of the " \
               "performance function over all candidate features for addition. The selection procedure begins with " \
               "an empty feature set and terminates when an added feature yields equal or lower performance to " \
               "the performance obtained without it. The performance of each subset of features considered is " \
               "computed as the prediction accuracy of a model trained using that subset of features as input. " \
               "All of the preference learning algorithms implemented in the tool can be used to train this model; " \
               "i.e., RankSVM and Backpropagation."
        super().__init__(description=desc, name=FSMethod.SFS.name)

    def select(self, objects, ranks, algorithm, test_objects=None, test_ranks=None,
               preprocessed_folds=None, progress_window=None, exec_stopper=None):
        """Carry out the feature selection process according to the SFS algorithm.

        :param objects: the objects data used to train the models used to evaluate and select features. If None, the
            data is obtained via the `preprocessed_folds` parameter instead.
        :type objects: `pandas.DataFrame` or None
        :param ranks: the pairwise rank data used to train the models used to evaluate and select features. If None, the
            data is obtained via the `preprocessed_folds` parameter instead.
        :type ranks: `pandas.DataFrame` or None
        :param algorithm: the algorithm used to train models to evaluate the features with (via the training accuracy).
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
            * the subset of features selected by SFS -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: list of str
        """
        if progress_window is not None:
            progress_window.put("Starting Sequential Feature Selection.")

        if objects is not None:
            self._orig_features = objects.columns
        else:
            self._orig_features = preprocessed_folds.get_features()

        # start with empty feat_set
        best_feat = None
        best_f = None
        best_perf = float("-inf")  # -infinity so at least 1 feature will be selected always
        feat_pool = list(self._orig_features.copy())
        # for each f in original features
        i = 0
        improved = True
        while improved:
            improved = False
            print("Iteration " + str(i) + ":")
            print("Selecting from the following feature pool:")
            print(feat_pool)
            algorithm.init_train(n_features=len(self._sel_features)+1)
            # ^ force reset of model topology used in algorithm to reflect reduced features!
            for f in range(len(feat_pool)):
                if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
                    # abort execution!
                    print("Aborting SFS execution...")
                    return
                feat = feat_pool[f]
                # add f to feat_set (temporarily)
                feat_set = self._sel_features + [feat]
                print("Training with feature set " + str(feat_set))
                # run pl algorithm with feat_set
                perf = self._evaluate(objects, ranks, feat_set, algorithm,
                                      test_objects=test_objects, test_ranks=test_ranks,
                                      preprocessed_folds=preprocessed_folds,
                                      exec_stopper=exec_stopper)
                # ^ no progress_window - do not show eval progress
                if perf is None:  # check if experiment was aborted
                    # abort execution!
                    print("Aborting SFS execution...")
                    return
                print("Feature " + str(feat) + ": " + str(perf))

                print("Evaluating features " + str(feat_set) + ": " + str(perf))
                if (progress_window is not None) and self._verbose:
                    progress_window.put("Evaluating features " + str(feat_set) + ": " + str(perf))

                if perf > best_perf:  # TODO: can make a better stopping criterion
                    improved = True
                    best_perf = perf
                    best_f = f
                    best_feat = feat
            # check if stop according to stopping criterion ^
            if improved:
                # add feature to selected features list
                self._sel_features += [best_feat]
                print("Selected feature " + str(best_feat) + ".")
                # remove feature from feature pool
                del feat_pool[best_f]
                print("Features selected so far: ")
                print(self._sel_features)
            else:
                print("No improvement... stopping SFS.")
                break
            i += 1

        if progress_window is not None:
            progress_window.put("Sequential Feature Selection complete.")

        return self._sel_features
