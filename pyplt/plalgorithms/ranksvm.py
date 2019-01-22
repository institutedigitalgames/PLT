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
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

from pyplt import ROOT_PATH
from pyplt.exceptions import InvalidParameterValueException
from pyplt.util.enums import KernelType, PLAlgo
from pyplt.plalgorithms.base import PLAlgorithm


class RankSVM(PLAlgorithm):
    """RankSVM algorithm implemented using the `scikit-learn` library.

    A Support Vector Machine (SVM) is a binary classifier that separates the input put samples linearly in a
    projected space. The decision boundary of the classifier is given by a linear combination of training samples
    (called support vectors) in the projected space. The projection in provided by the kernel function that the
    user must select. The support vector and weights are selected to satisfy a set of constrains derived from the
    input samples and a cost parameter which regulates the penalization of misclassified training samples.
    In PLT, the algorithm was implemented using the `scikit-learn` library. In this implementation, the quadratic
    programmer solver contained in LIBSVM is used. The RankSVM algorithm is a rank-based version of traditional
    SVM training algorithms. It uses the same solver as standard training algorithms for binary SVMs; the only
    difference lies in the set of constraints which are defined in terms of pairwise preferences between
    training samples.
    """

    def __init__(self, kernel=KernelType.RBF, gamma='auto', degree=3, debug=False):
        """Initializes the RankSVM object.

        :param kernel: the kernel function mapping the input samples to the projected space (default
            :attr:`pyplt.util.enums.KernelType.RBF`).
        :type kernel: :class:`pyplt.util.enums.KernelType`, optional
        :param gamma: the kernel coefficient for the ‘rbf’, ‘poly’ and ‘sigmoid’ kernels. If gamma
            is set to ‘auto’ then 1/n_features will be used instead (default 'auto').
        :type gamma: float or 'auto', optional
        :param degree: the degree of the polynomial (‘poly’) kernel function (default 3).
        :type degree: float, optional
        :param debug: specifies whether or not to print notes to console for debugging (default False).
        :type debug: bool, optional
        :raises InvalidParameterValueException: if the user attempts to use a gamma value <= 0.0.
        """
        desc = "A Support Vector Machine (SVM) is a binary classifier that separates the input put samples " \
               "linearly in a projected space. The decision boundary of the classifier is given by a linear " \
               "combination of training samples (called support vectors) in the projected space. The projection " \
               "in provided by the kernel function that the user must select. The support vector and weights are " \
               "selected to satisfy a set of constrains derived from the input samples and a cost parameter which " \
               "regulates the penalization of misclassified training samples. In PLT, the algorithm was implemented " \
               "using the scikit-learn library. In this implementation, the quadratic programmer solver contained " \
               "in LIBSVM is used. The RankSVM algorithm is a rank-based version of traditional SVM training " \
               "algorithms. It uses the same solver as standard training algorithms for binary SVMs; the only " \
               "difference lies in the set of constraints which are defined in terms of pairwise preferences between " \
               "training samples."

        if gamma != 'auto':
            if float(gamma) <= 0.0:
                raise InvalidParameterValueException(parameter="gamma", value=gamma,
                                                     method="RankSVM", is_algorithm=True,
                                                     additional_msg="Gamma must have a value greater than 0.0.")

        self._kernel = kernel.name
        if self._kernel == KernelType.RBF.name or self._kernel == KernelType.POLY.name:
            self._gamma = gamma
        else:
            self._gamma = None
        if self._kernel == KernelType.POLY.name:
            self._degree = degree
        else:
            self._degree = None
        self._r_svm = None
        self._real_objects = None
        self._real_ranks = None
        super().__init__(description=desc, name=PLAlgo.RANKSVM.name, kernel=self._kernel,
                         gamma=self._gamma, degree=self._degree, debug=debug)

    def train(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Train a RankSVM model on the given training data.

        :param train_objects: the objects data to train the model on.
        :type train_objects: `pandas.DataFrame`
        :param train_ranks: the pairwise rank data to train the model on.
        :type train_ranks: `pandas DataFrame`
        :param use_feats: a subset of the original features to be used when training;
            if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display
            a progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return: None -- if experiment is aborted before completion by `exec_stopper`.
        """
        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training execution...")
            return

        # K = matrix of precomputed kernels (shape = n_ranks x n_ranks)
        # y_trans = array of +1s for each rank (shape = 1 x n_ranks)
        print("precomputing kernels...")
        if progress_window is not None:
            progress_window.put("Pre-computing kernels for RankSVM.")
        result = self._transform_dataset(train_objects, train_ranks, use_feats, progress_window, exec_stopper)
        if result is None:  # check if experiment was aborted
            # abort execution!
            print("K is None ... Aborting RankSVM training execution...")
            return
        else:
            K, y_trans, real_objects = result

        print("precomputation of kernels complete.")
        if progress_window is not None:
            progress_window.put("Kernel pre-computation for RankSVM complete.")
        self._real_objects = real_objects.copy()
        self._real_ranks = train_ranks.copy()

        print("Starting training with RankSVM.")
        if progress_window is not None:
            progress_window.put("Starting training with RankSVM.")

        # training...
        self._r_svm = svm.OneClassSVM(kernel='precomputed', tol=1e-3, shrinking=True,
                                      cache_size=40, max_iter=5000, nu=0.5)

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training execution...")
            return

        self._r_svm.fit(K, y=y_trans)
        print("Training complete.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training execution...")
            return

        print("num of svs: ")
        print(len(self._r_svm.support_))
        print("sv indexes: ")
        print(self._r_svm.support_)
        # ... e.g. [  2   3   4   7  14  15  18  19  20  21  23  26  28  29  30  31  33  34
        #   35  37  38  39  42  43  45  47  52  54  55  56  57  58  59  62  63  65
        #   66  67  68  69  71  72  73  74  76  80  81  82  83  84  85  86  87  88
        #   89  91  93  94  95  97  98  99 100 101 105 106 108 109 111 112 114 115
        #  119 120 121 122 123 125 126 127 129 130 131 133 135 136 137 139 140 142
        #  143 144 146 148 149 150 151 154 155 157 159 160 162 163 164 165 166 167
        #  168 169 170 171 173 174 175 176 177 178 179 181 182 183 184 187 188 189
        #  191 192 194 195 196 198 199 200 201 205]
        # ^ equivalent to IDS OF RANKS in the training ranks set

        if self._debug:
            # pref & non feature vectors of the ranks constituting the support vectors
            sv_ranks = self._real_ranks.iloc[self._r_svm.support_].values
            sv_prefs = train_objects.loc[sv_ranks[:, 0]].values
            sv_nons = train_objects.loc[sv_ranks[:, 1]].values

            for r in range(len(sv_ranks)):
                print("sv_" + str(r) + ": " + str(sv_ranks[r, 0]) + " > " + str(sv_ranks[r, 1]))
                print("alpha: " + str(self._r_svm.dual_coef_[0, r]))
                print("pref_obj:")
                print(sv_prefs[r, :])
                print("non_pref_obj:")
                print(sv_nons[r, :])

        print("sv alphas: ")
        print(self._r_svm.dual_coef_)

        if progress_window is not None:
            progress_window.put("Training complete.")

    @staticmethod
    def transform_data(object_):
        """Transform an object into the format required by this particular implementation of RankSVM.

        :param object_: the object to be transformed.
        :type object_: one row from a `pandas.DataFrame`
        :return: the transformed object in the form of an array.
        :rtype: `numpy.ndarray`
        """
        return np.asarray(object_)

    def _transform_dataset(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Convert the data set for use by RankSVM prior to the training stage.

        The kernels of the training data are precomputed such that each row or column in the dataset corresponds
        to a rank (object pair) in train_ranks and thus each cell corresponds to a the value Qij in pg.3 of
        Herbrich et al. 1999 (Support Vector Learning for Ordinal Regression). This allows us to enforce the rank-based
        constraints of RankSVM, without having to modify the OneClassSVM algorithm. Additionally, a value of +1 is
        stored as the target class/label for each rank (object pair) in the training set.

        :param train_objects: the objects data to be converted.
        :type train_objects: `pandas.DataFrame`
        :param train_ranks: the pairwise rank data to be converted.
        :type train_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used when training (default None). If None, all
            original features are used.
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * a tuple containing the precomputed kernel matrix K, the array of target classes/labels y_trans of
              shape (k,) (in this case, all +1s), and the `pandas.DataFrame` of training objects containing only the
              features specified by `use_feats` -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: tuple (size 3)
        """
        if use_feats is None:
            train_objects_ = train_objects.copy()
        else:
            train_objects_ = train_objects.loc[:, use_feats].copy()
        train_ranks_ = train_ranks.copy()

        n_ranks = len(train_ranks_)

        y_trans = np.ones((n_ranks,), dtype=int)

        # matrix-based calculation
        X_i = train_ranks_.values
        X_j = train_ranks_.values
        K = self._ranks_kernel_m(train_objects_, X_i, X_j)

        # debug
        # print("train_ranks_.shape = " + str(train_ranks_.shape))
        # print("X_i.shape = " + str(X_i.shape))
        # print(X_i)
        # print("X_j.shape = " + str(X_j.shape))

        # check if experiment was aborted
        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training execution...")
            return
        # otherwise send hack-y request to progress_window to update gui
        elif progress_window is not None:
            progress_window.update_gui()

        # convert all values in K to float32 to compare with Java....
        # K = K.astype(np.float32)

        # print("shape(K) = " + str(K.shape))
        # print(K)

        return K, y_trans, train_objects_

    def _ranks_kernel_m(self, train_objects, X_i, X_j):
        """Pre-compute the n x n rank-based kernel matrix for the set of n training ranks.

        This method embeds the rank-based constraints that makes this implementation of SVM rank-based, without having
        to modify the algorithm itself. The values of each cell [i, j] in the output matrix corresponds to the value
        Qij in pg.3 of Herbrich et al. 1999 (Support Vector Learning for Ordinal Regression).

        :param train_objects: the training objects.
        :type train_objects: `pandas.DataFrame`
        :param X_i: array i of shape [n, 2] containing a copy the training ranks (i.e., pairs of object IDs).
        :type X_i: `numpy.ndarray`
        :param X_j: array j of shape [n, 2] containing another copy of the training ranks (i.e., pairs of object IDs).
        :type X_j: `numpy.ndarray`
        :return: the resulting kernel matrix K of shape [n, n].
        :rtype: `numpy.ndarray`
        """
        # aka "Value" equation from Java PLT's svm.java
        # get the feature vectors (lists) of the actual objects from each rank (i.e. pair of object ids)
        # pref & non feature vectors for rank x_i:
        X_i_prefs = train_objects.loc[X_i[:, 0]].values
        X_i_nons = train_objects.loc[X_i[:, 1]].values
        # pref & non feature vectors for rank x_j:
        X_j_prefs = train_objects.loc[X_j[:, 0]].values
        X_j_nons = train_objects.loc[X_j[:, 1]].values

        # debug
        # print("X_i_prefs.shape = " + str(X_i_prefs.shape))
        # print(X_i_prefs)
        # print("X_i_nons.shape = " + str(X_i_nons.shape))

        # aka "resulting equation" in Java PLT's svm.java
        return np.subtract(np.subtract(np.add(self._kernel_base_m(X_i_prefs, X_j_prefs),
                                              self._kernel_base_m(X_i_nons, X_j_nons)),
                                       self._kernel_base_m(X_i_prefs, X_j_nons)),
                           self._kernel_base_m(X_i_nons, X_j_prefs))

    def _kernel_base_m(self, A, B, reshape_a=False, reshape_b=False):
        """Compute the kernel function (:meth:`self._kernel()`) on each corresponding pair of objects in A and B.

        Internally uses the corresponding kernel functions in `sklearn.metrics.pairwise`.

        :param A: matrix of shape [n_samples_A, n_features] containing the feature vectors of objects in A
        :type A: `numpy.ndarray`
        :param B: matrix of shape [n_samples_B, n_features] containing the feature vectors of objects in B
        :type B: `numpy.ndarray`
        :param reshape_a: specifies whether to reshape input A into an array of shape (1, -1) indicating
            a single sample (default: False).
        :type reshape_a: bool, optional
        :param reshape_b: specifies whether to reshape input B into an array of shape (1, -1) indicating
            a single sample (default: False).
        :type reshape_b: bool, optional
        :return: a matrix of shape [n_samples_A, n_samples_B] containing the float output of the kernel function
            for each AxB object pair.
        :rtype: `numpy.ndarray`
        """
        # sklearn tells us to "Reshape your data either using array.reshape(-1, 1) if your data has a single feature
        # or array.reshape(1, -1) if it contains a single sample."
        # when predicting A=input_object, our case is the latter (single sample).
        # print("A.shape = " + str(A.shape))
        # print("B.shape = " + str(B.shape))
        if reshape_a:
            A = A.reshape(1, -1)
            # print("A.shape after reshape = " + str(A.shape))
        if reshape_b:
            B = B.reshape(1, -1)
            # print("B.shape after reshape = " + str(B.shape))

        # convert gamma 'auto' into None for rbf_kernel
        gamma = self._gamma
        if gamma == 'auto':
            gamma = None

        # calculate the kernel
        if self._kernel == KernelType.LINEAR.name:
            return linear_kernel(A, B)
        elif self._kernel == KernelType.POLY.name:
            return polynomial_kernel(A, B, degree=self._degree, gamma=gamma, coef0=0)
            # TODO: make coef0 a user-defined parameter!!!
        else:  # i.e. KernelType.RBF.name
            return rbf_kernel(A, B, gamma=gamma)
        # TODO: do the same for any other kernels! (e.g. sigmoid)

    def save_model(self, timestamp, path="", suppress=False):
        """Save the RankSVM model to a Comma Separated Value (CSV) file at the path indicated by the user.

        Optionally, the file creation may be suppressed and a `pandas.DataFrame` representation of the model
        returned instead.

        The file/DataFrame stores support vectors and corresponding alpha values of the SVM model.

        The first column contains the support vectors each representing a rank in the form of a tuple (int, int)
        containing the ID of the preferred object in the pair, followed by the ID of the non-preferred object in
        the pair.
        The second column contains the alpha values corresponding to the support vectors in the first column.

        The parameters (kernel, gamma and degree) used to construct the model are stored within the file name.

        :param timestamp: the timestamp to be included in the file name.
        :type timestamp: float
        :param path: the path at which the file is to be saved (default ""). If "", the file is saved to a logs folder
            in the project root directory by default. The kernel, gamma, and degree parameters
            are automatically included in the file name.
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

        params = "kernel" + self._kernel
        if (self._kernel == KernelType.RBF.name) or (self._kernel == KernelType.POLY.name):
            params += ("_gamma" + str(self._gamma))
        if self._kernel == KernelType.POLY.name:
            params += ("_degree" + str(self._degree))

        # Now, format model for saving to file
        if path == "":
            path = os.path.join(ROOT_PATH, "logs\\model_" + str(timestamp) + "_" + params + ".csv")
        else:
            # add params to file name
            path = path.rstrip(".csv")  # remove .csv
            path += ("_" + params + ".csv")  # add params and .csv

        cols = ["Support_Vector", "Alpha"]

        num_svs = len(self._r_svm.support_)
        svs = self._r_svm.support_
        sv_alphas = self._r_svm.dual_coef_

        model_arr = np.empty(shape=[num_svs, len(cols)], dtype=object)

        model_arr[:, 0] = [tuple(self._real_ranks.iloc[i].values) for i in svs]
        model_arr[:, 1] = sv_alphas

        model_df = pd.DataFrame(model_arr, columns=cols)

        if suppress:
            return model_df
        else:
            # Finally, save to file!
            model_df.to_csv(path, index=False)

    def predict_m(self, input_objects, progress_window=None, exec_stopper=None):
        """Predict the output of a given set of input samples by running them through the learned RankSVM model.

        :param input_objects: array of shape [n_samples, n_feats] containing the input data corresponding
            to a set of (test) objects.
        :type input_objects: `numpy.ndarray`
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a progress
            log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * a list containing the average predicted output resulting from running the learned model using the
              given input objects -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: list of float (size 1)
        """
        sv_idx = self._r_svm.support_  # indices of the support vectors
        sv_coefs = self._r_svm.dual_coef_  # coefficients of the support vectors in the decision function

        # matrix-based calculation
        Alphas_i = sv_coefs[0, :]  # shape = [1, n_sv]
        # only where alpha_i != 0 !!!
        non_zeros = Alphas_i != 0  # these are the indexes of the rows in Alphas_i where the value is !=0
        sv_idx = sv_idx[non_zeros]
        Alphas_i = Alphas_i[non_zeros]  # same as Alphas_i[Alphas_i != 0]
        # reshape Alphas_i from (n_sv, ) to (1, n_sv) so that it works properly with np.matmul() later
        Alphas_i = Alphas_i.reshape(1, -1)

        X_i = self._real_ranks.iloc[sv_idx].values  # shape = [n_sv, 2]
        req_utility = np.matmul(Alphas_i, np.transpose(self._predict_kernel_subtraction_m2(self._real_objects,
                                                                                           input_objects,
                                                                                           X_i)))
        # print("req_utility.shape = " + str(req_utility.shape))
        # final result = [1, n_samples]

        # debug
        # print("number alphas EXCLUDING zeros: " + str(len(Alphas_i)))
        # print("number svs EXCLUDING alpha=zeros: " + str(len(sv_idx)))
        # print("reshpaed Alphas_i.shape = " + str(Alphas_i.shape))
        # print("X_i.shape = " + str(X_i.shape))
        # print("req_utility.shape = " + str(req_utility.shape))
        # print(req_utility)
        # print("summed req_utility.shape = " + str(req_utility.shape))

        # check if experiment was aborted
        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM execution...")
            return
        # otherwise send hack-y request to progress_window to update gui
        elif progress_window is not None:
            progress_window.update_gui()

        return [req_utility]  # wrap around with a list in order for calc_train_accuracy() in base class to work

    def test(self, objects, test_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to testing/validating the model using the given test data.

        :param objects: the objects data that the model was trained on.
        :type objects: `pandas.DataFrame`
        :param test_ranks: the pairwise rank data for the model to be tested/validated on.
        :type test_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used during the testing/validation
            process; if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the test/validation accuracy of the learned model -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: float
        """
        if self._debug:
            print("Calculating TEST accuracy...")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM test execution...")
            return

        if use_feats is None:
            test_prefs_obj = objects.loc[test_ranks.iloc[:, 0], :].values
            test_nons_obj = objects.loc[test_ranks.iloc[:, 1], :].values
        else:
            test_prefs_obj = objects.loc[test_ranks.iloc[:, 0], use_feats].values
            test_nons_obj = objects.loc[test_ranks.iloc[:, 1], use_feats].values

        # print("test_prefs_obj.shape: " + str(test_prefs_obj.shape))  # shape = [n_test_ranks, n_feats]
        # print("test_nons_obj.shape: " + str(test_nons_obj.shape))  # shape = [n_test_ranks, n_feats]

        prefs_accuracies = self.predict_m(test_prefs_obj, progress_window=progress_window, exec_stopper=exec_stopper)
        nons_accuracies = self.predict_m(test_nons_obj, progress_window=progress_window, exec_stopper=exec_stopper)

        if (prefs_accuracies is None) or (nons_accuracies is None):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM test execution...")
            return

        total_correct = np.greater(prefs_accuracies, nons_accuracies)
        total_correct = np.sum(total_correct)
        accuracy = float(total_correct) / float(len(test_ranks)) * 100

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM test execution...")
            return

        return accuracy

    def _predict_kernel_subtraction_m2(self, train_objects, input_object, X_i):
        """Compute the subtraction part of deriving the order of the given object with respect to the given support vectors.

        :param train_objects: the objects used to train the SVM.
        :type train_objects: `pandas.DataFrame`
        :param input_object: array of shape [n_samples, n_feats] containing the feature vectors of the
            set of (test) objects to be predicted.
        :type input_object: `numpy.ndarray`
        :param X_i: array of shape [n_support_vectors, 2] containing RankSVM model's support vectors (each in the
            form of a pair of object IDs).
        :type X_i: `numpy.ndarray`
        :return: matrix of shape [n_samples, n_support_vectors] containing the float results of the kernel
            subtraction for each support vector for each sample in the given (test) set of objects.
        :rtype: `numpy.ndarray`
        """
        # get the feature vectors (lists) of the actual objects from each rank (i.e. pair of object ids)
        # pref & non feature vectors for rank x_i:
        X_i_prefs = train_objects.loc[X_i[:, 0]].values
        X_i_nons = train_objects.loc[X_i[:, 1]].values

        # print("X_i_prefs.shape = " + str(X_i_prefs.shape))
        # print(X_i_prefs)
        # print("input_object.shape = " + str(input_object.shape))

        # returns shape = [n_rows * n_rows] aka [n_sv, n_sv] usually
        answer = np.subtract(self._kernel_base_m(input_object, X_i_prefs),
                           self._kernel_base_m(input_object, X_i_nons))
        # print("answer.shape = " + str(answer.shape))
        return answer

    def calc_train_accuracy(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to calculates the training accuracy of the learned model.

        This method is tailored specifically for this algorithm implementation and therefore replaces
        the calc_train_accuracy() method of :class:`pyplt.plalgorithms.base.PLAlgorithm`.

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
            print("Calculating train accuracy...")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training accuracy execution...")
            return

        print("running RankSVM-specific calc_train_accuracy() method...")
        if use_feats is None:
            train_prefs_obj = train_objects.loc[train_ranks.iloc[:, 0], :].values
            train_nons_obj = train_objects.loc[train_ranks.iloc[:, 1], :].values
        else:
            train_prefs_obj = train_objects.loc[train_ranks.iloc[:, 0], use_feats].values
            train_nons_obj = train_objects.loc[train_ranks.iloc[:, 1], use_feats].values

        # print("train_prefs_obj.shape: " + str(train_prefs_obj.shape))  # shape = [n_train_ranks, n_feats]
        # print("train_nons_obj.shape: " + str(train_nons_obj.shape))  # shape = [n_train_ranks, n_feats]

        prefs_accuracies = self.predict_m(train_prefs_obj, progress_window=progress_window,
                                          exec_stopper=exec_stopper)
        nons_accuracies = self.predict_m(train_nons_obj, progress_window=progress_window, exec_stopper=exec_stopper)

        if (prefs_accuracies is None) or (nons_accuracies is None):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training accuracy execution...")
            return

        total_correct = np.greater(prefs_accuracies, nons_accuracies)
        total_correct = np.sum(total_correct)
        accuracy = float(total_correct) / float(len(train_ranks)) * 100

        if progress_window is not None:
            progress_window.put("Training accuracy: " + str(accuracy) + "%")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training accuracy execution...")
            return

        return accuracy
