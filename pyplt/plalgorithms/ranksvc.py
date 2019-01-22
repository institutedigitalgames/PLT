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
from sklearn.metrics import accuracy_score

from pyplt import ROOT_PATH
from pyplt.exceptions import InvalidParameterValueException
from pyplt.util.enums import KernelType, PLAlgo
from pyplt.plalgorithms.base import PLAlgorithm


class RankSVC(PLAlgorithm):
    """RankSVM algorithm implemented using the `scikit-learn` library.

    **N.B.** This implementation is similar to the implementation in the :class:`pyplt.plalgorithms.ranksvm.RankSVM`
    class but instead of using the `OneClassSVM` class of the `scikit-learn` libary, this implementation uses
    the `SVC` class of the same library. The input and output of the model are treated differently as the SVC model
    is a binary classifier (see :meth:`pairwise_transform_from_ranks()`). Consequently, unlike the RankSVM
    implementation, the model cannot predict a real-valued output for a single object/instance. Rather, the
    model can only be used on pairs of objects in order for the output to make sense. This implementation is only
    available in the API of PLT.

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

        print("transforming dataset...")
        if progress_window is not None:
            progress_window.put("Transforming dataset for RankSVM.")
        X_train_diff, y_train_pref, real_objects = self.pairwise_transform_from_ranks(train_objects, train_ranks,
                                                                                      use_feats)

        print("dataset transformation complete.")
        if progress_window is not None:
            progress_window.put("Dataset transformation for RankSVM complete.")
        self._real_objects = real_objects.copy()
        self._real_ranks = train_ranks.copy()

        print("Starting training with RankSVM.")
        if progress_window is not None:
            progress_window.put("Starting training with RankSVM.")

        # training...
        if self._kernel == KernelType.RBF.name:
            self._r_svm = svm.SVC(C=1, random_state=0, gamma=self._gamma,
                                  kernel=self._kernel.lower(), verbose=False,
                                  tol=1e-3, shrinking=True, cache_size=40, max_iter=5000)
        elif self._kernel == KernelType.POLY.name:
            self._r_svm = svm.SVC(C=1, random_state=0, gamma=self._gamma, degree=self._degree,
                                  kernel=self._kernel.lower(), verbose=False,
                                  tol=1e-3, shrinking=True, cache_size=40, max_iter=5000)
        else:  # LINEAR
            self._r_svm = svm.SVC(C=1, random_state=0, kernel=self._kernel.lower(), verbose=False,
                                  tol=1e-3, shrinking=True, cache_size=40, max_iter=5000)

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training execution...")
            return

        self._r_svm.fit(X_train_diff, y_train_pref)
        print("Training complete.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training execution...")
            return

        # if self._debug:
        print("num of svs: ")
        print(len(self._r_svm.support_))
        print("sv indexes: ")
        print(self._r_svm.support_)
        # ... e.g. [  1   3   5   7   9  11  13  15  17  19  21  23  25  27  29  31  33  35
        #   37  39  41  43  45  47  49  51  53  55  57  59  61  63  65  67  69  71
        #   73  75  77  79  81  83  85  87  89  91  93  95  97  99 101 103 105 107
        #  109 111 113 115 117 119 121 123 125 127 129 131 133 135 137 139 141 143
        #  145 147 149 151 153 155 157 159 161 163 165 167 169 171 173 175 177 179
        #  181 183 185 187 189 191 193 195 197 199 201 203 205 207 209 211 213 215
        #  217 219 221 223 225 227 229 231 233 235 237 239 241 243 245 247 249 251
        #  253 255 257 259 261 263 265 267 269 271 273 275 277 279 281 283 285 287
        #  289 291 293 295 297 299 301 303 305 307 309 311 313 315 317 319 321 323
        #  325 327 329 331 333 335 337 339 341 343 345 347 349 351 353 355 357 359
        #  361 363 365 367 369 371 373 375 377 379 381 383 385 387 389 391 393 395
        #  397 399 401 403 405 407 409 411   0   2   4   6   8  10  12  14  16  18
        #   20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54
        #   56  58  60  62  64  66  68  70  72  74  76  78  80  82  84  86  88  90
        #   92  94  96  98 100 102 104 106 108 110 112 114 116 118 120 122 124 126
        #  128 130 132 134 136 138 140 142 144 146 148 150 152 154 156 158 160 162
        #  164 166 168 170 172 174 176 178 180 182 184 186 188 190 192 194 196 198
        #  200 202 204 206 208 210 212 214 216 218 220 222 224 226 228 230 232 234
        #  236 238 240 242 244 246 248 250 252 254 256 258 260 262 264 266 268 270
        #  272 274 276 278 280 282 284 286 288 290 292 294 296 298 300 302 304 306
        #  308 310 312 314 316 318 320 322 324 326 328 330 332 334 336 338 340 342
        #  344 346 348 350 352 354 356 358 360 362 364 366 368 370 372 374 376 378
        #  380 382 384 386 388 390 392 394 396 398 400 402 404 406 408 410]
        # ^ equivalent to IDS OF RANKS x 2 in the training ranks set

        # print("(RankSVM) self._real_ranks: ")
        # print(self._real_ranks)
        #
        # # TODO: N.B. self._r_svm.support_ are not IDs of ranks but IDs of objects in ranks (starting with
        # non-preferred objects, followed by all preferred objects) i.e. can amount to as many as n_ranks*2.
        # if self._debug:
        #     # pref & non feature vectors of the ranks constituting the support vectors
        #     sv_ranks = self._real_ranks.iloc[self._r_svm.support_].values
        #     sv_prefs = train_objects.loc[sv_ranks[:, 0]].values
        #     sv_nons = train_objects.loc[sv_ranks[:, 1]].values
        #
        #     for r in range(len(sv_ranks)):
        #         print("sv_" + str(r) + ": " + str(sv_ranks[r, 0]) + " > " + str(sv_ranks[r, 1]))
        #         print("alpha: " + str(self._r_svm.dual_coef_[0, r]))
        #         print("pref_obj:")
        #         print(sv_prefs[r, :])
        #         print("non_pref_obj:")
        #         print(sv_nons[r, :])

        # if self._debug:
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

    def pairwise_transform_from_ranks(self, objects, ranks, use_feats=None):
        """Convert a rank-based dataset into the required format for use by RankSVM prior to the training stage.

        For each rank (pair of objects) in `ranks`, a feature vector subtraction is carried out between the two objects
        (both feature vectors) from either side (i.e., a-b and b-a for a given pair of objects/feature vectors
        a and b where a is preferred over b) and stored as a new transformed data point in `X_trans`.
        Additionally, for each positive difference (a-b), a value of +1 is stored as its corresponding target class
        label in `y_trans` whereas value of -1 is stored for each negative difference (b-a).

        :param objects: the objects data to be converted.
        :type objects: `pandas.DataFrame`
        :param ranks: the pairwise rank data to be converted.
        :type ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used when training (default None). If None, all
            original features are used.
        :type use_feats: list of str or None, optional
        :return: a tuple containing:

            * the converted dataset ready to be used by RankSVM in the form of two arrays:

                * array of shape (n_ranks*2, n_feaures) which stores the positive and negative feature vector
                  differences for each rank.
                * array of shape n_ranks*2 which stores the corresponding target class labels
                  (alternating +1s and -1s).
            * a copy of the actual objects data (`pandas.DataFrame`) used in the transformation.
        :rtype: tuple (size 3)
        """
        if use_feats is None:
            objects_ = objects.copy()
        else:
            objects_ = objects.loc[:, use_feats].copy()
        ranks_ = ranks.values

        pref_x = objects_.loc[ranks_[:, 0]].values
        non_x = objects_.loc[ranks_[:, 1]].values
        # for each rank (pair of objects) a, b in the dataset (where a is preferred over b),
        # we create a positive example and a negative example for the new transformed dataset
        # the feature vector of a positive example is the positive feature vector difference a - b
        # the feature vector of a negative example is the negative feature vector difference b - a
        X_pos = np.subtract(pref_x, non_x)  # positive examples i.e. the (a - b)s
        X_neg = np.subtract(non_x, pref_x)  # negative examples i.e. the (b - a)s
        # alternate positive examples with negative examples for each rank (pair of objects) in the dataset
        # we do this since this is faster than looping through each rank and doing it individually
        X_trans = np.empty((X_pos.shape[0]+X_neg.shape[0], X_pos.shape[1]))
        X_trans[::2, :] = X_pos  # feature vectors of positive examples
        X_trans[1::2, :] = X_neg  # feature vectors of negative examples
        y_trans = np.empty(X_pos.shape[0]+X_neg.shape[0])
        y_trans[::2] = 1  # target class of positive examples
        y_trans[1::2] = -1  # target class of negative examples

        # print("X_trans:")
        # print(X_trans)
        # print("y_trans:")
        # print(y_trans)

        X_trans = pd.DataFrame(X_trans).reset_index(drop=True)
        y_trans = pd.DataFrame(y_trans, columns=['preference']).reset_index(drop=True)

        # print("X_trans DataFrame:")
        # print(X_trans)
        # print("y_trans DataFrame:")
        # print(y_trans)

        return X_trans, y_trans, objects_

    def save_model(self, timestamp, path="", suppress=False):
        """Save the RankSVM model to a Comma Separated Value (CSV) file at the path indicated by the user.

        Optionally, the file creation may be suppressed and a `pandas.DataFrame` representation of the model
        returned instead.

        The file/DataFrame stores support vectors and corresponding alpha values of the SVM model.

        The first column contains the support vectors each representing an object ID.
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

        # model_arr[:, 0] = [tuple(self._real_ranks.iloc[i].values) for i in svs]
        model_arr[:, 0] = [self._real_ranks.iloc[i // 2, i % 2] for i in svs]
        # i//2 is integer division (without remainder) meaning
        # 0 & 1 will give (rank) 0; 2 & 3 will give (rank) 1; 4 & 5 will give (rank) 2; etc.
        # i%2 is the remainder of the division meaning
        # 0, 2, 4... will give (rank column index) 0 i.e. PREFS; 1, 3, 5... will give (rank column index) 1 i.e. NONS
        # this will result in the actual object ID corresponding to each support vector

        model_arr[:, 1] = sv_alphas

        model_df = pd.DataFrame(model_arr, columns=cols)

        if suppress:
            return model_df
        else:
            # Finally, save to file!
            model_df.to_csv(path, index=False)

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

        X_test_diff, y_test_pref, real_objects = self.pairwise_transform_from_ranks(objects, test_ranks, use_feats)
        test_pred = self._r_svm.predict(X_test_diff)
        test_acc = accuracy_score(y_test_pref, test_pred) * 100

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM test execution...")
            return

        return test_acc

    def calc_train_accuracy(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to calculates the training accuracy of the learned model.

        This method is tailored specifically for this algorithm implementation and therefore replaces
        the calc_train_accuracy() method of :class:`pyplt.plalgorithms.base.PLAlgorithm`.

        The training accuracy is determined by calculating the percentage of how many of the training ranks
        the binary classification model is able to predict correctly.

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
        X_train_diff, y_train_pref, real_objects = self.pairwise_transform_from_ranks(train_objects, train_ranks,
                                                                                      use_feats)
        train_pred = self._r_svm.predict(X_train_diff)
        train_acc = accuracy_score(y_train_pref, train_pred) * 100

        if progress_window is not None:
            progress_window.put("Training accuracy: " + str(train_acc) + "%")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankSVM training accuracy execution...")
            return

        return train_acc

    # def predict_m(self, input_objects, progress_window=None, exec_stopper=None):
    #     """Predict the output of a given set of input samples by running them through the learned RankSVM model.
    #
    #     :param input_objects: array of shape [n_samples, n_feats] containing the input data corresponding
    #         to a set of (test) objects.
    #     :type input_objects: `numpy.ndarray`
    #     :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a progress
    #         log and progress bar during the experiment execution (default None).
    #     :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
    #     :param exec_stopper: an abort flag object used to abort the execution before completion
    #         (default None).
    #     :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
    #     :return:
    #         * a list containing the average predicted output resulting from running the learned model using the
    #           given input objects -- if execution is completed successfully.
    #         * None -- if aborted before completion by `exec_stopper`.
    #     :rtype: list of float (size 1)
    #     """
    #     sv_idx = self._r_svm.support_  # indices of the support vectors
    #     sv_coefs = self._r_svm.dual_coef_  # coefficients of the support vectors in the decision function
    #
    #     # matrix-based calculation
    #     Alphas_i = sv_coefs[0, :]  # shape = [1, n_sv]
    #     # only where alpha_i != 0 !!!
    #     non_zeros = Alphas_i != 0  # these are the indexes of the rows in Alphas_i where the value is !=0
    #     sv_idx = sv_idx[non_zeros]
    #     Alphas_i = Alphas_i[non_zeros]  # same as Alphas_i[Alphas_i != 0]
    #     # reshape Alphas_i from (n_sv, ) to (1, n_sv) so that it works properly with np.matmul() later
    #     Alphas_i = Alphas_i.reshape(1, -1)
    #
    #     # X_i = self._real_ranks.iloc[sv_idx].values  # shape = [n_sv, 2]
    #
    #     prefs_ids = [self._real_ranks.iloc[i // 2, 0] for i in sv_idx if i % 2 == 0]
    #     nons_ids = [self._real_ranks.iloc[i // 2, 1] for i in sv_idx if i % 2 == 1]
    #     # i//2 is integer division (without remainder) meaning
    #     # 0 & 1 will give (rank) 0; 2 & 3 will give (rank) 1; 4 & 5 will give (rank) 2; etc.
    #     # i%2 is the remainder of the division meaning
    #     # 0, 2, 4... will give (rank column index) 0 i.e. PREFS; 1, 3, 5... will give (rank column index) 1 i.e. NONS
    #     # this will result in the actual object ID corresponding to each support vector
    #
    #     X_i_prefs = self._real_objects.loc[prefs_ids].values
    #     X_i_nons = self._real_objects.loc[nons_ids].values
    #
    #     req_utility = np.matmul(Alphas_i, np.transpose(self._predict_kernel_subtraction_m2(self._real_objects,
    #                                                                                        input_objects,
    #                                                                                        X_i_prefs,
    #                                                                                        X_i_nons)))
    #     # ^ TODO: we have a problem... ValueError: shapes (1,288) and (144,1) not aligned: 288 (dim 1) != 144 (dim 0)
    #     # obviously because len(alphas)==len(ranks*2) whereas the rest==len(ranks)
    #
    #     # print("req_utility.shape = " + str(req_utility.shape))
    #     # final result = [1, n_samples]
    #
    #     # debug
    #     # print("number alphas EXCLUDING zeros: " + str(len(Alphas_i)))
    #     # print("number svs EXCLUDING alpha=zeros: " + str(len(sv_idx)))
    #     # print("reshpaed Alphas_i.shape = " + str(Alphas_i.shape))
    #     # print("X_i.shape = " + str(X_i.shape))
    #     # print("req_utility.shape = " + str(req_utility.shape))
    #     # print(req_utility)
    #     # print("summed req_utility.shape = " + str(req_utility.shape))
    #
    #     # check if experiment was aborted
    #     if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
    #         # abort execution!
    #         print("Aborting RankSVM execution...")
    #         return
    #     # otherwise send hack-y request to progress_window to update gui
    #     elif progress_window is not None:
    #         progress_window.update_gui()
    #
    #     return [req_utility]  # wrap around with a list in order for calc_train_accuracy() in base class to work

    # def _predict_kernel_subtraction_m2(self, train_objects, input_object, X_i_prefs, X_i_nons):
    #     """Compute the subtraction part of deriving the order of the given object with respect to the given support vectors.
    #
    #     :param train_objects: the objects used to train the SVM.
    #     :type train_objects: `pandas.DataFrame`
    #     :param input_object: array of shape [n_samples, n_feats] containing the feature vectors of the
    #         set of (test) objects to be predicted.
    #     :type input_object: `numpy.ndarray`
    #     :param X_i: array of shape [n_support_vectors, 2] containing RankSVM model's support vectors (each in the
    #         form of a pair of object IDs).
    #     :type X_i: `numpy.ndarray`
    #     :return: matrix of shape [n_samples, n_support_vectors] containing the float results of the kernel
    #         subtraction for each support vector for each sample in the given (test) set of objects.
    #     :rtype: `numpy.ndarray`
    #     """
    #     # get the feature vectors (lists) of the actual objects from each rank (i.e. pair of object ids)
    #     # pref & non feature vectors for rank x_i:
    #     # X_i_prefs = train_objects.loc[X_i[:, 0]].values
    #     # X_i_nons = train_objects.loc[X_i[:, 1]].values
    #
    #     # print("X_i_prefs.shape = " + str(X_i_prefs.shape))
    #     # print(X_i_prefs)
    #     # print("input_object.shape = " + str(input_object.shape))
    #
    #     # returns shape = [n_rows * n_rows] aka [n_sv, n_sv] usually
    #     answer = np.subtract(self._kernel_base_m(input_object, X_i_prefs),
    #                        self._kernel_base_m(input_object, X_i_nons))
    #     # print("answer.shape = " + str(answer.shape))
    #     return answer

    # def _kernel_base_m(self, A, B, reshape_a=False, reshape_b=False):
    #     """Compute the kernel function (:meth:`self._kernel()`) on each corresponding pair of objects in A and B.
    #
    #     Internally uses the corresponding kernel functions in `sklearn.metrics.pairwise`.
    #
    #     :param A: matrix of shape [n_samples_A, n_features] containing the feature vectors of objects in A
    #     :type A: `numpy.ndarray`
    #     :param B: matrix of shape [n_samples_B, n_features] containing the feature vectors of objects in B
    #     :type B: `numpy.ndarray`
    #     :param reshape_a: specifies whether to reshape input A into an array of shape (1, -1) indicating
    #         a single sample (default: False).
    #     :type reshape_a: bool, optional
    #     :param reshape_b: specifies whether to reshape input B into an array of shape (1, -1) indicating
    #         a single sample (default: False).
    #     :type reshape_b: bool, optional
    #     :return: a matrix of shape [n_samples_A, n_samples_B] containing the float output of the kernel function
    #         for each AxB object pair.
    #     :rtype: `numpy.ndarray`
    #     """
    #     # sklearn tells us to "Reshape your data either using array.reshape(-1, 1) if your data has a single feature
    #     # or array.reshape(1, -1) if it contains a single sample."
    #     # when predicting A=input_object, our case is the latter (single sample).
    #     # print("A.shape = " + str(A.shape))
    #     # print("B.shape = " + str(B.shape))
    #     if reshape_a:
    #         A = A.reshape(1, -1)
    #         # print("A.shape after reshape = " + str(A.shape))
    #     if reshape_b:
    #         B = B.reshape(1, -1)
    #         # print("B.shape after reshape = " + str(B.shape))
    #
    #     # convert gamma 'auto' into None for rbf_kernel
    #     gamma = self._gamma
    #     if gamma == 'auto':
    #         gamma = None
    #
    #     # calculate the kernel
    #     if self._kernel == KernelType.LINEAR.name:
    #         return linear_kernel(A, B)
    #     elif self._kernel == KernelType.POLY.name:
    #         return polynomial_kernel(A, B, degree=self._degree, gamma=gamma, coef0=0)
    #         # TODO: make coef0 a user-defined parameter!!!
    #     else:  # i.e. KernelType.RBF.name
    #         return rbf_kernel(A, B, gamma=gamma)
    #     # TODO: do the same for any other kernels! (e.g. sigmoid)
