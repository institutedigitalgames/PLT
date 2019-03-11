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

"""This module contains several backend classes and functions relating to the setting up a single experiment."""

import os
import pandas as pd
import numpy as np
# import scipy.stats as st
import sklearn.preprocessing as skpp
import datetime

from pyplt import ROOT_PATH
from pyplt.evaluation.cross_validation import KFoldCrossValidation, PreprocessedFolds
from pyplt.evaluation.holdout import HoldOut
from pyplt.exceptions import ObjectsFirstException, RanksFormatException, IDsException, NormalizationValueError, \
    ObjectIDsFormatException, NonNumericFeatureException, NoFeaturesError, NoRanksDerivedError, \
    InvalidParameterValueException, IncompatibleFoldIndicesException, AutoencoderNormalizationValueError
from pyplt.gui.util import text
from pyplt.util.enums import FileType, NormalizationType, EvaluatorType


class Experiment:
    """Class encapsulating the set-up details of a single experiment."""

    # Attributes

    # Data (features already pre-processed/normalized)
    _objects = None
    _objects_have_id = False
    _ranks = None
    _ranks_have_id = False
    _data = None
    _is_single = False
    _samples_have_id = False

    # Rank Derivation (for single file format only)
    _is_dual_format = None
    _mdm = 0.0
    _memory = 'all'

    # Features & Pre-processing (normalization)
    _features = None
    _norm_settings = dict()
    _shuffle = None
    _random_seed = None
    _autoencoder = None
    _autoencoder_loss = None
    _autoencoder_details = None

    # Feature Selection
    _fs_method = None
    _fs_algo = None
    _fs_eval = None

    # Preference Learning
    _pl_algo = None
    _pl_eval = None

    # Evaluation
    _fold_indices_based_on = None

    # Experiment Log
    _start_time = None
    _end_time = None
    _obj_path = None
    _ranks_path = None
    _single_path = None
    _orig_feats = None
    _eval_metrics = None
    _fold_metrics = None

    # Methods

    # Methods for API use
    def load_object_data(self, file_path, has_fnames=False, has_ids=False, separator=',', col_names=None,
                         na_filter=True):
        """Attempt to load an objects file as specified by the user and carries out validation checks.

        If the data fails a validation check, a :exc:`PLTException` is raised.
        
        :param file_path: the path of the file to be loaded.
        :type file_path: str
        :param has_fnames: specifies whether the file already contains feature names
            in the first row (default False).
        :type has_fnames: bool, optional
        :param has_ids: specifies whether the file already contains object IDs
            in the first column (default False).
        :type has_ids: bool, optional
        :param separator: the character separating items in the CSV file (default ',').
        :type separator: str, optional
        :param col_names: specifies the column names to be used (default None).
        :type col_names: list of str or None, optional
        :param na_filter: specifies whether to detect missing value markers (default True).
        :type na_filter: bool, optional
        :raises ObjectIDsFormatException: if one or more non-numeric object IDs are detected.
        :raises NonNumericFeatureException: if the one or more feature with one or more
            non-numeric values are detected.
        """
        objects = _load_data(FileType.OBJECTS, file_path, has_ids=has_ids, has_fnames=has_fnames,
                             separator=separator, col_names=col_names, na_filter=na_filter)
        features = list(objects.columns)

        # print(objects)

        # Validation: Check if objects file contains numeric-only IDs
        if has_ids:  # otherwise PLT automatically makes them all numeric
            try:
                # try to convert to integers...
                test = objects.iloc[:, 0].values.astype(int)
            except ValueError:
                raise ObjectIDsFormatException  # automatically skips the rest of the method...

        # Validation: Check for non-numeric features (& convert them to binary)
        try:
            can_be_float = objects.values.astype(float)
        except (ValueError, TypeError):
            raise NonNumericFeatureException
        # ^ TODO: make binary (binarization?) as in Java PLT instead of raising exception

        # only when all validation checks are passed, set/reset actual variables!
        # Delete any old ranks (because they need to be re-validated w.r.t. the new objects...)
        self._ranks = None
        self._ranks_path = None
        self._ranks_have_id = False

        self._is_single = False
        self._obj_path = file_path
        self._single_path = None
        self._data = None
        self._objects = objects
        self._features = features
        self._objects_have_id = True
        # (re) init norm_settings
        self._norm_settings = dict.fromkeys(np.arange(len(self._features)-1).tolist(), NormalizationType.NONE.name)
        # ^ -1 to account for ID column

    def load_rank_data(self, file_path, has_fnames=False, has_ids=False, separator=',', col_names=None,
                       na_filter=True):
        """Attempt to load a ranks file as specified by the user and carries out validation checks.

        If the data fails a validation check, a :exc:`PLTException` is raised.

        :param file_path: the path of the file to be loaded.
        :type file_path: str
        :param has_fnames: specifies whether the file already contains feature names
            in the first row (default False).
        :type has_fnames: bool, optional
        :param has_ids: specifies whether the file already contains object IDs
            in the first column (default False).
        :type has_ids: bool, optional
        :param separator: the character separating items in the CSV file (default ',').
        :type separator: str, optional
        :param col_names: specifies the column names to be used (default None).
        :type col_names: list of str or None, optional
        :param na_filter: a boolean indicating whether to detect missing value markers (default True).
        :type na_filter: bool, optional
        :raises ObjectsFirstException: if the objects have not been loaded first.
        :raises RanksFormatException: if the dataset contains an unexpected amount of columns.
        :raises IDsException: if the dataset contains entries that do not refer to any object ID in the objects dataset.
        """
        ranks = _load_data(FileType.RANKS, file_path, has_ids=has_ids, has_fnames=has_fnames,
                           separator=separator, col_names=col_names, na_filter=na_filter)

        # Validation: Check if the objects (not single) have been loaded first
        if self._is_single or self._objects is None:
            # Old objects from single file do not count
            # (but no need to delete them since this 1st validation step will never allow it anyway)
            raise ObjectsFirstException

        # Validation: Check if ranks file contains 3 columns: (ID), Preferred_ID, Other_ID
        if len(ranks.columns) != 3:
            raise RanksFormatException

        # Validation: Check if all entries in ranks file refer to some ID (col 0) in objects.
        ids = list(self._objects.iloc[:, 0])
        arein = ranks.isin(ids).values[:, 1:]
        invalid = not (np.sum(arein, axis=(0, 1)) == arein.size)
        if invalid:
            raise IDsException

        # only when all validation checks are passed, set/reset actual variables!
        self._is_single = False
        self._ranks_path = file_path
        self._single_path = None

        self._ranks = ranks
        self._ranks_have_id = True

    def load_single_data(self, file_path, has_fnames=False, has_ids=False, separator=',', col_names=None,
                         na_filter=True, mdm=0.0, memory='all'):
        """Attempt to load a single file as specified by the user and carries out validation checks.

        When the experiment is run, pairwise preferences are automatically derived based on the ratings (last column
        in the given dataset) of the given objects/samples. The dataset is thus split into objects and ranks. The
        derivation of the pairwise preferences/ranks may be controlled via the optional minimum distance
        margin (mdm) and memory arguments of this method.
        If the data fails a validation check, a :exc:`PLTException` is raised.

        :param file_path: the path of the file to be loaded.
        :type file_path: str
        :param has_fnames: specifies whether the file already contains feature names
            in the first row (default False).
        :type has_fnames: bool, optional
        :param has_ids: specifies whether the file already contains object IDs
            in the first column (default False).
        :type has_ids: bool, optional
        :param separator: the character separating items in the CSV file (default ',').
        :type separator: str, optional
        :param col_names: specifies the column names to be used (default None).
        :type col_names: list of str or None, optional
        :param na_filter: a boolean indicating whether to detect missing value markers (default True).
        :type na_filter: bool, optional
        :param mdm: the minimum distance margin i.e., the minimum difference between the ratings of a given pair of
            objects/samples that is required for the object pair to be considered a valid and clear preference
            (default 0.0).
        :type mdm: float, optional
        :param memory: specifies how many neighbouring objects/samples are to be compared with a given object/sample
            when constructing the pairwise ranks (default 'all'). If 'all', all objects/samples are compared to each
            other.
        :type memory: int or 'all', optional
        :raises ObjectIDsFormatException: if one or more non-numeric object IDs are detected.
        :raises NonNumericFeatureException: if the one or more feature with one or more
            non-numeric values are detected.
        :raises NoRanksDerivedError: if no pairwise preferences could be derived from the given data. This is either
            because there are no clear pairwise preferences in the data or because none of the clear pairwise
            preferences in the data conform to the chosen values for the rank derivation parameters (i.e., the minimum
            distance margin (`mdm`) and the memory (`memory`) parameters).
        :raises InvalidParameterValueException: if the user attempts to use a negative value (i.e., smaller than 0.0)
            for the `mdm` parameter.
        """
        single_data = _load_data(FileType.SINGLE, file_path, has_ids=has_ids, has_fnames=has_fnames,
                                 separator=separator, col_names=col_names, na_filter=na_filter)
        features = list(single_data.columns[:-1])  # all columns except last (rating)

        # print(objects)

        # Validation: Check if objects file contains numeric-only IDs
        if has_ids:  # otherwise PLT automatically makes them all numeric
            try:
                # try to convert to integers...
                test = single_data.iloc[:, 0].values.astype(int)
            except ValueError:
                raise ObjectIDsFormatException  # automatically skips the rest of the method...

        # Validation: Check for non-numeric features (& convert them to binary)
        try:
            can_be_float = single_data.values.astype(float)
        except (ValueError, TypeError):
            raise NonNumericFeatureException
        # ^ TODO: make binary (binarization?) as in Java PLT instead of raising exception

        # only when all validation checks are passed, set/reset actual variables!
        self._obj_path = None
        self._ranks_path = None
        self._single_path = file_path
        self._objects = None
        self._ranks = None
        self._data = single_data
        self.set_rank_derivation_params(mdm, memory)
        self._is_single = True
        self._features = features
        self._samples_have_id = True
        # (re) init norm_settings
        self._norm_settings = dict.fromkeys(np.arange(len(self._features)-1).tolist(), NormalizationType.NONE.name)
        # ^ -1 to account for ID column

    # Preprocessing

    def set_normalization(self, feature_ids, norm_method):
        """Set the normalization method to be used for the given feature or features.

        The actual application of the normalization method to the features occurs when the experiment is run.

        N.B. If the dataset includes an object ID column as its first column, it is ignored by this method. Therefore,
        in such a case, an argument of 0 passed to the parameter `feature_ids` is taken to refer to the first feature
        in the dataset (the second column in the dataset) and not the object ID column.

        :param feature_ids: the index of the feature or the list of features (columns in the dataset) for which
            the normalization method is to be set.
        :type feature_ids: int or list of ints
        :param norm_method: the normalization method to be used.
        :type norm_method: :class:`pyplt.util.enums.NormalizationType`
        """
        # no need to +1 to account for 'ID' column bc self._norm_settings is only used after ID column is removed
        if isinstance(feature_ids, int):  # if just one feature id
            self._norm_settings[feature_ids] = norm_method.name
        else:  # i.e. list of feature ids
            for feat in feature_ids:
                self._norm_settings[feat] = norm_method.name

    def _set_norm_settings(self, norm_settings):
        """Set the normalization methods to be used for each of the features in the dataset.

        The actual application of the normalization methods to the features occurs when the experiment is run.
        :param norm_settings: a dict with the indices of the features as the dict's keys and
            names of enumerated constants of type :class:`pyplt.util.enums.NormalizationType` (indicating how the
            corresponding feature is to be normalized) as the dict's values.
        :type norm_settings: dict of str (names of :class:`pyplt.util.enums.NormalizationType`)
        """
        self._norm_settings = norm_settings

    # Feature Selection

    def set_fs_method(self, fs_method):
        """Set the feature selection method of the experiment to the given method.

        :param fs_method: the given feature selection method.
        :type fs_method: :class:`pyplt.fsmethods.base.FeatureSelectionMethod`
        """
        self._fs_method = fs_method

    def set_fs_algorithm(self, fs_algorithm):
        """Set the preference learning algorithm used in the feature selection phase of the experiment to the given algorithm.

        :param fs_algorithm: the given preference learning algorithm.
        :type fs_algorithm: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        """
        self._fs_algo = fs_algorithm

    def set_fs_evaluator(self, fs_evaluator):
        """Set the evaluation method used in the feature selection phase of the experiment to the given method.

        :param fs_evaluator: the given evaluation method.
        :type fs_evaluator: :class:`pyplt.evaluation.base.Evaluator`
        """
        self._fs_eval = fs_evaluator

    def set_pl_algorithm(self, pl_algorithm):
        """Set the preference learning algorithm of the experiment to the given algorithm.

        :param pl_algorithm: the given preference learning algorithm.
        :type pl_algorithm: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        """
        self._pl_algo = pl_algorithm

    def set_pl_evaluator(self, pl_evaluator):
        """Set the evaluation method of the experiment to the given method.

        :param pl_evaluator: the given evaluation method.
        :type pl_evaluator: :class:`pyplt.evaluation.base.Evaluator`
        """
        self._pl_eval = pl_evaluator

    def set_rank_derivation_params(self, mdm=None, memory=None):
        """Set the values of the parameters used during the derivation of ranks from ratings.

        These only apply if a single file format is used.

        :param mdm: the minimum distance margin i.e., the minimum difference between the ratings of a given pair of
            objects/samples that is required for the object pair to be considered a valid and clear preference
            (default None). If None, a value of 0.0 is used by default during rank derivation.
        :type mdm: float or None, optional
        :param memory: specifies how many neighbouring objects/samples are to be compared with a given object/sample
            when constructing the pairwise ranks (default None). If None, a value of 'all' (i.e., all objects/samples
            are compared to each other) is used by default during rank derivation.
        :type memory: int or 'all' or None, optional
        """
        if mdm is not None:
            self._mdm = mdm
        if memory is not None:
            self._memory = memory

    def set_autoencoder(self, autoencoder):
        """Set the autoencoder algorithm to be used to extract features from the dataset in the experiment.

        :param autoencoder: the given autoencoder algorithm.
        :type autoencoder: `pyplt.autoencoder.Autoencoder` or None
        """
        self._autoencoder = autoencoder

    def run(self, shuffle=False, random_state=None, debug=False, progress_window=None, exec_stopper=None):
        """Run the the experiment: feature selection first (if applicable), then preference learning.

        Prior to running feature selection and preference learning, this method applies all specified pre-processing 
        steps (e.g., fold-splitting, rank derivation, normalization) to the loaded data (if applicable). The 
        method also stores the experiment details and returns the results for further use.

        :param shuffle: specifies whether or not to shuffle the data (samples in the case of the single file format;
            ranks in the case of the dual file format) at the start of executing the experiment; i.e., prior to fold
            splitting, rank derivation, and normalization (if applicable) (default False).
        :type shuffle: bool, optional
        :param random_state: seed for the random number generator (if int), or numpy RandomState object, used to
            shuffle the dataset if `shuffle` is True (default None).
        :type random_state: int or `numpy.random.RandomState`, optional
        :param debug: specifies whether or not to print notes to console for debugging purposes (default False).
        :type debug: bool, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the experiment results -- if experiment is completed successfully.
                
              * eval_metrics -- the resulting average train and, if applicable, average test accuracies
              * fold_metrics -- the fold-specific start timestamp, end timestamp, evaluation metrics, and a 
                `pandas.DataFrame` representation of the trained model.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: 
            * eval_metrics -- dict with keys:

              * '`Training Accuracy`'
              * '`Test Accuracy`' (if applicable)
            * fold_metrics -- list of tuple, each containing:
            
              * start_time -- `datetime` timestamp (UTC timezone)
              * end_time -- `datetime` timestamp (UTC timezone)
              * eval_metrics -- dict with keys:
              
                * '`Training Accuracy`'
                * '`Test Accuracy`' (if applicable)
              * model -- `pandas.DataFrame`
        :raises NoFeaturesError: if there are no features/attributes in the objects data.
        :raises NoRanksDerivedError: if rank derivation fails because no pairwise preferences could be derived 
            from the given data. This is either
            because there are no clear pairwise preferences in the data or because none of the clear pairwise
            preferences in the data conform to the chosen values for the rank derivation parameters (i.e., the minimum
            distance margin (`mdm`) and the memory (`memory`) parameters).
        :raises InvalidParameterValueException: if the user attempted to use a negative value (i.e., smaller than 0.0)
            for the `mdm` rank derivation parameter.
        :raises NormalizationValueError: if normalization fails because one of the given values cannot be converted
            to int or float prior to the normalization.
        :raises IncompatibleFoldIndicesException: if the amount of user-specified fold indices for cross validation
            does not match the amount of samples in the dataset.
        :raises AutoencoderNormalizationValueError: if normalization prior to feature extraction via autoencoder
            fails due to the presence of non-numeric values in the dataset.
        """
        self._shuffle = shuffle
        self._random_seed = random_state

        # Step i. Determine if data is in dual file format or not (single file format)
        if self._data is not None:
            self._is_dual_format = False
        else:
            self._is_dual_format = True

        # Step A. Convert '(_)*ID' column in dataset/s to its index (if applicable)
        if self._objects_have_id:
            objects_copy = self._objects.copy(deep=True)
            self._objects = objects_copy.set_index(objects_copy.columns[0])
        if self._ranks_have_id:
            ranks_copy = self._ranks.copy(deep=True)
            self._ranks = ranks_copy.set_index(ranks_copy.columns[0])
        if self._samples_have_id:
            data_copy = self._data.copy(deep=True)
            self._data = data_copy.set_index(data_copy.columns[0])

        # Step B1. Get features
        if self._autoencoder is not None:
            self._features = ["ExtractedFeature" + str(f+1) for f in range(self._autoencoder.get_code_size())]
        else:
            if self._data is not None:
                self._features = list(self._data.columns[:-1])  # all except last column (ratings)!
            else:
                self._features = list(self._objects.columns)

        self._orig_feats = self._features.copy()

        # Step B2. Validation: Must have at least one feature included!!

        if len(self._features) < 1:
            raise NoFeaturesError()

        # Step C. Print experiment details

        if debug:
            print("Running Experiment...")
            print("---------------- DATA ----------------")
            if self._is_dual_format:
                # print("Objects:")
                # print(self._objects)
                # print("Ranks:")
                # print(self._ranks)
                print("Details:")
                print(str(len(self._objects)) + " objects.")
                print(str(len(self._ranks)) + " ranks.")
            else:
                # print("Data:")
                # print(self._data)
                print("Details:")
                print(str(len(self._data)) + " samples.")
            print("Original features: ")
            print(self._features)
            if self._fs_method is not None:
                print("---------------- FS ----------------")
                # fs
                print("FS Method: " + str(self._fs_method))
                print("FS Params:")
                print(self._fs_method.get_params_string())
                print("FS algorithm: " + str(self._fs_algo))
                if self._fs_algo is not None:
                    print("FS algorithm params:")
                    print(self._fs_algo.get_params_string())
                print("FS evaluator: " + str(self._fs_eval))
                if self._fs_eval is not None:
                    print("FS evaluator params:")
                    print(self._fs_eval.get_params_string())
            print("---------------- PL ----------------")
            # pl
            print("PL Algorithm: " + str(self._pl_algo))
            print("PL Algorithm params:")
            print(self._pl_algo.get_params_string())

        self._start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        print("STARTED AT " + str(self._start_time))

        # Step D. Prepare dataset

        # D0a. Optionally, shuffle the data (before splitting the folds!)
        # in the case of single file format, shuffle the samples
        # in the case of dual file format, shuffle the ranks only
        # also reset index (N.B. this replaces ID column which had already been made the new index)!
        if shuffle:
            print("before shuffle")
            print(self._data)
            print(self._ranks)

            if self._is_dual_format:  # dual file format
                # shuffle the ranks
                ranks_copy = self._ranks.copy(deep=True)
                self._ranks = ranks_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)
            else:  # single file format
                # shuffle the samples
                data_copy = self._data.copy(deep=True)
                self._data = data_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)

            print("after shuffle")
            print(self._data)
            print(self._ranks)

        # E0b. data compression / feature extraction via autoencoder
        if self._autoencoder is not None:
            if self._is_dual_format:  # dual file format
                samples = self._objects.copy(deep=True).values
            else:  # single file format
                samples = self._data.iloc[:, :-1].copy(deep=True).values  # all columns but last

            # first, make sure data is normalized to the range [0..1]
            try:
                scaler = skpp.MinMaxScaler(feature_range=(0, 1), copy=True)
                samples = scaler.fit_transform(samples)
            except (ValueError, TypeError):
                # value=str(str(ve.args[0]).split("\'")[1])
                raise AutoencoderNormalizationValueError(norm_method=NormalizationType.MIN_MAX)

            # train encoder
            self._autoencoder_loss = self._autoencoder.train(samples, progress_window=progress_window,
                                                             exec_stopper=exec_stopper)
            if self._autoencoder_loss is None:  # check if execution was aborted!
                print("Aborting experiment execution...")
                # make sure to do final clean up (close tf.Session) before aborting!
                self._autoencoder.clean_up()
                return

            # encode samples
            encoded_samples = np.array(self._autoencoder.encode(samples, progress_window=progress_window,
                                                                exec_stopper=exec_stopper))
            if encoded_samples is None:  # check if execution was aborted!
                print("Aborting experiment execution...")
                # make sure to do final clean up (close tf.Session) before aborting!
                self._autoencoder.clean_up()
                return

            print("encoded_samples")
            print(encoded_samples)
            n_new_feats = encoded_samples.shape[1]  # n_cols
            new_cols = ["ExtractedFeature" + str(n+1) for n in range(n_new_feats)]
            print("new features:")
            print(n_new_feats)

            if self._is_dual_format:  # dual file format
                old_index = self._objects.index
                # print("old_index")
                # print(old_index)
                self._objects = pd.DataFrame(encoded_samples, index=old_index, columns=new_cols)
                self._features = list(self._objects.columns)
            else:  # single file format
                old_index = self._data.index
                ratings_ = self._data.iloc[:, -1]
                ratings_col_label = self._data.columns[-1]
                self._data = pd.DataFrame(encoded_samples, index=old_index, columns=new_cols)
                self._data.insert(n_new_feats, ratings_col_label, ratings_, allow_duplicates=True)
                self._features = list(self._data.columns[:-1])  # all except last column (ratings)!

            # do final clean up (close tf.Session)
            self._autoencoder.clean_up()

            # store autoencoder details
            self._autoencoder_details = {"Autoencoder-Topology": str(self._autoencoder.get_topology_incl_input()),
                                         "Autoencoder-Code Size": str(self._autoencoder.get_code_size()),
                                         "Autoencoder-Learning Rate": str(self._autoencoder.get_learn_rate()),
                                         "Autoencoder-Error Threshold": str(self._autoencoder.get_error_thresh()),
                                         "Autoencoder-Epochs": str(self._autoencoder.get_epochs())}

        # D0c. group everything up into single data variable
        if not self._is_dual_format:  # single file format
            data = self._data
        else:  # dual file format
            data = self._objects, self._ranks

        # D1. Split CV folds and start looping through each! (or use threads!)

        # first make sure that manual folds (if applicable) match dataset in size
        # remember, if the single file format is used, the indices should correspond to objects/samples
        # otherwise (if the dual file format is used), the indices should correspond to ranks
        for eval_ in [self._fs_eval, self._pl_eval]:  # do for both fs folds and pl folds!
            if isinstance(eval_, KFoldCrossValidation):
                if eval_._test_folds is not None:
                    if not self._is_dual_format:  # single file format
                        if len(eval_._test_folds) != len(self._data):
                            raise IncompatibleFoldIndicesException
                    else:  # dual file format
                        if len(eval_._test_folds) != len(self._ranks):
                            raise IncompatibleFoldIndicesException

        # D1a. Split folds for FS
        fs_folds_ready = False
        # pl_folds_ready = False
        if isinstance(self._fs_eval, KFoldCrossValidation) or isinstance(self._fs_eval, HoldOut):  # create folds for FS
            fs_folds = self._fs_eval.split(data)
            if isinstance(self._fs_eval, HoldOut):
                fs_folds = [fs_folds]  # place single tuple in list for consistency
            else:  # kfcv
                fs_folds_ready = True
        else:  # i.e. NOT kfcv or holdout i.e. no splitting occurs i.e. training only
            if self._data is not None:
                fs_folds = [(np.arange(self._data.shape[0]), None)]  # train set includes all sample indices, no test set
                # ^ not self._data.index.values because that might require loc not iloc!
            else:
                fs_folds = [(np.arange(self._ranks.shape[0]), None)]  # train set includes all rank indices, no test set
                # ^ not self._ranks.index.values because that might require loc not iloc!

        # D1b. split folds for PL
        if isinstance(self._pl_eval, KFoldCrossValidation) or isinstance(self._pl_eval, HoldOut):
            if fs_folds_ready:  # copy folds from FS
                pl_folds = fs_folds
            else:  # create folds for PL only
                pl_folds = self._pl_eval.split(data)
                if isinstance(self._pl_eval, HoldOut):
                    pl_folds = [pl_folds]  # place single tuple in list for consistency
            # pl_folds_ready = True
        else:  # i.e. NOT kfcv or holdout i.e. no splitting occurs i.e. training only
            if self._data is not None:
                pl_folds = [(np.arange(self._data.shape[0]), None)]  # train set includes all sample indices, no test set
                # ^ not self._data.index.values because that might require loc not iloc!
            else:
                pl_folds = [(np.arange(self._ranks.shape[0]), None)]  # train set includes all rank indices, no test set
                # ^ not self._ranks.index.values because that might require loc not iloc!

        # debug
        f = 0
        print("fs_folds: ")
        for fold in fs_folds:
            # print(fold)
            fold_train = fold[0]
            fold_test = fold[1]
            print("********* fold " + str(f) + " / train (" + str(len(fold_train)) + "): ")
            print(fold_train)
            if fold_test is not None:
                print("+++++++++ fold " + str(f) + " / test: (" + str(len(fold_test)) + ")")
            else:
                print("+++++++++ fold " + str(f) + " / test: (None)")
            print(fold_test)
            f = f + 1
        f = 0
        print("pl_folds: ")
        for fold in pl_folds:
            # print(fold)
            fold_train = fold[0]
            fold_test = fold[1]
            print("********* fold " + str(f) + " / train: (" + str(len(fold_train)) + ")")
            print(fold_train)
            if fold_test is not None:
                print("+++++++++ fold " + str(f) + " / test: (" + str(len(fold_test)) + ")")
            else:
                print("+++++++++ fold " + str(f) + " / test: (None)")
            print(fold_test)
            f = f + 1

        # ^ confirmed to work for ranks-auto, ratings-auto

        # create dictionary where keys are the fold-combo number and values are a tuple of two pd.DataFrames
        # containing the training set pd.DataFrame and the test set pd.DataFrame of that fold-combo
        # fs_folds_data = {f: (pd.DataFrame(), pd.DataFrame()) for f in range(fs_folds)}
        #
        # for train, test in fs_folds:

        # get the actual data for every fold
        fs_folds_data = PreprocessedFolds([self._apply_pre_processing(train, test) for train, test in fs_folds])
        if fs_folds_ready:
            pl_folds_data = fs_folds_data
        else:
            pl_folds_data = PreprocessedFolds([self._apply_pre_processing(train, test) for train, test in pl_folds])

        # Step 1: Feature Selection

        if self._fs_method is not None:
            self._features = self._fs_method.select(None, None, self._fs_algo, preprocessed_folds=fs_folds_data,
                                                    progress_window=progress_window,
                                                    exec_stopper=exec_stopper)
            self._fs_algo.clean_up()  # do any final clean ups required by the fs algorithm class

            if self._features is None:  # check if execution was aborted!
                print("Aborting experiment execution...")
                return

            if debug:
                print("***COMPLETE*** Features selected: ")
                print(self._features)

        # Step 2: Preference Learning

        self._fold_metrics = []
        self._eval_metrics = dict()
        avg_train_acc = 0.0
        avg_test_acc = None
        f = 0
        if self._pl_eval is None:  # train only
            for train_objects, train_ranks, test_objects, test_ranks in pl_folds_data.next_fold():
                start_time = datetime.datetime.now(tz=datetime.timezone.utc)
                print("Fold " + str(f) + " started at " + str(start_time))

                eval_metrics = dict()

                print("train_objects: ")
                print(train_objects.loc[:, self._features])
                print("train_ranks: ")
                print(train_ranks)

                # try calling pl_algo.init_train() in case the model setup (e.g. weights) needs to be reset per fold
                self._pl_algo.init_train(len(self._features))

                self._pl_algo.train(train_objects, train_ranks, use_feats=self._features,
                                    progress_window=progress_window,
                                    exec_stopper=exec_stopper)
                # algo_copy = copy.deepcopy(self._pl_algo)  # make a deep copy of the trained algorithm for model saving
                model = self._pl_algo.save_model(timestamp=0.0, suppress=True)  # store the trained model for saving
                train_acc = self._pl_algo.calc_train_accuracy(train_objects, train_ranks, use_feats=self._features,
                                                              progress_window=progress_window,
                                                              exec_stopper=exec_stopper)
                # finally, do any final clean ups required by the pl algorithm class
                self._pl_algo.clean_up()  # TODO: apply this before if aborted on train_acc!!!
                if train_acc is None:  # check if execution was aborted!
                    print("Aborting experiment execution...")
                    return
                eval_metrics['Training Accuracy'] = train_acc
                avg_test_acc = None
                avg_train_acc = avg_train_acc + train_acc

                end_time = datetime.datetime.now(tz=datetime.timezone.utc)
                # ^ any final clean ups required by the pl algorithm class are carried out at the
                # closing of the results window (to enable saving of model before session closes).

                if debug:
                    print("***Fold " + str(f) + " complete*** Training Accuracy: " + str(train_acc) +
                          ";  no validation.")

                f = f + 1
                self._fold_metrics.append((start_time, end_time, eval_metrics, model))
                # end of fold
            avg_train_acc = avg_train_acc / float(f)

        else:  # train and test
            if debug:
                print("PL evaluator: " + str(self._pl_eval))
                print("PL evaluator params:")
                print(self._pl_eval.get_params_string())
                print("--------------------------------------")
            avg_test_acc = 0.0
            for train_objects, train_ranks, test_objects, test_ranks in pl_folds_data.next_fold():
                start_time = datetime.datetime.now(tz=datetime.timezone.utc)
                print("Fold " + str(f) + " started at " + str(start_time))

                eval_metrics = dict()

                print("train_objects: ")
                print(train_objects.loc[:, self._features])
                print("train_ranks: ")
                print(train_ranks)
                print("test_objects: ")
                print(test_objects.loc[:, self._features])
                print("test_ranks: ")
                print(test_ranks)

                # try calling pl_algo.init_train() in case the model setup (e.g. weights) needs to be reset per fold
                self._pl_algo.init_train(len(self._features))

                # first train
                self._pl_algo.train(train_objects, train_ranks, use_feats=self._features,
                                    progress_window=progress_window,
                                    exec_stopper=exec_stopper)
                # algo_copy = copy.deepcopy(self._pl_algo)  # make a deep copy of the trained algorithm for model saving
                model = self._pl_algo.save_model(timestamp=0.0, suppress=True)  # store the trained model for saving
                train_acc = self._pl_algo.calc_train_accuracy(train_objects, train_ranks, use_feats=self._features,
                                                              progress_window=progress_window,
                                                              exec_stopper=exec_stopper)
                if train_acc is None:  # check if execution was aborted!
                    print("Aborting experiment execution...")
                    return
                eval_metrics['Training Accuracy'] = train_acc
                avg_train_acc = avg_train_acc + train_acc

                # then test
                test_acc = self._pl_algo.test(test_objects, test_ranks,
                                              use_feats=self._features,
                                              progress_window=progress_window, exec_stopper=exec_stopper)
                # finally, do any final clean ups required by the pl algorithm class
                self._pl_algo.clean_up()  # TODO: apply this before if aborted on train_acc!!!
                if (train_acc is None) or (test_acc is None):  # check if execution was aborted!
                    print("Aborting experiment execution...")
                    return
                eval_metrics['Test Accuracy'] = test_acc
                avg_test_acc = avg_test_acc + test_acc

                end_time = datetime.datetime.now(tz=datetime.timezone.utc)
                # ^ any final clean ups required by the pl algorithm class are carried out at the
                # closing of the results window (to enable saving of model before session closes).

                if debug:
                    print("***Fold " + str(f) + " complete*** Training Accuracy: " + str(train_acc) +
                          ";  Test Accuracy: " + str(test_acc))

                f = f + 1
                self._fold_metrics.append((start_time, end_time, eval_metrics, model))
                # end of fold
            avg_train_acc = avg_train_acc / float(f)
            avg_test_acc = avg_test_acc / float(f)

        self._end_time = datetime.datetime.now(tz=datetime.timezone.utc)

        if pl_folds_data.get_n_folds() > 1:  # aka if f > 1
            prefix = 'Average '
        else:
            prefix = ''
        self._eval_metrics[prefix+'Training Accuracy'] = avg_train_acc
        if avg_test_acc is not None:
            self._eval_metrics[prefix+'Test Accuracy'] = avg_test_acc

        if debug:
            if avg_test_acc is not None:
                print("***COMPLETE*** Average Training Accuracy: " + str(avg_train_acc) +
                      ";  Average Test Accuracy: " + str(avg_test_acc))
            else:
                print("***COMPLETE*** Average Training Accuracy: " + str(avg_train_acc) + ";  no validation.")

        return self._eval_metrics, self._fold_metrics

    def _apply_pre_processing(self, train, test=None):
        """Apply any pre-processing steps to the dataset or a given fold of the dataset (train-test combination).

        This includes normalization and, if a single file format was used, preference/rank derivation.

        If the single file format is used, the indices are given with respect to objects. Otherwise (if the dual
        file format is used), the indices are given with respect to the ranks.

        :param train: indices of the samples in the training set.
        :type train: `numpy.ndarray`
        :param test: indices of the samples in the test set (default None).
        :type test: `numpy.ndarray` or None, optional
        :return: the pre-processed training set and test set (if applicable) of the dataset or dataset fold.
        :rtype:
            * train_objects: `pandas.DataFrame`
            * train_ranks: `pandas.DataFrame`
            * test_objects: `pandas.DataFrame` (if applicable) or None
            * test_ranks: `pandas.DataFrame` (if applicable) or None
        :raises NoRanksDerivedError: if rank derivation fails because no pairwise preferences could be derived
            from the given data. This is either
            because there are no clear pairwise preferences in the data or because none of the clear pairwise
            preferences in the data conform to the chosen values for the rank derivation parameters (i.e., the minimum
            distance margin (`mdm`) and the memory (`memory`) parameters).
        :raises InvalidParameterValueException: if the user attempted to use a negative value (i.e., smaller than 0.0)
            for the `mdm` rank derivation parameter.
        :raises NormalizationValueError: if normalization fails because one of the given values cannot be converted
            to int or float prior to the normalization.
        """
        test_objects_labels = None
        test_objects = None
        test_ranks = None

        train_objects_indices = None
        test_objects_indices = None

        # 0. if data is ranks-based (dual file format), split the objects into train and test for this fold combo
        # train objects are all objects that occur within the train ranks
        # test objects are all objects that occur within the test ranks
        if self._is_dual_format:  # confirmed to work for ranks-auto
            train_objects_labels = np.unique(self._ranks.iloc[train].values)  # (sorted)
            train_objects = self._objects.loc[train_objects_labels].copy(deep=True)
            train_objects_indices = np.flatnonzero(np.isin(self._objects.index, train_objects_labels))
            if test is not None:
                test_objects_labels = np.unique(self._ranks.iloc[test].values)  # (sorted)
                test_objects = self._objects.loc[test_objects_labels].copy(deep=True)
                test_objects_indices = np.flatnonzero(np.isin(self._objects.index, test_objects_labels))
        # if data is ratings-based, the objects are already split into train and test!
        else:  # confirmed to work for ratings-auto
            train_objects_labels = self._data.iloc[train, :-1].index.tolist()  # self._objects.index[train]
            train_objects = self._data.iloc[train, :-1].copy(deep=True)
            if test is not None:
                test_objects_labels = self._data.iloc[test, :-1].index.tolist()  # self._objects.index[test]
                test_objects = self._data.iloc[test, :-1].copy(deep=True)

        print("train_objects_labels: ")
        print(train_objects_labels)
        print("train_objects_indices: ")
        print(train_objects_indices)

        print("test_objects_labels: ")
        print(test_objects_labels)
        print("test_objects_indices: ")
        print(test_objects_indices)

        # 1. Apply normalization # confirmed to work for ranks-auto, ratings-auto
        # first fit_transform on training set/s, then transform on test set/s (all taken care of by _normalize() method)
        # get all unique norm methods chosen from norm_settings
        print("norm_settings.values(): ")
        print(self._norm_settings.values())
        norm_methods = list(set(list(self._norm_settings.values())))  # confirmed to work!
        print("(unique) norm_methods: ")
        print(norm_methods)
        if NormalizationType.NONE.name in norm_methods:
            norm_methods.remove(NormalizationType.NONE.name)  # ignore NONE method type!
        # get list of feats for each chosen norm_method
        nm_feat_ids = [[k for k, v in self._norm_settings.items() if v == nm] for nm in norm_methods]
        # ^ confirmed to work for ranks-auto, ratings-auto
        print("nm_feat_ids: ")
        print(nm_feat_ids)
        # get subsets of data for each chosen norm_method
        n = 0
        for nm in norm_methods:
            ids = nm_feat_ids[n]
            print("normalizing features <"+str(ids)+"> with normalization method " + str(nm))
            if self._is_dual_format:  # dual file format
                print("train objects before normalization: ")
                print(train_objects.iloc[:, ids])
                # print(self._objects.iloc[train_objects_indices, ids])
                norm_train, norm_test = _normalize(self._objects, ids, NormalizationType[nm],
                                                   train=train_objects_indices,
                                                   test=test_objects_indices)
                train_objects.iloc[:, ids] = norm_train
                # self._objects.iloc[train_objects_indices, ids] = norm_train
                print("train objects after normalization: ")
                print(train_objects.iloc[:, ids])
                # print(self._objects.iloc[train_objects_indices, ids])
                if test is not None:
                    print("test objects before normalization: ")
                    print(test_objects.iloc[:, ids])
                    # print(self._objects.iloc[test_objects_indices, ids])
                    test_objects.iloc[:, ids] = norm_test
                    # self._objects.iloc[test_objects_indices, ids] = norm_test
                    print("test objects after normalization: ")
                    print(test_objects.iloc[:, ids])
                    # print(self._objects.iloc[test_objects_indices, ids])
            else:  # single file format
                print("train objects before normalization: ")
                print(train_objects.iloc[:, ids])
                # print(self._data.iloc[train, ids])
                norm_train, norm_test = _normalize(self._data, ids, NormalizationType[nm],
                                                   train=train,
                                                   test=test)
                print(norm_train)
                train_objects.iloc[:, ids] = norm_train
                # self._data.iloc[train, ids] = norm_train
                print("train objects after normalization: ")
                print(train_objects.iloc[:, ids])
                # print(self._data.iloc[train, ids])
                if test is not None:
                    print("test objects before normalization: ")
                    print(test_objects.iloc[:, ids])
                    # print(self._data.iloc[test, ids])
                    test_objects.iloc[:, ids] = norm_test
                    # self._data.iloc[test, ids] = norm_test
                    print("test objects after normalization: ")
                    print(test_objects.iloc[:, ids])
                    # print(self._data.iloc[test, ids])
            n = n + 1

        # 2. Rank derivation # confirmed to work for ranks-auto, ratings-auto
        if not self._is_dual_format:  # single file format
            _, train_ranks = _split_single(self._data.iloc[train], self._mdm, self._memory)  # TODO: use query-id?
            # ignore train_objects bc we already have it (normalized)
            if test is not None:
                _, test_ranks = _split_single(self._data.iloc[test], self._mdm, self._memory)  # TODO: use query-id?
                # ignore test_objects bc we already have it (normalized)
        else:  # dual file format
            # TODO: filter ranks by query-id?
            # train_objects = self._objects.copy(deep=True)
            # if test is not None:
            #     test_objects = self._objects.copy(deep=True)
            train_ranks = self._ranks.iloc[train].copy(deep=True)
            if test is not None:
                test_ranks = self._ranks.iloc[test].copy(deep=True)

            # convert back to DataFrames; no need to reset index (... right?)
            # train_objects = pd.DataFrame(train_objects)
            # train_ranks = pd.DataFrame(train_ranks)
            # test_objects = pd.DataFrame(test_objects)
            # test_ranks = pd.DataFrame(test_ranks)

        return train_objects, train_ranks, test_objects, test_ranks

    def is_dual_format(self):
        """Indicate whether or not the data which has been loaded so far is in the dual file format.

        :return: specifies whether the data is in the dual file format or not (single file format).
        :rtype: bool
        """
        if self._data is not None:
            return False  # single file format
        else:
            return True  # dual file format

    def save_exp_log(self, timestamp, path=""):
        """Save a log of the experiment to a Comma Separated Value (CSV) file at the path indicated by the user.

        The file contains several log items. The first column of the file contains the type of information presented
        by the given item while the second column contains the information itself.

        :param timestamp: the timestamp to be included in the file name.
        :type timestamp: float
        :param path: the path at which the file is to be saved (default ""). If "", the file is saved to a logs folder
            in the project root directory by default.
        :type path: str, optional
        """
        log_file = list()

        ############
        # DATA
        ############
        # -- HORIZONTAL VERSION --
        # objects/ranks/single file paths
        if self._single_path is None:
            header = ["Objects File Path", "Ranks File Path"]
            values = [str(self._obj_path), str(self._ranks_path)]
        else:
            header = ["Single File Path"]
            values = [str(self._single_path)]
        # num of objects
        # num of ranks
        header.extend(["NumSamples", "NumPreferences"])
        if self._is_dual_format:  # dual file format
            values.extend([str(len(self._objects)), str(len(self._ranks))])
        else:  # single file format
            values.extend([str(len(self._data)), "N/A"])
        # -- VERTICAL VERSION --
        # # objects/ranks/single file paths
        # if self._single_path is None:
        #     log_file.append(["Objects file path: ", str(self._obj_path)])
        #     log_file.append(["Ranks file path: ", str(self._ranks_path)])
        # else:
        #     log_file.append(["Single file path: ", str(self._single_path)])
        # # num of objects
        # # num of ranks
        # log_file.append(["Number of samples: ", str(len(self._objects))])
        # log_file.append(["Number of pairwise preferences (ranks): ", str(len(self._ranks))])

        ####################################
        # FEATURES INCLUDED & PRE-PROCESSING
        ####################################
        # -- HORIZONTAL VERSION --
        # autoencoder (yes/no + details)
        header.append("Automatic Feature Extraction")
        if self._autoencoder is None:
            values.append("No")
        else:
            values.append("Yes")
            # autoencoder loss
            header.append("Autoencoder-Loss")
            values.append(str(self._autoencoder_loss))
            # autoencoder details
            header.extend(list(self._autoencoder_details.keys()))
            values.extend(list(self._autoencoder_details.values()))

        # original (included) features
        # normalization
        norm_feat_names = [self._orig_feats[f] for f in range(len(self._orig_feats))]
        header.extend(["Original Included Features", "Normalization"])
        values.extend([str(norm_feat_names), str(list(self._norm_settings.values()))])
        # ^ for normalization, save dict values only !!!

        # shuffle
        header.extend(["Shuffle", "Shuffle Seed"])
        shuffle = "No"
        random_seed = "N/A"
        if self._shuffle:
            shuffle = "Yes"
            random_seed = str(self._random_seed)
        values.extend([shuffle, random_seed])

        # -- VERTICAL VERSION --
        # # original (included) features
        # log_file.append(["Original included features: ", self._orig_feats])
        # # normalization
        # log_file.append(["Normalization: ", self._norm_settings])

        ###################
        # FEATURE SELECTION
        ###################
        # -- HORIZONTAL VERSION --
        # selected features
        if self._fs_method is not None:
            header.append("FS Method")
            values.append(str(self._fs_method.get_name()))

            params = self._fs_method.get_params_string()
            params_string = params.strip("{")  # trim opening bracket
            params_string = params_string.strip("}")  # trim closing bracket
            params = params_string.split(";")  # extract list of params
            if params[0] != '':
                r = 0
                for param in params:
                    print(param)
                    # extract the name/key and value of the given parameter
                    key, value = param.split(":")
                    # trim whitespaces from left & right
                    key = key.strip()
                    value = value.strip()
                    # add to grid
                    name = text.real_param_name(str(key))
                    val = text.real_type_name(str(value))
                    header.append("FS-" + name)
                    values.append(str(val))
                    r += 1

            if self._fs_algo is not None:
                header.append("FS Algorithm")
                fs_algo_name = str(self._fs_algo.get_name())
                values.append(fs_algo_name)

                params = self._fs_algo.get_params_string()
                params_string = params.strip("{")  # trim opening bracket
                params_string = params_string.strip("}")  # trim closing bracket
                params = params_string.split(";")  # extract list of params
                if params[0] != '':
                    r = 0
                    for param in params:
                        print(param)
                        # extract the name/key and value of the given parameter
                        key, value = param.split(":")
                        # trim whitespaces from left & right
                        key = key.strip()
                        value = value.strip()
                        # add to grid
                        name = text.real_param_name(str(key))
                        val = text.real_type_name(str(value))
                        header.append("FS-Algo-" + name)
                        values.append(str(val))
                        r += 1

            if self._fs_eval is not None:
                header.append("FS Evaluation")
                fs_eval_name = str(self._fs_eval.get_name())
                values.append(fs_eval_name)

                params = self._fs_eval.get_params_string()
                params_string = params.strip("{")  # trim opening bracket
                params_string = params_string.strip("}")  # trim closing bracket
                params = params_string.split(";")  # extract list of params
                if params[0] != '':
                    r = 0
                    for param in params:
                        print(param)
                        # extract the name/key and value of the given parameter
                        key, value = param.split(":")
                        # trim whitespaces from left & right
                        key = key.strip()
                        value = value.strip()
                        # add to grid
                        name = text.real_param_name(str(key))
                        val = text.real_type_name(str(value))
                        header.append("FS-Eval-" + name)
                        values.append(str(val))
                        r += 1
            header.append("Selected features")
            values.append(str(self._features))
        # -- VERTICAL VERSION --
        # # selected features
        # if self._fs_method is not None:
        #     log_file.append(["FS Method: ", str(self._fs_method.get_name())])
        #     log_file.append(["FS Method params: ", self._fs_method.get_params_string()])
        #     if self._fs_algo is not None:
        #         log_file.append(["FS Algorithm: ", str(self._fs_algo.get_name())])
        #         log_file.append(["FS Algorithm params: ", self._fs_algo.get_params_string()])
        #     if self._fs_eval is not None:
        #         log_file.append(["FS Evaluator: ", str(self._fs_eval.get_name())])
        #         log_file.append(["FS Evaluator params: ", self._fs_eval.get_params_string()])
        #     log_file.append(["Selected features: ", self._features])

        #####################
        # PREFERENCE LEARNING
        #####################
        # -- HORIZONTAL VERSION --
        header.append("PL Algorithm")
        algo_name = str(self._pl_algo.get_name())
        values.append(algo_name)

        params = self._pl_algo.get_params_string()
        params_string = params.strip("{")  # trim opening bracket
        params_string = params_string.strip("}")  # trim closing bracket
        params = params_string.split(";")  # extract list of params
        if params[0] != '':
            r = 0
            for param in params:
                print(param)
                # extract the name/key and value of the given parameter
                key, value = param.split(":")
                # trim whitespaces from left & right
                key = key.strip()
                value = value.strip()
                # add to grid
                name = text.real_param_name(str(key))
                val = text.real_type_name(str(value))
                header.append("PL-Algo-" + name.strip())  # trim/strip whitespaces for RankSVM gamma param
                values.append(str(val))
                r += 1
        if self._pl_eval is not None:
            header.append("PL Evaluation")
            eval_name = str(self._pl_eval.get_name())
            values.append(eval_name)

            params = self._pl_eval.get_params_string()
            params_string = params.strip("{")  # trim opening bracket
            params_string = params_string.strip("}")  # trim closing bracket
            params = params_string.split(";")  # extract list of params
            if params[0] != '':
                r = 0
                for param in params:
                    print(param)
                    # extract the name/key and value of the given parameter
                    key, value = param.split(":")
                    # trim whitespaces from left & right
                    key = key.strip()
                    value = value.strip()
                    # add to grid
                    name = text.real_param_name(str(key))
                    val = text.real_type_name(str(value))
                    header.append("PL-Eval-" + name)
                    values.append(str(val))
                    r += 1
        # -- VERTICAL VERSION --
        # algorithm name
        # log_file.append(["PL Algorithm: ", str(self._pl_algo.get_name())])
        # log_file.append(["PL Algorithm params: ", self._pl_algo.get_params_string()])
        # if self._pl_eval is not None:
        #     log_file.append(["PL Evaluator: ", str(self._pl_eval.get_name())])
        #     log_file.append(["PL Evaluator params: ", self._pl_eval.get_params_string()])

        ###################################################################
        # ------------------------ FOLD-BASED INFO ------------------------
        ###################################################################

        # ======================== headers ========================

        n_folds = len(self._fold_metrics)
        header.append("Fold")

        ############
        # META DATA
        ############
        # start timestamp
        # end timestamp
        # duration
        # -- HORIZONTAL VERSION --
        header.extend(["Start Timestamp (UTC)", "End Timestamp (UTC)", "Duration"])
        ###################
        # MODEL PERFORMANCE
        ###################
        # training accuracy
        # test accuracy (if applicable)
        # any other metrics (if applicable)
        # -- HORIZONTAL VERSION --
        eval_metrics_example = self._fold_metrics[0][2]
        eval_metrics_headers = list(eval_metrics_example.keys())
        # print(eval_metrics_headers)
        header.extend(eval_metrics_headers)

        # print(header)
        # print(values)

        # ======================== values ========================

        # init last values of first column
        values.extend([None, None, None, None])  # fold, start, end, duration
        for metric in eval_metrics_example:
            values.append(None)
        # print(values)
        values = np.array(values)
        values = np.tile(values, (n_folds, 1))
        # print(values)
        log_file = pd.DataFrame(values, columns=header)

        # populate folds column!
        if n_folds > 1:
            log_file['Fold'] = np.arange(1, n_folds+1)
        else:  # if n_folds == 1
            log_file['Fold'] = 'N/A'

        # print(log_file)

        fold_metrics = np.array(self._fold_metrics)
        # print(fold_metrics)
        log_file.loc[:, ["Start Timestamp (UTC)"]] = fold_metrics[:, 0]
        log_file.loc[:, ["End Timestamp (UTC)"]] = fold_metrics[:, 1]
        print(log_file)
        log_file.loc[:, "Duration"] = log_file.loc[:, "End Timestamp (UTC)"] - log_file.loc[:, "Start Timestamp (UTC)"]
        log_file.loc[:, eval_metrics_headers] = np.array([[v for k, v in eval_metrics.items()] for
                                                          eval_metrics in fold_metrics[:, 2]])

        # for start_time, end_time, eval_metrics in self._fold_metrics:
        #     ############
        #     # META DATA
        #     ############
        #     # start timestamp
        #     # end timestamp
        #     # duration
        #     # -- HORIZONTAL VERSION --
        #     header.extend(["Start Timestamp (UTC)", "End Timestamp (UTC)", "Duration"])
        #     values = [str(start_time), str(end_time), str(end_time - start_time)]
        #     # -- VERTICAL VERSION --
        #     # log_file.append(["Start timestamp (UTC): ", str(self._start_time)])
        #     # log_file.append(["End timestamp (UTC): ", str(self._end_time)])
        #     # log_file.append(["Duration: ", str(self._end_time - self._start_time)])
        #
        #     ###################
        #     # MODEL PERFORMANCE
        #     ###################
        #     # training accuracy
        #     # test accuracy (if applicable)
        #     # any other metrics (if applicable)
        #     # -- HORIZONTAL VERSION --
        #     for metric in self._eval_metrics:
        #         header.append(str(metric))
        #         values.append(str(self._eval_metrics[metric]))
        #     # -- VERTICAL VERSION --
        #     # for metric in self._eval_metrics:
        #     #     log_file.append([str(metric) + ": ", str(self._eval_metrics[metric])])

        # Finally, save to file!
        # -- HORIZONTAL VERSION --
        if path == "":
            path = os.path.join(ROOT_PATH, "logs\\exp_" + str(timestamp) + ".csv")
        # values = [values]
        # print(header)
        # print(values)
        print(log_file)
        log_file.to_csv(path, index=False)
        # -- VERTICAL VERSION --
        # if path == "":
        #     path = os.path.join(ROOT_PATH, "logs\\exp_" + str(timestamp) + ".csv")
        # log_file = pd.DataFrame(log_file)
        # print(log_file)
        # log_file.to_csv(path)

    def save_model(self, timestamp, fold_idx=0, path=""):
        """Save the model or one of the models inferred in the experiment to file at the path indicated by the user.

        The resulting file is of a Comma Separated Value (CSV) format.

        :param timestamp: the timestamp to be included in the file name.
        :type timestamp: float
        :param fold_idx: the index of the fold for which the model is to be saved (default 0). This parameter should only
            be used in the case of multiple folds (e.g., when K-Fold Cross Validation is used).
        :type fold_idx: int, optional
        :param path: the path at which the file is to be saved (default ""). If "", the file is saved to a logs folder
            in the project root directory by default.
        :type path: str, optional
        """
        if self._fold_metrics is not None:
            # Get the model for the given fold
            model_df = self._fold_metrics[fold_idx][3]
            # Prepare file path
            if path == "":
                # default path
                path = os.path.join(ROOT_PATH, "logs\\model_" + str(timestamp) + ".csv")
            else:
                # add timestamp to file name
                path = path.rstrip(".csv")  # remove .csv
                path += ("_" + str(timestamp) + ".csv")  # add timestamp and .csv
            # Finally, save to file!
            model_df.to_csv(path, index=False)

    # Other getters and setters (mainly for use in GUI mode):

    def _set_objects(self, data, has_ids=False):
        """Set the objects data to be used in the experiment.

        :param data: the given objects data.
        :type data: `pandas.DataFrame`
        :param has_ids: specifies whether (True) or not (False) the objects data set already contains
            the object IDs in the first column (default False).
        :type has_ids: bool, optional
        """
        self._objects_have_id = has_ids
        self._objects = data
        self._data = None
        self._ranks = None

    def _set_ranks(self, data, has_ids):
        """Set the pairwise rank data to be used in the experiment.

        :param data: the given ranks data.
        :type data: `pandas.DataFrame`
        :param has_ids: specifies whether (True) or not (False) the ranks data set already contains
            the rank IDs in the first column.
        :type has_ids: bool, optional
        """
        self._ranks_have_id = has_ids
        self._ranks = data
        self._data = None

    def _set_single_data(self, data, has_ids):
        """Set the single file format data to be used in the experiment.

        :param data: the given data.
        :type data: `pandas.DataFrame`
        :param has_ids: specifies whether (True) or not (False) the data set already contains
            the object IDs in the first column.
        :type has_ids: bool, optional
        """
        self._samples_have_id = has_ids
        self._data = data
        self._objects = None
        self._ranks = None

    def get_time_meta_data(self):
        """Get meta data about the experiment related to time.

        :return: a list containing the start timestamp (UTC), end timestamp (UTC), and duration of the
            experiment.
        :rtype: list of float (size 3)
        """
        return [self._start_time, self._end_time, (self._end_time - self._start_time)]

    def _set_file_meta_data(self, obj_path=None, ranks_path=None, single_path=None,
                            norm_settings=None):
        """Set some of the meta data of the experiment.

        :param obj_path: the path of the objects data file used in the experiment (if applicable) (default None).
        :type obj_path: str, optional
        :param ranks_path: the path of the ranks data file used in the experiment (if applicable) (default None).
        :type ranks_path: str, optional
        :param single_path: the path of the single data file used in the experiment (if applicable) (default None).
        :type single_path: str, optional
        :param norm_settings: specifies the normalization settings for each feature with the indices of the
            features as the dict's keys and the corresponding type of normalization used as the dict's values.
        :type norm_settings: dict of str (names of :class:`pyplt.util.enums.NormalizationType`)
        """
        # hack for GUI only
        if not (obj_path == ""):
            self._obj_path = obj_path
        else:
            self._obj_path = None
        if not (ranks_path == ""):
            self._ranks_path = ranks_path
        else:
            self._ranks_path = None
        if not (single_path == ""):
            self._single_path = single_path
        else:
            self._single_path = None
        self._norm_settings = norm_settings

    def _get_objects(self):
        """Get the objects data used in the experiment (if applicable).

        :return: the objects data used in the experiment.
        :rtype: `pandas.DataFrame`
        """
        return self._objects

    def _get_ranks(self):
        """Get the pairwise rank data used in the experiment (if applicable).

        :return: the ranks data used in the experiment.
        :rtype: `pandas.DataFrame`
        """
        return self._ranks

    def _get_single_data(self):
        """Get the single file format data used in the experiment (if applicable).

        :return: the single file format data used in the experiment.
        :rtype: `pandas.DataFrame`
        """
        return self._data

    def get_data(self):
        """Get the loaded data.

        If the single file format is used, a single `pandas.DataFrame` containing the data is returned. If the dual
        file format is used, a tuple containing both the objects and ranks (each a `pandas.DataFrame`) is returned.

        :return: the loaded data.
        :rtype: `pandas.DataFrame` or tuple of `pandas.DataFrame` (size 2)
        """
        if self._data is None:
            return self._objects, self._ranks
        else:
            return self._data

    def get_features(self):
        """Get the features used in the experiment.

        :return: the names of the features used in the experiment.
        :rtype: list of str
        """
        return self._features

    def get_norm_settings(self):
        """Get the normalization settings for each feature in the original objects data.

        :return: a dict with the indices of the features as the dict's keys and the corresponding methods with which
            the features are to be normalized as the dict's values.
        :rtype: dict of str (names of :class:`pyplt.util.enums.NormalizationType`)
        """
        return self._norm_settings

    def get_pl_algorithm(self):
        """Get the preference learning algorithm used in the experiment.

        :return: the preference learning algorithm used in the experiment.
        :rtype: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        """
        return self._pl_algo

    def get_orig_features(self):
        """Get the original features used in the experiment.

        If automatic feature extraction was enabled, this method will return the extracted features.

        :return: the names of the features.
        :rtype: list of str
        """
        return self._orig_feats

    def get_autoencoder_loss(self):
        """Get the training loss of the autoencoder used in the experiment (if applicable).

        :return: the training loss.
        :rtype: float or None
        """
        return self._autoencoder_loss

    def get_autoencoder_details(self):
        """Get the details of the autoencoder used in the experiment (if applicable).

        :return: a dict containing the autoencoder parameter names as its keys and the parameter values as its values.
        :rtype: dict or None
        """
        return self._autoencoder_details

    # Debugging methods

    def print_objects(self):
        """Print the objects data used in the experiment to console."""
        print(self._objects)

    def print_ranks(self):
        """Print the pairwise rank data used in the experiment to console."""
        print(self._ranks)


# Internal functions (used by GUI modules or Experiment):

def _load_file(file_path, separator, headers, col_names, na_filter=True):
    """Extract a `pandas.DataFrame` object from a given CSV file using the :func:`pandas.read_csv()`.

    :param file_path: the path of the file to be loaded.
    :type file_path: str
    :param separator: the character separating items in the CSV file.
    :type separator: str
    :param headers: row number(s) to use as the column names, and the start of the data.
    :type headers: int or list of int or 'infer'
    :param col_names: an optional list of column names to be used.
    :type col_names: array-like or None
    :param na_filter: a boolean indicating whether to detect missing value markers (default True).
    :type na_filter: bool, optional
    :return: the dataframe object extracted from the file.
    :rtype: `pandas.DataFrame`
    """
    df = pd.read_csv(
        file_path,
        sep=separator,  # default ','
        header=headers,  # default 'infer'
        names=col_names,  # default None
        na_filter=na_filter,  # default True
        engine='python'  # because default engine 'c' does not support regex separators
    )
    return df


def _load_data(file_type, file_path, has_fnames=False, has_ids=False, separator=',', col_names=None,
               na_filter=True):
    """Load data from the given file as a pandas DataFrame and processes it according to its type.

    :param file_type: the type of file being loaded.
    :type file_type: :class:`pyplt.util.enums.FileType`
    :param file_path: the path of the file to be loaded.
    :type file_path: str
    :param has_fnames: specifies whether or not the file already contains feature names
        in the first row (default False).
    :type has_fnames: bool, optional
    :param has_ids: specifies whether or not the file already contains object IDs
        in the first column (default False).
    :type has_ids: bool, optional
    :param separator: the character separating items in the CSV file (default ',').
    :type separator: str, optional
    :param col_names: an optional list of column names to be used (default None).
    :type col_names: list of str or None, optional
    :param na_filter: specifies whether to detect missing value markers (default True).
    :type na_filter: bool, optional
    :return: the `pandas.DataFrame` object containing the data extracted and processed from the given file.
    :rtype: `pandas.DataFrame`
    """
    headers = None
    if has_fnames:
        headers = 'infer'

    # load df
    data = _load_file(file_path, separator, headers, col_names, na_filter)

    # to add/reset ID column name, add underscores at beginning (e.g. "__ID") until the column name is unique
    id_col_name = "ID"
    while id_col_name in data.columns:
        id_col_name = "_" + id_col_name

    # if file does not contain object ids as first column
    if not has_ids:
        new_ids_col = list(np.arange(len(data)))
        # add row_id column as "(_)*ID"
        data.insert(loc=0, column=id_col_name, value=new_ids_col)

    # if file does not contain feature names
    if headers is None:
        if file_type == FileType.OBJECTS:
            old_col_names = data.columns
            new_col_names = [id_col_name] + ["F" + str(x) for x in range(len(data.columns) - 1)]
            # print(new_col_names)
            # len-1 in above line accounts for ID column
            data.rename(columns={i: j for i, j in zip(old_col_names, new_col_names)}, inplace=True)
        elif file_type == FileType.RANKS:
            old_col_names = data.columns
            new_col_names = [id_col_name] + ["Preferred Object", "Non-Preferred Object"]
            # print(new_col_names)
            # len-1 in above line accounts for ID column
            data.rename(columns={i: j for i, j in zip(old_col_names, new_col_names)}, inplace=True)
        else:  # i.e. FileType.SINGLE
            old_col_names = data.columns
            new_col_names = [id_col_name] + ["F" + str(x) for x in range(len(data.columns) - 2)] + ['Order']
            # print(new_col_names)
            # len-1 in above line accounts for ID column
            data.rename(columns={i: j for i, j in zip(old_col_names, new_col_names)}, inplace=True)

    # otherwise make sure to rename first column to "(_)*ID" just in case
    if has_ids and headers == 'infer':
        data.rename(columns={0: id_col_name}, inplace=True)

    return data


def _split_single(single_df, mdm=0.0, memory='all'):
    """Split a `pandas.DataFrame` object extracted from a single file into an objects dataframe and a ranks dataframe.

    A single file is indicated by the type :attr:`pyplt.util.enums.FileType.SINGLE`.

    :param single_df: the data extracted from the single data file.
    :type single_df: `pandas.DataFrame`
    :param mdm: the minimum distance margin i.e., the minimum difference between the ratings of a given pair of
        objects/samples that is required for the object pair to be considered a valid and clear preference
        (default 0.0).
    :type mdm: float, optional
    :param memory: specifies how many neighbouring objects/samples are to be compared with a given object/sample
        when constructing the pairwise ranks (default 'all'). For example, pairs of objects that satisfy a memory of 2
        should be no more than 2 objects away from each other in terms of their object ID (integer).
        If 'all', all objects/samples are compared to each other.
    :type memory: int or 'all', optional
    :return: a tuple of size 2 containing the objects dataframe and the ranks dataframe derived from
        the single data file.
    :rtype: tuple of `pandas.DataFrame` (size 2)
    :raises NoRanksDerivedError: if no pairwise preferences could be derived from the given data. This is either
        because there are no clear pairwise preferences in the data or because none of the clear pairwise
        preferences in the data conform to the chosen values for the rank derivation parameters (i.e., the minimum
        distance margin (`mdm`) and the memory (`memory`) parameters).
    :raises InvalidParameterValueException: if the user attempts to use a negative value (i.e., smaller than 0.0)
        for the `mdm` parameter.
    """
    print("Deriving ranks with mdm=" + str(mdm) + " and memory=" + str(memory))

    # parse for objects
    objects = single_df.iloc[:, :-1]  # all rows and all columns except last (i.e., the ratings)
    # print(objects)

    # parse for ranks
    ratings = single_df.iloc[:, -1]  # all rows and only last column

    # compare every object to every other object
    # pairs = matrix n x n matrix (where n = #objects)
    pairs = np.subtract.outer(ratings, ratings)

    if mdm < 0.0:
        raise InvalidParameterValueException(parameter="minimum distance margin (MDM)", value=mdm,
                                             method="rank derivation", is_algorithm=False,
                                             additional_msg="MDM cannot be negative. It must have a value greater "
                                                            "than or equal to 0.0.")

    # select ranks where difference between object ratings is > minimum distance margin (e.g. 0)
    # ranks_a contains the preferred object indexes, ranks_b contains the non-preferred object indexes
    # n.b. this is only one way so no duplicates are created!
    ranks_a, ranks_b = np.where(pairs > mdm)

    print(ranks_a)
    print(ranks_b)

    # convert object indexes into actual OBJECT IDS
    prefs = objects.iloc[ranks_a].index.tolist()
    nons = objects.iloc[ranks_b].index.tolist()
    # ^ N.B. ID col has already been set as index
    # ^^^ confirmed to work for ratings-auto, ratings-manual

    print(prefs)
    print(nons)

    # convert to numpy array of 2 columns
    _ranks = np.array(list(zip(prefs, nons)))

    print(_ranks)

    if _ranks.shape[0] == 0:  # if zero rows i.e. no clear/valid pairwise preferences were found
        raise NoRanksDerivedError

    # weed out rank pairs that do not conform to memory parameter (if applicable)
    # do this by comparing the OBJECT IDS (not indices) of the objects in the preference pairs
    if memory != 'all':
        memory_dist = np.absolute(np.subtract(_ranks[:, 0], _ranks[:, 1]))
        r_indices = np.where(memory_dist <= memory)
        r_indices = r_indices[0]  # because it is 1D
        _ranks = _ranks[r_indices, :]

    # print("_ranks.shape:")
    # print(_ranks.shape)

    if _ranks.shape[0] == 0:  # if zero rows i.e. no clear/valid pairwise preferences were found
        raise NoRanksDerivedError

    # convert object indexes into actual object IDs
    # and store ranks properly in array of 2 columns (actually list of lists)
    # prefs = objects.iloc[_ranks[:, 0], 0].values.reshape(-1, 1)
    # nons = objects.iloc[_ranks[:, 1], 0].values.reshape(-1, 1)
    # prefs = np.array(objects.iloc[_ranks[:, 0]].index.tolist()).reshape(-1, 1)
    # nons = np.array(objects.iloc[_ranks[:, 1]].index.tolist()).reshape(-1, 1)
    # ^ N.B. ID col has already been set as index
    # ^^^ confirmed to work for ratings-auto
    # print(prefs)
    # print(nons)
    # ranks = np.concatenate((prefs, nons), axis=1).tolist()
    ranks = _ranks

    # print(ranks)

    # Convert ranks to pandas DataFrame
    ranks = pd.DataFrame(ranks, columns=["Preferred Object", "Non-Preferred Object"])

    # Add rank ids as first column
    new_ids_col = list(np.arange(len(ranks)))
    ranks.insert(loc=0, column='ID', value=new_ids_col)

    # Convert '(_)*ID' column in dataset to its index
    ranks = ranks.set_index(ranks.columns[0])

    return objects.copy(deep=True), ranks


def _normalize(objects, feature_ids, norm_method, train=None, test=None, feature_names=None, min_val=0, max_val=1):
    """Internal method used for the normalization of values for the given feature.

    :param objects: the objects data to be normalized.
    :type objects: `pandas.DataFrame`
    :param feature_ids: the index or list of indexes of the feature/s (column/s in the objects data set) to be
        normalized.
    :type feature_ids: int or list of int
    :param norm_method: the normalization method to be used.
    :type norm_method: :class:`pyplt.util.enums.NormalizationType`
    :param train: integer-based indices of the samples in the training set (default None). The
        transformer first fits to these samples and then transforms them. If none, it is assumed that all samples
        in `objects` constitute the training set and are therefore all treated in this manner.
    :type train: `numpy.ndarray` or None, optional
    :param test: integer-based indices of the samples in the test set (default None). These samples
        are transformed based on the fit acquired from the training set.
    :type test: `numpy.ndarray` or None, optional
    :param feature_names: the name or names of the features (column/s in the objects data set) to
        be normalized (default None).
    :type feature_names: str or list of str, optional
    :param min_val: the value to be considered the minimum of the newly normalized values (if applicable,
        e.g., for Min-Max method) (default 0).
    :type min_val: int or float, optional
    :param max_val: the value to be considered the maximum of the newly normalized values (if applicable,
        e.g., for Min-Max method) (default 1).
    :type max_val: int or float, optional
    :return: the objects data updated with the newly normalized values for the given feature.
    :rtype: `pandas.DataFrame`
    :raises NormalizationValueError: if one of the given values in `objects` cannot be converted to int or float
        prior to the normalization, indicating a failure in carrying out normalization.
    """
    norm_test_objects = None
    scaler = None
    if norm_method == NormalizationType.MIN_MAX:
        scaler = skpp.MinMaxScaler(feature_range=(min_val, max_val), copy=True)
    elif norm_method == NormalizationType.Z_SCORE:
        scaler = skpp.StandardScaler(copy=True)
    # TODO: Binary method (but not skpp.binarize) as in Java PLT?

    if scaler is None:
        return objects.copy(deep=True), None
    else:
        print("normalizing features " + str(feature_ids))
        try:
            if train is None:
                norm_objects = scaler.fit_transform(objects.iloc[:, feature_ids])
            else:
                norm_objects = scaler.fit_transform(objects.iloc[train, feature_ids])
            if test is not None:
                norm_test_objects = scaler.transform(objects.iloc[test, feature_ids])
        except (ValueError, TypeError):
            # value=str(str(ve.args[0]).split("\'")[1])
            raise NormalizationValueError(f_id=feature_ids, f_name=feature_names, norm_method=norm_method)
            # ^ TODO: error info probably not very helpful since method is now list-based not individual...
    return norm_objects, norm_test_objects
