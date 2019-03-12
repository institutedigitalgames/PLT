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

"""This module specifies which algorithms and methods are supported by the GUI of PLT.

This module also provides generic helper functions to create instances of algorithm/method classes given enumerated
constants representing them.

Developers who would like to add new feature selection methods, preference learning algorithms, and evaluation methods
to the GUI of PLT, can do so easily by adding the algorithm/method class and GUI menu class (if applicable) to the
dicts declared in this module:

* :attr:`supported_fs_methods` -- for feature selection methods (in Expert Mode);
* :attr:`supported_algorithms` -- for preference learning algorithms (in Expert Mode);
* :attr:`supported_algorithms_beginner` -- for preference learning algorithms (in Beginner Mode);
* :attr:`supported_evaluation_methods` -- for evaluation methods (in Expert Mode).
"""

from pyplt.util.enums import FSMethod, PLAlgo, EvaluatorType
from pyplt.fsmethods import sfs
from pyplt.plalgorithms import ranksvm, backprop_tf, ranknet
from pyplt.evaluation import holdout, cross_validation
from pyplt.gui.experiment.preflearning import ranksvm_menu, backprop_menu, evaluator_menus, ranknet_menu

# lists of supported algorithms/methods and their corresponding GUI menus
# as dicts of tuples with
# - algorithm/method type enums as keys, and
# - tuples of size 2 with format (algorithm/method class, GUI menu class for algorithm/method) as values

# -------------------------------- Feature Selection Methods --------------------------------

supported_fs_methods = {
    FSMethod.NONE: (None, None),  # NONE - No feature selection
    FSMethod.SFS: (sfs.SFS, None)  # SFS
    # ^ TODO: add new feature selection methods (for ADVANCED MODE) here in the same way
}

# -------------------------------- Preference Learning Algorithms --------------------------------
supported_algorithms = {
    PLAlgo.RANKSVM: (ranksvm.RankSVM, ranksvm_menu.RankSVMMenu),  # RankSVM
    PLAlgo.BACKPROPAGATION: (backprop_tf.BackpropagationTF, backprop_menu.BackpropMenu),  # Backpropagation (tensorflow)
    PLAlgo.RANKNET: (ranknet.RankNet, ranknet_menu.RankNetMenu)  # RankNet
    # ^ TODO: add new algorithms (for ADVANCED MODE) here in the same way
    # N.B. if the GUI Menu requires additional parameters (like BackpropMenu, for example), make sure to update the
    #   pyplt.gui.experiment.featureselection.featselectiontab.FeatureSelectionFrame#_update_algo_menu() and
    #   pyplt.gui.experiment.preflearning.pltab.PLFrame#_update_algo_menu() methods accordingly.
}

# for the Beginner mode, we only need the classes, not the menus
supported_algorithms_beginner = {
    PLAlgo.RANKSVM: ranksvm.RankSVM,  # RankSVM
    PLAlgo.BACKPROPAGATION: backprop_tf.BackpropagationTF,  # Backpropagation (tensorflow)
    PLAlgo.RANKNET: ranknet.RankNet  # RankNet
    # ^ TODO: add new algorithms (for BEGINNER MODE) here in the same way
    # N.B. to specify parameter values other than the defaults, make sure to update the preference learning section
    #   of the pyplt.gui.beginnermenu.BeginnerMenu#_run_exp() method accordingly.
}

# -------------------------------- Evaluation Methods --------------------------------
supported_evaluation_methods = {
    EvaluatorType.NONE: (None, None),  # NONE - No evaluation
    EvaluatorType.HOLDOUT: (holdout.HoldOut, evaluator_menus.HoldoutMenu),  # Holdout validation
    EvaluatorType.KFCV: (cross_validation.KFoldCrossValidation, evaluator_menus.KFCVMenu)  # K-Fold Cross Validation
    # ^ TODO: add new evaluation methods (for ADVANCED MODE) here in the same way
    # N.B. if the GUI Menu requires additional parameters (like KFCVMenu, for example), make sure to update the
    #   pyplt.gui.experiment.featureselection.featselectiontab.FeatureSelectionFrame#_update_eval_menu() and
    #   pyplt.gui.experiment.preflearning.pltab.PLFrame#_update_eval_menu() methods accordingly.
}


def get_algorithm_instance(algorithm_enum, params=None, beginner_mode=False):
    """Create an instance of the preference learning algorithm class represented by the given enum constant.

        Each enumerated constant of type :class:`pyplt.util.enums.PLAlgo` in the :attr:`supported_algorithms` dict
        corresponds to a class of type (extending) :class:`pyplt.plalgorithms.base.PLAlgorithm`.

        If `params` is specified, the instance is initialized with the given algorithm parameter values. Otherwise,
        the default values are used.

        :param algorithm_enum: the algorithm type (enum).
        :type algorithm_enum: :class:`pyplt.util.enums.PLAlgo`
        :param params: optional algorithm parameter values in the form of a dict (default None). The keys of the
            dict should match the keywords of the arguments that would be passed to the corresponding
            :class:`pyplt.plalgorithms.base.PLAlgorithm` constructor.
            For example, for the `Backpropagation` algorithm the dict should contain the following items:

            * ann_topology: the topology of the neurons in the network
            * learn_rate: the learning rate
            * error_threshold: the error threshold
            * epochs: the number of epochs
            * activation_functions: the activation functions for each neuron layer in the network

            On the other hand, for the `RankSVM` algorithm the dict should contain the following items:

            * kernel: the kernel name
            * gamma: the gamma kernel parameter value
            * degree: the degree kernel parameter value

        :type params: dict or None, optional
        :param beginner_mode: specifies whether or not the algorithm is being used in the beginner mode (default False).
        :type beginner_mode: bool, optional
        :return: an instance of the class corresponding to the given algorithm.
        :rtype: :class:`pyplt.plalgorithms.base.PLAlgorithm`
        :raises InvalidParameterValueException: if the user attempted to use a value smaller or equal to 0.0
            for the `gamma` parameter of the `RankSVM` algorithm.
    """
    if beginner_mode:
        algorithm_class = supported_algorithms_beginner[algorithm_enum]  # get algorithm class
    else:
        algorithm_class = supported_algorithms[algorithm_enum][0]  # get algorithm class
    if algorithm_class is not None:
        if params is None:
            algorithm_instance = algorithm_class()
        else:
            algorithm_instance = algorithm_class(**params)
        return algorithm_instance
    return None


def get_fs_method_instance(fs_method_enum, params=None):
    """Create an instance of the feature selection method class represented by the given enum constant.

    Each enumerated constant of type :class:`pyplt.util.enums.FSMethod` in the :attr:`supported_fs_methods` dict
    corresponds to a class of type (extending) :class:`pyplt.fsmethods.base.FeatureSelectionMethod`.

    If `params` is specified, the instance is initialized with the given feature selection method parameter values.
    Otherwise, the default values are used.

    :param fs_method_enum: the feature selection method type (enum).
    :type fs_method_enum: :class:`pyplt.util.enums.FSMethod`
    :param params: optional feature selection method parameter values in the form of a dict (default None). The keys
        of the dict should match the keywords of the arguments that would be passed to the corresponding
        :class:`pyplt.fsmethods.base.FeatureSelectionMethod` constructor.
    :type params: dict or None, optional
    :return: an instance of the class corresponding to the given feature selection method.
    :rtype: :class:`pyplt.fsmethods.base.FeatureSelectionMethod`
    """
    fs_method_class = supported_fs_methods[fs_method_enum][0]  # get algorithm class
    if fs_method_class is not None:
        if params is None:
            fs_method_instance = fs_method_class()  # ignore warning - we take care of this by checking for None
        else:
            fs_method_instance = fs_method_class(**params)  # ignore warning - we take care of this by checking for None
        return fs_method_instance
    return None


def get_eval_method_instance(eval_method_enum, params=None):
    """Create an instance of the evaluation method class represented by the given enum constant.

    Each enumerated constant of type :class:`pyplt.util.enums.EvaluatorType` in the
    :attr:`supported_evaluation_methods` dict corresponds to a class of type (extending)
    :class:`pyplt.evaluation.base.Evaluator`.

    If `params` is specified, the instance is initialized with the given evaluation method parameter values.
    Otherwise, the default values are used.

    :param eval_method_enum: the evaluation method type (enum).
    :type eval_method_enum: :class:`pyplt.util.enums.EvaluatorType`
    :param params: optional evaluation method parameter values in the form of a dict (default None).
        The keys of the dict should match the keywords of the arguments
        that would be passed to the corresponding :class:`pyplt.evaluation.base.Evaluator` constructor.
        For example, for the `Holdout` method, the dict should contain the following items:

        * test_proportion: a float specifying the proportion of data to be used as training data (the rest
          is to be used as test data) or None

        On the other hand, for the `KFoldCrossValidation` method, the dict should contain the following items:

        * k: the number of folds to uniformly split the data into when using the automatic approach or None
        * test_folds: an array specifying the fold index for each sample in the dataset when using
          the manual approach or None

    :type params: dict or None, optional
    :return: an instance of the class corresponding to the given evaluation method.
    :rtype: :class:`pyplt.evaluation.base.Evaluator`
    :raises InvalidParameterValueException: if the user attempts to use a value smaller than 2 for
        the `k` parameter of K-Fold Cross Validation.
    :raises MissingManualFoldsException: if the user chooses to specify folds manually for cross validation but
        fails to load the required file containing the fold IDs.
    """
    eval_method_class = supported_evaluation_methods[eval_method_enum][0]  # get algorithm class
    if eval_method_class is not None:
        if params is None:
            eval_method_instance = eval_method_class()  # ignore warning - we take care of this by checking for None
        else:
            eval_method_instance = eval_method_class(**params)  # ignore warning - we take care of this by checking for None
        return eval_method_instance
    return None
