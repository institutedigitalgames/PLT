# Preference Learning Toolbox
# Copyright (C) 2019 Institute of Digital Games, University of Malta
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
import time
# from keras import backend
import keras
import pandas as pd
from keras import backend
from keras.engine import InputLayer
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model

from pyplt import ROOT_PATH
from pyplt.experiment import Experiment
from pyplt.plalgorithms.backprop_tf import ActivationType
from pyplt.util.enums import NormalizationType, PLAlgo
from pyplt.evaluation.cross_validation import KFoldCrossValidation
from pyplt.plalgorithms.base import PLAlgorithm


class RankNet(PLAlgorithm):
    """RankNet algorithm implemented with the `keras` package.

    The RankNet algorithm is an extension of the Backpropagation algorithm which uses a probabilistic cost function to
    handle ordered pairs of data. As in Backpropagation, the algorithm iteratively (over a given number of epochs)
    optimizes the error function by adjusting the weights of an artificial neural network (ANN) model proportionally
    to the gradient of the error with respect to the current value of the weights and current data samples. The
    error function used is the binary cross-entropy function. The proportion and therefore the strength of each
    update is regulated by the given learning rate. The total error is averaged over the complete set of pairs in the
    training set. In PLT, the algorithm was implemented using the `keras` library.
    """

    def __init__(self, ann_topology=None, learn_rate=0.001, epochs=100, hidden_activation_functions=None,
                 batch_size=32, debug=False):
        """Initialize the RankNet instance.

        :param ann_topology: a list indicating the topology of the artificial neural network (ANN) to be used with the
            algorithm. The list contains the number of neurons in each layer of the ANN, excludes the input layer but
            including the output layer (must always be 1 neuron in size); a value of None is equivalent to [1]
            indicating an ANN with no hidden layers and only an output layer (consisting of 1 neuron) (default None).
        :type ann_topology: list or None, optional
        :param hidden_activation_functions: a list of the activation function to be used across the neurons
            for each hidden layer of the ANN; if None, all hidden layers will use the Rectified Linear Unit (ReLU)
            function i.e. :attr:`pyplt.plalgorithms.backprop_tf.ActivationType.RELU` (default None). Note that
            this parameter excludes the activation function at the output layer of the network which is fixed.
        :type hidden_activation_functions: list of :class:`pyplt.plalgorithms.backprop_tf.ActivationType` or None,
            optional
        :param learn_rate: the learning rate used in the weight update step of the Backpropagation algorithm
            (default 0.001).
        :type learn_rate: float, optional
        :param epochs: the maximum number of iterations the algorithm should make over the entire pairwise rank
            training set (default 10).
        :type epochs: int, optional
        :param batch_size: number of samples per gradient update (default 32).
        :type batch_size: int, optional
        :param debug: specifies whether or not to print notes to console for debugging (default False).
        :type debug: bool, optional
        """
        self._n_feats = None
        self._rel_score = None
        self._irr_score = None
        self._model = None
        self._rel_doc = None
        self._pref_score = None
        self._non_pref_score = None
        self._pref_x = None
        if ann_topology is None:
            ann_topology = [128, 1]
        num_hidden_layers = len(ann_topology)-1
        if hidden_activation_functions is None:
            hidden_activation_functions = [ActivationType.RELU.name for _ in range(num_hidden_layers)]
        else:
            hidden_activation_functions = [actf.name for actf in hidden_activation_functions]  # get name of enums
        self._hidden_activation_functions = hidden_activation_functions
        self._topology = ann_topology
        self._learn_rate = learn_rate
        self._epochs = epochs
        self._batch_size = batch_size

        desc = "The RankNet algorithm is an extension of the Backpropagation algorithm which uses a probabilistic " \
               "cost function to handle ordered pairs of data. As in Backpropagation, the algorithm " \
               "iteratively (over a given number of epochs) optimizes the error function by adjusting the " \
               "weights of an artificial neural network (ANN) model proportionally to the gradient of the " \
               "error with respect to the current value of the weights and current data samples. The error " \
               "function used is the binary cross-entropy function. The proportion and therefore the strength " \
               "of each update is regulated by the given learning rate. The total error is averaged over the " \
               "complete set of pairs in the training set. In PLT, the algorithm was implemented using the " \
               "`keras` library."

        super().__init__(description=desc, debug=debug, name=PLAlgo.RANKNET.name,
                         ann_topology=self._topology, hidden_activation_functions=self._hidden_activation_functions,
                         learn_rate=self._learn_rate, epochs=self._epochs)

    def init_train(self, n_features):
        """Initialize the model (topology).

        This is done by declaring `keras` placeholders, variables, and operations. This may also be used,
        for example, to simply modify (re-initialize) the topology of the model while evaluating different
        feature sets during wrapper-type feature selection processes.

        :param n_features: the number of features to be used during the training process.
        :type n_features: int
        """
        self._n_feats = n_features

        num_of_hidden_layers = len(self._topology) - 1

        h = []
        for i in range(num_of_hidden_layers):
            h.append(Dense(self._topology[i], activation=str(self._hidden_activation_functions[i]).lower()))
            # ^ e.g. "relu"

        s = Dense(1)  # no activation function (linear) i.e. a(x) = x

        # Preferred example score.
        pref_x = Input(shape=(self._n_feats,), dtype="float32")
        out = pref_x
        for i in range(num_of_hidden_layers):
            out = h[i](out)
        pref_score = s(out)

        # Non preferred example score.
        non_pref_x = Input(shape=(self._n_feats,), dtype="float32")
        out = non_pref_x
        for i in range(num_of_hidden_layers):
            out = h[i](out)
        non_pref_score = s(out)

        # Subtract scores.
        diff = Subtract()([pref_score, non_pref_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)  # output layer - activation function is fixed to sigmoid

        # Build model.
        model = Model(inputs=[pref_x, non_pref_x], outputs=prob)
        optimizer = keras.optimizers.Adam(lr=self._learn_rate)
        model.compile(optimizer=optimizer, loss="binary_crossentropy")

        self._pref_score = pref_score
        self._non_pref_score = non_pref_score
        self._model = model
        self._pref_x = pref_x

    def train(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Infer an ANN model using the given training data.

        :param train_objects: the objects data to train the model on.
        :type train_objects: `pandas.DataFrame`
        :param train_ranks: the pairwise rank data to train the model on.
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
            * True -- if execution is completed successfully.
            * None -- if experiment is aborted before completion by `exec_stopper`.
        """
        print("Starting training with RankNet.")
        if progress_window is not None:
            progress_window.put("Starting training with RankNet.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankNet execution...")
            return

        train_objects_ = train_objects.copy(deep=True)
        train_ranks_ = train_ranks.copy(deep=True)
        self._n_feats = len(train_objects_.columns)

        self.init_train(self._n_feats)

        prefs_x = train_objects_.loc[list(train_ranks_.iloc[:, 0])].values
        nons_x = train_objects_.loc[list(train_ranks_.iloc[:, 1])].values

        y = np.ones((prefs_x.shape[0], 1))

        print("ACTUALLY STARTING TRAINING...")

        BATCH_SIZE = self._batch_size
        self._model.fit([prefs_x, nons_x], y, batch_size=BATCH_SIZE, epochs=self._epochs, verbose=1)

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankNet execution...")
            return

        print("Training complete.")
        if progress_window is not None:
            progress_window.put("Training complete.")

        return True

        # Generate scores from features.

    #        get_score = backend.function([self._pref_x], [self._pref_score])
    #        get_score([prefs_x])
    #        get_score([nons_x])

    def calc_train_accuracy(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to calculating the training accuracy of the learned model.

        This method is implemented explicitly for this algorithm since this approach is substantially more efficient
        for algorithms using the `keras` package than the calc_train_accuracy() method of
        :class:`pyplt.plalgorithms.base.PLAlgorithm` objects allows.

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
        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankNet execution...")
            return

        train_objects_ = train_objects.copy(deep=True)
        train_ranks_ = train_ranks.copy(deep=True)
        self._n_feats = len(train_objects_.columns)

        prefs_x = train_objects_.loc[list(train_ranks_.iloc[:, 0])].values
        nons_x = train_objects_.loc[list(train_ranks_.iloc[:, 1])].values

        predictions = self._model.predict([prefs_x, nons_x])
        errors = np.where(predictions < 0.5)[0].shape[0]
        acc = 100.0 - (100.0 * (errors / predictions.shape[0]))

        print("Performance (TRAINING): " + str(acc))

        if progress_window is not None:
            progress_window.put("Training accuracy: " + str(acc))

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankNet execution...")
            return

        return acc

    def test(self, objects, test_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to testing/validating the model using the given test data.

        This method is implemented explicitly for this algorithm since this approach is substantially more efficient
        for algorithms using the `keras` package than the test() method of the base class
        :class:`pyplt.plalgorithms.base.PLAlgorithm`.

        :param objects: the objects data for the model to be tested/validated on.
        :type objects: `pandas.DataFrame`
        :param test_ranks: the pairwise rank data for the model to be tested/validated on.
        :type test_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used during the testing/validation process;
            if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the test/validation accuracy of the learned model -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: float
        """
        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankNet test execution...")
            return

        objects_ = objects.copy(deep=True)
        test_ranks_ = test_ranks.copy(deep=True)
        self._n_feats = len(objects_.columns)

        prefs_x = objects_.loc[list(test_ranks_.iloc[:, 0])].values
        nons_x = objects_.loc[list(test_ranks_.iloc[:, 1])].values

        predictions = self._model.predict([prefs_x, nons_x])
        errors = np.where(predictions < 0.5)[0].shape[0]
        acc = 100.0 - (100.0 * (errors / predictions.shape[0]))

        print("Performance (TEST): " + str(acc))

        if self._debug:
            print("Performance (TEST): " + str(acc) + "%")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting RankNet test execution...")
            return

        return acc

    def predict(self, input_object, progress_window=None, exec_stopper=None):
        # TODO: write docstring
        # TODO: implement
        return  # None

    def transform_data(self, object_):
        """Transform a sample (object) into the format required by this particular algorithm implementation.

        In this case, no transformation is needed.

        :param object_: the data sample (object) to be transformed.
        :type object_: one row from a `pandas.DataFrame`
        :return: the transformed object (same as object_ in this case).
        """
        return object_

    def save_model(self, timestamp, path="", suppress=False):
        """Save the trained model to file in a human-readable format.

        Optionally, the file creation may be suppressed and a `pandas.DataFrame` representation of the model
        returned instead.

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
        # save (export) model to file

        # format model for saving to file
        if path == "":
            path = os.path.join(ROOT_PATH, "logs\\model_" + str(timestamp) + ".csv")

        # for debug:
        # print("MODEL WEIGHTS AND BIASES:")
        # print(self._model.get_weights())  # N.B. last layer in self._model.layers is the subtraction layer thing...
        #
        # print("number of layers in model: " + str(len(self._model.layers)))
        # print("number of layers (WITH WEIGHTS) in model: " + str(len(self._model.get_weights())))
        #
        # print("model summary:")
        # print(self._model.summary())
        #
        # weights_and_biases = self._model.get_weights()
        # for layer in range(len(self._model.layers)):
        #     print(str(self._model.layers[layer]))
        #     print("layer " + str(layer) + " shape [0] " + str(weights_and_biases[layer].shape[0]))
        #     try:
        #         print("layer " + str(layer) + " shape [1] " + str(weights_and_biases[layer].shape[1]))
        #     except IndexError:
        #         continue

        n_rows = 0
        cols = ["Layer", "Neuron"]
        max_layer_size = 0
        for layer in self._model.layers:
            # skip Input layers, Subtract layer, and final output / Activation layer bc they do not have weights
            if isinstance(layer, InputLayer) or isinstance(layer, Subtract) or isinstance(layer, Activation):
                continue
            # print("layer " + str(layer))
            print("layer weights and biases:")
            print(layer.get_weights())
            n = layer.get_weights()[0].shape[0]
            # ^ i.e. n_in for that layer # self._ann_topology[layer]
            n_rows += layer.get_weights()[0].shape[1]  # i.e. n_out for that layer
            if n > max_layer_size:
                max_layer_size = n
        cols.extend(["w" + str(w) for w in range(max_layer_size)])
        cols.append("bias")
        cols.append("activation_fn")

        # print(cols)
        model_arr = np.empty(shape=[n_rows, len(cols)], dtype=object)

        n_layers = len(self._topology)
        l_curr = 0
        row = 0
        for layer in self._model.layers:
            # skip Input layers, Subtract layer, and final output / Activation layer bc they do not have weights
            if isinstance(layer, InputLayer) or isinstance(layer, Subtract) or isinstance(layer, Activation):
                continue
            if l_curr == n_layers - 1:
                name = "OUTPUT"
            else:
                name = "h" + str(l_curr + 1)
            w_and_b = layer.get_weights()
            layer_weights = w_and_b[0]
            layer_biases = w_and_b[1]
            n_in = layer_weights.shape[0]
            # print("n_in " + str(n_in))
            n_out = layer_weights.shape[1]
            # print("n_out " + str(n_out))
            if l_curr == n_layers - 1:
                activation_fns = "LINEAR"
            else:
                activation_fns = self._hidden_activation_functions[l_curr]
            model_arr[row:row + n_out, 0] = name
            model_arr[row:row + n_out, 1] = range(n_out)
            # 1. store weights for each neuron in the layer
            model_arr[row:row + n_out, 2:2 + n_in] = layer_weights.T
            # 2. store bias for each neuron in the layer
            model_arr[row:row + n_out, -2] = layer_biases
            # 3. store activation function for each neuron in a new last column!
            model_arr[row:row + n_out, -1] = activation_fns
            row += n_out
            l_curr += 1

        model_df = pd.DataFrame(model_arr, columns=cols)

        # 3. n_layers can be inferred from num unique values in 'Layer' col +1 (input layer)
        # 4. n_outputs can be inferred from num unique 'Neuron' values where 'Layer'=='OUTPUT'

        if suppress:
            return model_df
        else:
            # Finally, save to file!
            model_df.to_csv(path, index=False)

    def clean_up(self):
        """Close the backend `tensorflow` session once the algorithm class instance is no longer needed."""
        backend.clear_session()

# # test it
# exp = Experiment()
#
# dataset_path = "C:\\Users\\Elizabeth Camilleri\\Documents\\PLT Project\\sonanciadataset_modified\\"
# o_path = dataset_path+"OpenSMILE Low-Level Descriptors\\"
# r_path = dataset_path+"Preference_Annotations\\"
# exp.load_object_data(o_path+"General_AllFeatures_DataSet(Unpruned).csv", has_ids=True, has_fnames=True)
# exp.load_rank_data(r_path+"Tension_General_Preferences.csv", has_ids=False, has_fnames=False)
#
# exp.set_normalization(list(np.arange(387)), NormalizationType.MIN_MAX)
#
# pl_algorithm = RankNet(ann_topology=[256, 128, 64, 1], batch_size=64, epochs=300)  # 72.24
#
# exp.set_pl_algorithm(pl_algorithm)
#
# pl_evaluator = KFoldCrossValidation(k=3)
# exp.set_pl_evaluator(pl_evaluator)
#
# exp.run(shuffle=True)
#
# t = time.time()
# exp.save_exp_log(t, path="my_results.csv")
#
# exp.save_model(t, fold_idx=0, path="my_model.csv")  # save model of first fold
