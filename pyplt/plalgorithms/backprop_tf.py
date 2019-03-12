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
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from pyplt import ROOT_PATH
from pyplt.util.enums import PLAlgo, ActivationType
from pyplt.plalgorithms.base import PLAlgorithm


class BackpropagationTF(PLAlgorithm):
    """Backpropagation algorithm implemented with the `tensorflow` package.

    This is a gradient-descent algorithm that iteratively (over a given number of epochs)
    optimizes an error function by adjusting the weights of an artificial neural network (ANN)
    model proportionally to the gradient of the error with respect to the current value of
    the weights and current data samples. The proportion and therefore the strength of each
    update is regulated by the given learning rate. The error function used is the Rank Margin
    function which for a given pair of data samples (A and B, with A preferred over B)
    yields 0 if the network output for A (fA) is more than one unit larger than the network
    output for B (fB) and 1.0-((fA)-(fB)) otherwise. The total error is averaged over the
    complete set of pairs in the training set. If the error is below a given threshold, training
    stops before reaching the specified number of epochs, and the current weight values are
    returned as the final model. In PLT, the algorithm was implemented using the `tensorflow` library.
    """

    def __init__(self, ann_topology=None, learn_rate=0.001, error_threshold=0.001, epochs=10,
                 activation_functions=None, batch_size=32, debug=False):
        """Initializes the BackpropagationTF object.

        :param ann_topology: a list indicating the topology of the artificial neural network (ANN) to be used with the
            algorithm. The list contains the number of neurons in each layer of the ANN, excludes the input layer but
            including the output layer (must always be 1 neuron in size); a value of None is equivalent to [1]
            indicating an ANN with no hidden layers and only an output layer (consisting of 1 neuron) (default None).
        :type ann_topology: list or None, optional
        :param learn_rate: the learning rate used in the weight update step of the Backpropagation algorithm
            (default 0.001).
        :type learn_rate: float, optional
        :param error_threshold: a threshold at or below which the error of a model is considered to be
            sufficiently trained (default 0.001).
        :type error_threshold: float, optional
        :param epochs: the maximum number of iterations the algorithm should make over the entire pairwise rank
            training set (default 10).
        :type epochs: int, optional
        :param activation_functions: a list of the activation function to be used across the neurons
            for each layer of the ANN; if None, all layers will use the Rectified Linear Unit (ReLU) function i.e.
            :attr:`pyplt.plalgorithms.backprop_tf.ActivationType.RELU` (default None).
        :type activation_functions: list of :class:`pyplt.plalgorithms.backprop_tf.ActivationType` or None, optional
        :param batch_size: number of samples per gradient update (default 32).
        :type batch_size: int, optional
        :param debug: specifies whether or not to print notes to console for debugging (default False).
        :type debug: bool, optional
        """
        desc = "This is a gradient-descent algorithm that iteratively (over a given number of epochs) " \
               "optimizes an error function by adjusting the weights of an artificial neural network (ANN) " \
               "model proportionally to the gradient of the error with respect to the current value of " \
               "the weights and current data samples. The proportion and therefore the strength of each " \
               "update is regulated by the given learning rate. The error function used is the Rank Margin " \
               "function which for a given pair of data samples (A and B, with A preferred over B) " \
               "yields 0 if the network output for A (fA) is more than one unit larger than the network " \
               "output for B (fB) and 1.0-((fA)-(fB)) otherwise. The total error is averaged over the " \
               "complete set of pairs in the training set. If the error is below a given threshold, training " \
               "stops before reaching the specified number of epochs, and the current weight values are " \
               "returned as the final model. In PLT, the algorithm was implemented using the tensorflow library."
        self._n_feats = None
        self._n_out = 1
        if ann_topology is None:
            self._ann_topology = [self._n_out]  # excludes input layer; output layer is always 1 neuron in size
        else:
            self._ann_topology = ann_topology
        self._learn_rate = learn_rate
        self._error_threshold = error_threshold
        self._epochs = epochs
        self._batch_size = batch_size

        # convert activation function enums to actual tensorflow functions
        # but keep ActivationType names (in activation_fn_names) for more readable printing
        self._activation_fns = []
        activation_fn_names = []
        if activation_functions is None:
            for _ in self._ann_topology:  # for each layer
                self._activation_fns.append(tf.nn.relu)
                activation_fn_names.append(ActivationType.RELU.name)
        else:
            for act_fn in activation_functions:
                if (act_fn == ActivationType.SIGMOID) or (act_fn == ActivationType.SIGMOID.name):  # enum or enum name
                    self._activation_fns.append(tf.nn.sigmoid)
                    activation_fn_names.append(ActivationType.SIGMOID.name)
                elif (act_fn == ActivationType.RELU) or (act_fn == ActivationType.RELU.name):  # enum or enum name
                    self._activation_fns.append(tf.nn.relu)
                    activation_fn_names.append(ActivationType.RELU.name)
                elif (act_fn == ActivationType.LINEAR) or (act_fn == ActivationType.LINEAR.name):
                    self._activation_fns.append(None)
                # TODO: do same for new activation functions

        self._vars_declared = False
        self._init_op = None
        self._optimiser = None
        self._total_loss = None
        self._pref_x = None
        self._non_x = None
        self._saver = None
        self._graph = None
        self._session = None

        super().__init__(description=desc, debug=debug, name=PLAlgo.BACKPROPAGATION.name,
                         ann_topology=self._ann_topology, activation_functions=activation_fn_names,
                         learn_rate=self._learn_rate, error_threshold=self._error_threshold, epochs=self._epochs)

    def init_train(self, n_features):
        """Initialize the model (topology).

        This method is to be called if one wishes to initialize the model (topology) explicitly.
        This is done by declaring `tensorflow` placeholders, variables, and operations. This may be used, for example,
        to use the same :class:`BackpropagationTF` object but simply modify its topology while evaluating different
        feature sets during wrapper-type feature selection processes. If not called explicitly, the
        :meth:`train()` method will call it once implicitly.

        :param n_features: the number of features to be used during the training process.
        :type n_features: int
        """
        # print("called init_train()")

        # start with a whole new graph (delete any previous tf.Variables, etc.)
        self._graph = tf.Graph()  # reset the graph - equivalent to tf.reset_default_graph()
        # first, close any existing session, if applicable
        # unless already handled by Experiment.run()
        if self._session is not None:
            still_open = not self._session._closed
            print("Is tf.Session still open? " + str(still_open))
            if still_open:
                self.clean_up()
        print("Opening tf.Session...")
        self._session = tf.Session(graph=self._graph)
        print("Done.")

        with self._graph.as_default():
            print("INITIALIZING MODEL TOPOLOGY")

            # start training...

            self._n_feats = n_features

            # declare the training data placeholders
            # input x - number of features of objects
            # TODO: see if make input dtype dynamic based on data set e.g. integer or nominal data....?
            self._pref_x = tf.placeholder(tf.float32, [None, self._n_feats],
                                          name="pref_x")  # placeholder for preferred objects
            self._non_x = tf.placeholder(tf.float32, [None, self._n_feats],
                                         name="non_x")  # placeholder for non-preferred objects

            # set up parameters
            W = []  # weights
            b = []  # biases
            pref_outs = []
            non_outs = []

            # now declare the weights connecting the input to the next (and possibly last) layer
            W.append(tf.Variable(tf.random_normal([self._n_feats, self._ann_topology[0]], stddev=0.03), name='W0'))
            b.append(tf.Variable(tf.random_normal([self._ann_topology[0]]), name='b0'))

            # and the weights connecting the hidden layers to each other (but the last is connected to the output layer)
            for layer in range(1, len(self._ann_topology)):
                n_in = self._ann_topology[layer - 1]  # incoming neurons
                n_out = self._ann_topology[layer]  # outgoing neurons
                W.append(tf.Variable(tf.random_normal([n_in, n_out], stddev=0.03), name='W' + str(layer)))
                b.append(tf.Variable(tf.random_normal([n_out]), name='b' + str(layer)))

            # ^ we initialise the values of the weights using a
            # random normal distribution with a mean of zero and a standard deviation of 0.03

            # (there's always L-1 number of weights/bias tensors, where L is the number of
            # layers (L incl. input & output))

            # now setup node inputs and activation functions

            # calculate the outputs of the first (and possibly last) layer
            if len(self._ann_topology) == 1:
                suffix = 'OUT'
            else:
                suffix = 'L0Out'
            if self._activation_fns[0] is None:  # if activation function None, use linear
                summ = tf.add(tf.matmul(self._pref_x, W[0]), b[0], name='Pref' + suffix)  # i.e. sum(xi*wi)+b
                pref_outs.append(summ)  # i.e. linear activation
                summ = tf.add(tf.matmul(self._non_x, W[0]), b[0], name='Non' + suffix)  # i.e. sum(xi*wi)+b
                non_outs.append(summ)  # i.e. linear activation
            else:
                summ = tf.add(tf.matmul(self._pref_x, W[0]), b[0])  # i.e. sum(xi*wi)+b
                pref_outs.append(self._activation_fns[0](summ, name='Pref' + suffix))  # i.e. activation fn
                summ = tf.add(tf.matmul(self._non_x, W[0]), b[0])  # i.e. sum(xi*wi)+b
                non_outs.append(self._activation_fns[0](summ, name='Non' + suffix))  # i.e. activation fn

            # calculate the outputs of any subsequent layers (the last being the output layer)
            for layer in range(1, len(self._ann_topology)):
                if layer == (len(self._ann_topology) - 1):
                    suffix = 'OUT'
                else:
                    suffix = 'L' + str(layer) + 'Out'
                if self._activation_fns[layer] is None:  # if activation function None, use linear
                    pref_outs.append(tf.add(tf.matmul(pref_outs[layer - 1], W[layer]), b[layer], name='Pref' + suffix))
                    # ^ last pref_out shape = (r x 1)
                    non_outs.append(tf.add(tf.matmul(non_outs[layer - 1], W[layer]), b[layer], name='Non' + suffix))
                    # ^ last non_out shape = (r x 1)
                else:
                    pref_outs.append(self._activation_fns[layer](tf.add(tf.matmul(pref_outs[layer - 1], W[layer]),
                                                                        b[layer]), name='Pref' + suffix))
                    # ^ last pref_out shape = (r x 1)
                    non_outs.append(self._activation_fns[layer](tf.add(tf.matmul(non_outs[layer - 1], W[layer]),
                                                                       b[layer]), name='Non' + suffix))
                    # ^ last non_out shape = (r x 1)

            # Also include a cost/loss function for the optimisation/backpropagation to work on
            # Here we use error/loss = max(1 - (predict_pref - predict_other), 0)
            # (since we have pairwise ranks, the loss function is based on the outputs of 2 objects, not just 1!)
            pref_y = pref_outs[-1]
            non_y = non_outs[-1]

            one = tf.constant(1.0)
            zero = tf.constant(0.0)
            _loss = tf.maximum(tf.subtract(one, (tf.subtract(pref_y, non_y))), zero,
                               name="individualLoss")  # individual loss
            self._total_loss = tf.reduce_mean(_loss, name='totalLoss')  # total loss i.e. sum(individual losses)/n_ranks
            # or sum/n_ranks*2 as in Java PLT:
            # self._total_loss = tf.divide(tf.reduce_sum(_loss), tf.to_float(tf.multiply(tf.shape(self._pref_x)[0], 2)),
            #  name='totalLoss')  # total loss i.e. sum(individual losses)/n_ranks
            # but results should theoretically be (and seem to be) the same

            # now setup an optimiser
            self._optimiser = tf.train.AdamOptimizer(
                learning_rate=self._learn_rate).minimize(self._total_loss)

            # finally setup the initialisation operator
            self._init_op = tf.global_variables_initializer()

        self._vars_declared = True

    def train(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """Run a `tensorflow` session to infer an ANN model using the given training data.

        The given pairwise rank data is split into a set of preferred objects and a set of non-preferred objects, which
        are then fed into the ANN. The resulting (predicted) model output of each object in a given rank pair is
        compared to the actual preference and the error is calculated via a Rank Margin error function. The algorithm
        attempts to optimize the average error over the entire set of ranks across several iterations (epochs)
        until it reaches the maximum number number of iterations (epochs) or reaches the error threshold.

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
        print("already initialized?" + str(self._vars_declared))

        print("Starting training with Backpropagation.")
        if progress_window is not None:
            progress_window.put("Starting training with Backpropagation.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Backpropagation execution...")
            return

        if use_feats is None:
            train_objects_ = train_objects.copy()
        else:
            train_objects_ = train_objects.loc[:, use_feats].copy()
        train_ranks_ = train_ranks.copy()

        # start training...

        self._n_feats = len(train_objects_.columns)

        print("REFORMATTING DATA FOR TENSORFLOW...")

        # reformat ranks for tensorflow - split into a set of preferred objects and a set of non-preferred objects
        prefs_x = train_objects_.loc[list(train_ranks_.iloc[:, 0])].values
        nons_x = train_objects_.loc[list(train_ranks_.iloc[:, 1])].values
        # ^ assigned to prefs_x and non_x here directly since DataFrame.values is already a numpy (nd)array!

        # print(prefs_x.shape)
        # print(nons_x.shape)

        if not self._vars_declared:
            # run first time training only (unless already forced externally)
            self.init_train(self._n_feats)

        # FINALLY...
        print("ACTUALLY STARTING TRAINING...")

        # start the session
        loss_curve = []
        with self._graph.as_default():
            sess = self._session

            # initialise the variables
            sess.run(self._init_op)  # initialize the values of the Variables, etc.! (i.e. random for W&b)

            if self._debug:
                # print weights !!!!!!!!!!!
                W = []
                b = []
                for layer in range(len(self._ann_topology)):
                    w = self._graph.get_tensor_by_name("W"+str(layer) + ":0").eval(session=sess)
                    bias = self._graph.get_tensor_by_name("b"+str(layer)+":0").eval(session=sess)
                    W.append(w)
                    b.append(bias)
                print("Weights before training...")
                print(W)
                print("Biases before training...")
                print(b)

            print("variables have been (re)initialized.")
            for epoch in range(self._epochs):
                if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
                    # abort execution!
                    print("Aborting Backpropagation execution...")
                    return

                # actual training
                # _, loss_ = sess.run([self._optimiser, self._total_loss],
                #                     feed_dict={self._pref_x: prefs_x, self._non_x: nons_x})
                # loss_curve.append(loss_)

                batch_size = self._batch_size
                # do in batches...
                # print("num samples: " + str(prefs_x.shape[0]))
                num_batches = prefs_x.shape[0] // batch_size
                # print("num_batches: " + str(num_batches))
                if num_batches == 0:  # if dataset is smaller than batch_size, use 1 batch (whole dataset)
                    num_batches = 1
                for iteration in range(num_batches):
                    # print("batch " + str(iteration))
                    if iteration == num_batches - 1:
                        # last iteration
                        prefs_x_batch = prefs_x[iteration * batch_size:]
                        nons_x_batch = nons_x[iteration * batch_size:]
                        # print("using samples " + str(iteration * batch_size) + ":")
                    else:
                        prefs_x_batch = prefs_x[iteration * batch_size:(iteration + 1) * batch_size]
                        nons_x_batch = nons_x[iteration * batch_size:(iteration + 1) * batch_size]
                        # print("using samples "+str(iteration * batch_size) + ":" + str((iteration + 1) * batch_size))
                    # actual training
                    sess.run([self._optimiser, self._total_loss],
                             feed_dict={self._pref_x: prefs_x_batch, self._non_x: nons_x_batch})

                train_loss = self._total_loss.eval(feed_dict={self._pref_x: prefs_x,
                                                              self._non_x: nons_x}, session=sess)

                loss_curve.append(train_loss)
                if self._debug:
                    print("Epoch:", (epoch + 1), "loss =", str(train_loss).format("{:.3f}"))
                if train_loss <= self._error_threshold:
                    print("Reached loss below or equal to error threshold. Training terminated.")
                    break

                print("epoch {} loss {}".format(epoch, train_loss))

            if self._debug:
                # print weights !!!!!!!!!!!
                W = []
                b = []
                for layer in range(len(self._ann_topology)):
                    w = self._graph.get_tensor_by_name("W"+str(layer) + ":0").eval(session=sess)
                    bias = self._graph.get_tensor_by_name("b"+str(layer)+":0").eval(session=sess)
                    W.append(w)
                    b.append(bias)
                print("Weights after training...")
                print(W)
                print("Biases after training...")
                print(b)

            if self._debug:
                plt.plot(loss_curve)
                plt.ylabel("Loss")
                plt.xlabel("Iterations")
                plt.xticks(range(self._epochs))
                plt.show()

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Backpropagation execution...")
            return

        print("Training complete.")
        if progress_window is not None:
            progress_window.put("Training complete.")

        return True

    def calc_train_accuracy(self, train_objects, train_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to calculating the training accuracy of the learned model.

        This method is implemented explicitly for this algorithm since this approach is substantially more efficient
        for algorithms using the `tensorflow` package than the calc_train_accuracy() method of
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
        print("Calculating TRAINING accuracy...")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Backpropagation execution...")
            return

        if use_feats is None:
            train_objects_ = train_objects.copy()
        else:
            train_objects_ = train_objects.loc[:, use_feats].copy()
        train_ranks_ = train_ranks.copy()

        self._n_feats = len(train_objects_.columns)  # 18

        print("REFORMATTING DATA FOR TENSORFLOW (again)...")
        # reformat ranks for tensorflow - split into a set of preferred objects and a set of non-preferred objects
        prefs_x = train_objects_.loc[list(train_ranks_.iloc[:, 0])].values
        nons_x = train_objects_.loc[list(train_ranks_.iloc[:, 1])].values
        # ^ assigned to prefs_x and non_x here directly since DataFrame.values is already a numpy (nd)array!

        print("ACTUALLY STARTING CALCULATION...")

        # Now, launch the model, restore the variables, and do some work with the model.
        with self._graph.as_default():
            sess = self._session
            graph = tf.get_default_graph()

            # Now, let's access and create placeholders variables and create feed-dict to feed new data
            # (access saved variable/Tensor/placeholders)
            pref_x = graph.get_tensor_by_name("pref_x:0")
            non_x = graph.get_tensor_by_name("non_x:0")

            # Now, access the op that we want to run
            pref_y_ = graph.get_tensor_by_name("PrefOUT:0")
            non_y_ = graph.get_tensor_by_name("NonOUT:0")

            # define an accuracy assessment operation
            # in our case, pref output should be > non-pref output
            correct_prediction = tf.greater(pref_y_, non_y_,
                                            name='isCorrectPredict')
            accuracy = tf.scalar_mul(100, tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="TrainAccuracy"))
            # ^ count how many were correctly predicted (True) (equivalent to below:)
            # accuracy returns the (n_Trues / (n_Trues + n_Falses)) * 100

            acc = sess.run(accuracy, feed_dict={pref_x: prefs_x, non_x: nons_x})

            print("Performance (TRAINING): " + str(acc))
            self._train_accuracy = acc

            if progress_window is not None:
                progress_window.put("Training accuracy: " + str(acc))

            if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
                # abort execution!
                print("Aborting Backpropagation execution...")
                return

            return acc

    def predict(self, input_object, progress_window=None, exec_stopper=None):
        """Predict the output of a given input object by running it through the learned model.

        :param input_object: the input data corresponding to a single object.
        :type input_object: one row from a `pandas.DataFrame`
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return: the predicted output resulting from running the learned model using the given input.
        :rtype: float
        """
        # Now, launch the model, restore the variables, and do some work with the model.
        with self._graph.as_default():
            sess = self._session
            graph = tf.get_default_graph()

            # Now, let's access and create placeholders variables and create feed-dict to feed new data
            # (access saved variable/Tensor/placeholders)
            # just need one input->output variable set not necessarily pref or non...
            x = graph.get_tensor_by_name("pref_x:0")

            # Now, access the op that we want to run
            y_ = graph.get_tensor_by_name("PrefOUT:0")

            out = sess.run(y_, feed_dict={x: input_object})
            # Check the values of the variables

            return out[0]

    def test(self, objects, test_ranks, use_feats=None, progress_window=None, exec_stopper=None):
        """An algorithm-specific approach to testing/validating the model using the given test data.

        This method is implemented explicitly for this algorithm since this approach is substantially more efficient
        for algorithms using the `tensorflow` package than the test() method of the base class
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
        if self._debug:
            print("Calculating TEST accuracy...")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting Backpropagation test execution...")
            return

        if use_feats is None:
            objects_ = objects.copy()
        else:
            objects_ = objects.loc[:, use_feats].copy()
        test_ranks_ = test_ranks.copy()

        self._n_feats = len(objects_.columns)

        prefs_x = objects_.loc[list(test_ranks_.iloc[:, 0])].values
        nons_x = objects_.loc[list(test_ranks_.iloc[:, 1])].values
        # ^ assigned to prefs_x and non_x here directly since DataFrame.values is already a numpy (nd)array!

        # Now, launch the model, restore the variables, and do some work with the model.
        with self._graph.as_default():
            sess = self._session
            graph = tf.get_default_graph()

            # Now, let's access and create placeholders variables and create feed-dict to feed new data
            # (access saved variable/Tensor/placeholders)
            pref_x = graph.get_tensor_by_name("pref_x:0")
            non_x = graph.get_tensor_by_name("non_x:0")

            # Now, access the op that we want to run
            pref_y_ = graph.get_tensor_by_name("PrefOUT:0")
            non_y_ = graph.get_tensor_by_name("NonOUT:0")

            # define an accuracy assessment operation
            # in our case, pref output should be > non-pref output
            correct_prediction = tf.greater(pref_y_, non_y_,
                                            name='isCorrectPredict_test')
            accuracy = tf.scalar_mul(100, tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="TestAccuracy"))
            # ^ count how many were correctly predicted (True) (equivalent to below:)
            # accuracy returns the (n_Trues / (n_Trues + n_Falses)) * 100

            acc = sess.run(accuracy, feed_dict={pref_x: prefs_x, non_x: nons_x})

            if self._debug:
                print("Performance (TEST): " + str(accuracy) + "%")

            if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
                # abort execution!
                print("Aborting Backpropagation test execution...")
                return

            return acc

    @staticmethod
    def transform_data(object_):
        """Transform an object into the format required by this particular algorithm implementation.

        In this case, nothing changes.

        :param object_: the object to be transformed.
        :type object_: one row from a `pandas.DataFrame`
        :return: the transformed object in the form of an array.
        :rtype: `numpy.ndarray`
        """
        # do nothing
        transformed_obj = np.asarray([list(object_)])
        # print(transformed_obj)
        return transformed_obj

    def save_model(self, timestamp, path="", suppress=False):
        """Save the ANN model to a Comma Separated Value (CSV) file at the path indicated by the user.

        Optionally, the file creation may be suppressed and a `pandas.DataFrame` representation of the model
        returned instead.

        The file/DataFrame stores the weights, biases, and activation functions of each neuron in each layer of the ANN.
        Each row represents these values for a neuron in a layer, starting from the first neuron in the first hidden
        layer (if applicable), and moving forward neuron-by-neuron, layer-by-layer, until the output neuron is reached.
        The number of columns is variable as the file stores enough columns to represent the maximum number of weights
        across all neurons in the network.

        Weights columns are labeled with the letter 'w' followed by the index of the incoming neuron from which
        the given weight is connected the current neuron. Hidden layers in the 'layer' column are labelled with the
        letter 'h' followed by the index of the layer. The output layer is simply labelled as 'OUTPUT'.

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
        # similar to get_params_string()
        # load model as in predict(), then just save (export) model to file

        # Now, launch the model, restore the variables, and do some work with the model.
        with self._graph.as_default():
            sess = self._session
            graph = tf.get_default_graph()

            # Now, let's access and create placeholders variables and create feed-dict to feed new data
            # (access saved variable/Tensor/placeholders)
            W = []
            b = []
            for layer in range(len(self._ann_topology)):
                w = graph.get_tensor_by_name("W"+str(layer) + ":0").eval(session=sess)
                bias = graph.get_tensor_by_name("b"+str(layer)+":0").eval(session=sess)
                W.append(w)
                b.append(bias)

        # Now, format model for saving to file
        if path == "":
            path = os.path.join(ROOT_PATH, "logs\\model_" + str(timestamp) + ".csv")

        # print(W)
        # print(b)

        n_rows = 0
        cols = ["Layer", "Neuron"]
        max_layer_size = 0
        for layer in range(len(self._ann_topology)):
            n = W[layer].shape[0]  # i.e. n_in for that layer # self._ann_topology[layer]
            n_rows += W[layer].shape[1]  # i.e. n_out for that layer
            if n > max_layer_size:
                max_layer_size = n
        cols.extend(["w" + str(w) for w in range(max_layer_size)])
        cols.append("bias")
        cols.append("activation_fn")

        # print(cols)
        model_arr = np.empty(shape=[n_rows, len(cols)], dtype=object)

        n_layers = len(self._ann_topology)
        l_curr = 0
        row = 0
        for layer in W:
            if l_curr == n_layers - 1:
                name = "OUTPUT"
            else:
                name = "h" + str(l_curr + 1)
            n_in = layer.shape[0]
            # print("n_in " + str(n_in))
            n_out = layer.shape[1]
            # print("n_out " + str(n_out))
            biases = b[l_curr]
            activation_fns = self._activation_fns[l_curr]
            model_arr[row:row + n_out, 0] = name
            model_arr[row:row + n_out, 1] = range(n_out)
            # 1. store weights for each neuron in the layer
            model_arr[row:row + n_out, 2:2 + n_in] = layer.T
            # 2. store bias for each neuron in the layer
            model_arr[row:row + n_out, -2] = biases
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

    def load_model(self):
        """Load a model which was trained using this algorithm.  # TODO: to be actually implemented.
        """
        # TODO: a parsing version of get_params_string()
        # TODO: use Phil's script!
        # parse what is saved via save_model() and set values in self._algo!
        # user can then use predict() and/or train() using this loaded model
        return

    def clean_up(self):
        """Close the `tensorflow` session once the algorithm class instance is no longer needed.

        V.IMP. THIS FUNCTION MUST BE CALLED WHEN THE CLASS INSTANCE IS NO LONGER IN USE unless a context manager
        is used around the BackpropagationTF class instance!!!
        """
        # v.imp to close tf.Session whether manually or via context manager (see documentation)!!
        if self._session is not None:
            print("Closing tf.Session...")
            self._session.close()
            print("Done.")

    # makes the class require a context manager like tensorflow so that session is always closed at the end:
    # if using context manager do something similar to
    # with BackpropagationTF() as ann_bp:
    #     ann_bp.train()
    # as per https://stackoverflow.com/questions/43975090/tensorflow-close-session-on-object-destruction
    # TODO: add documentation to explain this ^ for developers / API users

    def __enter__(self):
        return self

    def __exit__(self, exc_type):
        self.clean_up()
