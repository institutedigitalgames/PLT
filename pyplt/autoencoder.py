import tensorflow as tf

from pyplt.util.enums import ActivationType


class Autoencoder:
    """Autoencoder class."""

    def __init__(self, input_size, code_size, encoder_topology, decoder_topology, activation_functions=None,
                 learn_rate=0.001, error_threshold=0.001, epochs=10, batch_size=32):
        """Initialize the Autoencoder.

        :param input_size: the number of input features that will be fed into the network. This determines the number of
            neurons in the input layer (and therefore also in the output layer).
        :type input_size: int
        :param code_size: the number of neurons in the code layer (and therefore the size of the encoding).
        :type code_size: int
        :param encoder_topology: specifies the number of neurons in each layer of the encoder part of the network,
            excluding the input layer and the code layer.
        :type encoder_topology: list of int
        :param decoder_topology: specifies the number of neurons in each layer of the decoder part of the network,
            excluding the code layer and the output layer.
        :type decoder_topology: list of int
        :param activation_functions: a list of the activation functions to be used across the neurons
            for each layer of the ANN (excluding input layer) (default None); if None, all layers will use the
            Rectified Linear Unit (ReLU) function i.e. :attr:`pyplt.util.enums.ActivationType.RELU`, except for
            the output layer which will use the Logisitic Sigmoid function i.e.
            :attr:`pyplt.util.enums.ActivationType.SIGMOID`.
        :type activation_functions: list of :class:`pyplt.util.enums.ActivationType` or None, optional
        :param learn_rate: the learning rate used in the weight update step of the algorithm (default 0.1).
        :type learn_rate: float, optional
        :param error_threshold: a threshold at or below which the error of a model is considered to be
            sufficiently trained (default 0.1).
        :type error_threshold: float, optional
        :param epochs: the maximum number of iterations the algorithm should make over the entire set of
            training examples (default 10).
        :type epochs: int, optional
        :param batch_size: number of samples per gradient update (default 32).
        :type batch_size: int, optional
        """
        self._vars_declared = False
        self._num_inputs = input_size
        self._num_output = self._num_inputs
        self._code_size = code_size
        self._ann_topology = encoder_topology + [self._code_size] + decoder_topology + [self._num_output]
        self._code_layer_idx = len(encoder_topology)
        self._learn_rate = learn_rate
        self._error_threshold = error_threshold
        self._num_epochs = epochs
        self._batch_size = batch_size
        self._training_examples = None
        self._graph = None
        self._session = None
        self._X = None
        self._encoding = None
        self._output_layer = None
        self._loss = None
        self._optimiser = None
        self._init_op = None

        # activation functions
        self._activation_fns = []
        self._activation_fn_names = []

        # convert activation function enums to actual tensorflow functions
        # but keep ActivationType names (in self._activation_fn_names) for more readable printing
        if activation_functions is None:
            for layer in range(len(self._ann_topology)):  # for each layer
                if layer == len(self._ann_topology)-1:  # for output layer, use sigmoid
                    self._activation_fns.append(tf.nn.sigmoid)
                    self._activation_fn_names.append(ActivationType.SIGMOID.name)
                else:
                    self._activation_fns.append(tf.nn.relu)
                    self._activation_fn_names.append(ActivationType.RELU.name)
        else:
            for act_fn in activation_functions:
                if (act_fn == ActivationType.SIGMOID) or (act_fn == ActivationType.SIGMOID.name):  # enum or enum name
                    self._activation_fns.append(tf.nn.sigmoid)
                    self._activation_fn_names.append(ActivationType.SIGMOID.name)
                elif (act_fn == ActivationType.RELU) or (act_fn == ActivationType.RELU.name):
                    self._activation_fns.append(tf.nn.relu)
                    self._activation_fn_names.append(ActivationType.RELU.name)
                elif (act_fn == ActivationType.LINEAR) or (act_fn == ActivationType.LINEAR.name):
                    self._activation_fns.append(None)
                    self._activation_fn_names.append(ActivationType.LINEAR.name)
                # TODO: do same for new activation functions

        print("AUTOENCODER activation functions:")
        print(self._activation_fns)

    def init_train(self, progress_window=None, exec_stopper=None):
        """Initialize the model (topology).

        This method is to be called if one wishes to initialize the model (topology) explicitly.
        This is done by declaring `tensorflow` placeholders, variables, and operations. If not called explicitly, the
        :meth:`train()` method will call it once implicitly.

        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        """
        # SET UP TENSORFLOW STUFF

        # start a new graph and session (delete any previous tf.Variables, etc.)
        self._graph = tf.Graph()  # reset the graph - equivalent to tf.reset_default_graph()
        # first, close any existing session, if applicable
        # unless already handled by Experiment.run()
        if self._session is not None:
            still_open = not self._session._closed
            print("(autoencoder) Is tf.Session still open? " + str(still_open))
            if still_open:
                self.clean_up()
        print("(autoencoder) Opening tf.Session...")
        self._session = tf.Session(graph=self._graph)
        print("(autoencoder) Done.")

        print("Initializing autoencoder.")
        if progress_window is not None:
            progress_window.put("Initializing autoencoder.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting autoencoder execution during init...")
            return

        with self._graph.as_default():
            # input
            self._X = tf.placeholder(tf.float32, shape=[None, self._num_inputs])

            # set up parameters
            W = []  # weights
            b = []  # biases
            outs = []

            initializer = tf.contrib.layers.xavier_initializer()

            # now declare the weights connecting the input to the next (and possibly last) layer
            W.append(tf.Variable(initializer([self._num_inputs, self._ann_topology[0]]), name='W0'))
            b.append(tf.Variable(tf.zeros([self._ann_topology[0]]), name='b0'))

            # and the weights connecting the hidden layers to each other (but the last is connected to the output layer)
            for layer in range(1, len(self._ann_topology)):
                n_in = self._ann_topology[layer - 1]  # incoming neurons
                n_out = self._ann_topology[layer]  # outgoing neurons
                W.append(tf.Variable(initializer([n_in, n_out]), name='W' + str(layer)))
                b.append(tf.Variable(tf.zeros([n_out]), name='b' + str(layer)))

            # ^ we initialise the values of the weights using the Xavier initializer
            # and the values of the biases to zero

            # (there's always L-1 number of weights/bias tensors, where L is the number of
            # layers (L incl. input & output))

            # now setup node inputs and activation functions

            # calculate the outputs of the first (and possibly last) layer
            if len(self._ann_topology) == 1:
                label = 'OUT'
            else:
                label = 'L0Out'
            summ = tf.add(tf.matmul(self._X, W[0]), b[0])  # i.e. sum(xi*wi)+b
            if self._activation_fns[0] is None:
                outs.append(summ)  # if activation function None, use linear
            else:
                outs.append(self._activation_fns[0](summ, name=label))  # i.e. activation fn

            # calculate the outputs of any subsequent layers (the last being the output layer)
            for layer in range(1, len(self._ann_topology)):
                if layer == (len(self._ann_topology) - 1):
                    label = 'OUT'
                else:
                    label = 'L' + str(layer) + 'Out'
                if self._activation_fns[layer] is None:  # if activation function None, use linear
                    outs.append(tf.add(tf.matmul(outs[layer - 1], W[layer]), b[layer]))
                else:
                    outs.append(self._activation_fns[layer](tf.add(tf.matmul(outs[layer - 1], W[layer]),
                                                                   b[layer]), name=label))
                # ^ last out shape = (n_samples x n_input)

            # get code layer output (encoding)
            self._encoding = outs[self._code_layer_idx]

            # Also include a cost/loss function for the optimisation/backpropagation to work on
            # Here we use mean squared error (MSE)
            self._output_layer = outs[-1]
            self._loss = tf.reduce_mean(tf.square(self._output_layer-self._X))

            # now setup an optimiser
            self._optimiser = tf.train.AdamOptimizer(  # GradientDescentOptimizer
                learning_rate=self._learn_rate).minimize(self._loss)

            # finally setup the initialisation operator
            self._init_op = tf.global_variables_initializer()

            if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted after init
                # abort execution!
                print("Aborting autoencoder execution during init...")
                return

        self._vars_declared = True
        print("Autoencoder initialization complete.")

    def train(self, training_examples, progress_window=None, exec_stopper=None):
        """Train the autoencoder.

        :param training_examples: the input examples used to train the network.
        :type training_examples: array-like of shape n_samples x n_features
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * training loss -- if execution is completed successfully.
            * None -- if experiment is aborted before completion by `exec_stopper`.
        """
        self._training_examples = training_examples

        train_loss = float('inf')  # infinity

        print("Starting autoencoder training.")
        if progress_window is not None:
            progress_window.put("Starting autoencoder training.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting autoencoder execution...")
            return

        if not self._vars_declared:
            # run first time training only (unless already forced externally)
            self.init_train(progress_window=progress_window, exec_stopper=exec_stopper)
            if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted after init
                # abort execution!
                print("Aborting autoencoder execution during/after init...")
                return

        with self._graph.as_default():
            sess = self._session
            # graph = tf.get_default_graph()

            sess.run(self._init_op)  # initialise the variables
            for epoch in range(self._num_epochs):
                if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
                    # abort execution!
                    print("Aborting autoencoder execution...")
                    return

                # for tensorboard
                # writer = tf.summary.FileWriter("tensorboard-output", sess.graph)

                # sess.run(self._optimiser, feed_dict={self._X: self._training_examples})
                # train_loss = self._loss.eval(feed_dict={self._X: self._training_examples}, session=sess)

                batch_size = self._batch_size
                # or do in batches...
                # print("num samples: " + str(len(training_examples)))
                num_batches = len(training_examples) // batch_size
                # print("num_batches: " + str(num_batches))
                if num_batches == 0:  # if dataset is smaller than batch_size, use 1 batch (whole dataset)
                    num_batches = 1
                for iteration in range(num_batches):
                    if iteration == num_batches-1:
                        # last iteration
                        X_batch = self._training_examples[iteration*batch_size:]
                    else:
                        X_batch = self._training_examples[iteration*batch_size:(iteration+1)*batch_size]
                    sess.run(self._optimiser, feed_dict={self._X: X_batch})

                train_loss = self._loss.eval(feed_dict={self._X: self._training_examples}, session=sess)

                # for tensorboard
                # writer.close()

                if train_loss <= self._error_threshold:
                    print("Reached loss below or equal to error threshold. Training terminated.")
                    break

                print("AUTOENCODER epoch {} loss {}".format(epoch, train_loss))

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting autoencoder execution...")
            return

        print("Autoencoder training complete.")
        if progress_window is not None:
            progress_window.put("Autoencoder training complete.")

        return train_loss

    def predict(self, samples):
        """Run the given samples through the entire autoencoder, thus obtaining a reconstruction of the input.

        :param samples: the samples to be input into the autoencoder.
        :type samples: array-like of shape n_samples x n_features
        :return: the autoencoder output for the given input samples.
        :rtype: array-like of shape n_samples x n_features
        """
        with self._graph.as_default():
            sess = self._session

            # run it though just the encoder
            output = sess.run(self._output_layer, feed_dict={self._X: samples})
            # output = self._output_layer.eval(feed_dict={self._X: samples}, session=sess)

        # TODO: check for abort?

        return output

    def encode(self, samples, progress_window=None, exec_stopper=None):
        """Encode the given samples by running the given samples through the encoder part of the network.

        :param samples: the samples to be input into the autoencoder.
        :type samples: array-like of shape n_samples x n_features
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the encoded sample -- if execution is completed successfully.
            * None -- if experiment is aborted before completion by `exec_stopper`.
        :rtype: array-like of shape n_samples x code_size
        """
        if progress_window is not None:
            progress_window.put("Encoding dataset.")

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting autoencoder execution...")
            return

        with self._graph.as_default():
            sess = self._session

            # run it though just the encoder
            encoded_samples = sess.run(self._encoding, feed_dict={self._X: samples})

        if (exec_stopper is not None) and (exec_stopper.stopped()):  # check if experiment was aborted
            # abort execution!
            print("Aborting autoencoder execution...")
            return

        if progress_window is not None:
            progress_window.put("Dataset encoding complete.")

        return encoded_samples

    def get_code_size(self):
        """Get the value of the code size parameter.

        :return: the code size value.
        :rtype: int
        """
        return self._code_size

    def get_topology(self):
        """Get the topology of the network.

        :return: the topology.
        :rtype: list of int
        """
        return self._ann_topology

    def get_topology_incl_input(self):
        """Get the topology of the network including the input layer.

        :return: the topology including the input layer.
        :rtype: list of int
        """
        return [self._num_inputs] + self._ann_topology

    def get_actfs(self):
        """Get the names of the activation functions of each layer in the network.

        :return: the names of the activation functions of each layer.
        :rtype: list of str
        """
        return self._activation_fn_names

    def get_learn_rate(self):
        """Get the value of the learning rate parameter.

        :return: the learning rate value.
        :rtype: float
        """
        return self._learn_rate

    def get_error_thresh(self):
        """Get the value of the error threshold parameter.

        :return: the error threshold value.
        :rtype: float
        """
        return self._error_threshold

    def get_epochs(self):
        """Get the value of the epochs parameter.

        :return: the epochs value.
        :rtype: int
        """
        return self._num_epochs

    def clean_up(self):
        """Close the `tensorflow` session once the algorithm class instance is no longer needed.

        V.IMP. THIS FUNCTION MUST BE CALLED WHEN THE CLASS INSTANCE IS NO LONGER IN USE unless a context manager
        is used around the Autoencoder class instance!!!
        """
        # v.imp to close tf.Session whether manually or via context manager (see documentation)!!
        if self._session is not None:
            print("Closing tf.Session...")
            self._session.close()
            print("Done.")

    # makes the class require a context manager like tensorflow so that session is always closed at the end:
    # if using context manager do something similar to
    # with Autoencoder() as autoencoder:
    #     autoencoder.train()
    # as per https://stackoverflow.com/questions/43975090/tensorflow-close-session-on-object-destruction
    # TODO: add documentation to explain this ^ for developers / API users

    def __enter__(self):
        return self

    def __exit__(self, exc_type):
        self.clean_up()

# train_count = 35
# BATCH_SIZE = 32
#
# i=0
# for start, end in zip(range(0, train_count, BATCH_SIZE),
#                       range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
#     print("batch " + str(i))
#     print("using samples " + str(start) + ":" + str(end))
#     i += 1
