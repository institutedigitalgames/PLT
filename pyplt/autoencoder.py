import tensorflow as tf

from pyplt.util.enums import ActivationType


class Autoencoder:
    """Autoencoder class."""

    def __init__(self, input_size, code_size, encoder_topology, decoder_topology, activation_functions=None,
                 learn_rate=0.001, error_threshold=0.001, epochs=10):
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
        :param activation_functions: a list of the activation function to be used across the neurons
            for each layer of the ANN (excluding input layer); if None, all layers will use the Rectified Linear Unit
            (ReLU) function i.e. :attr:`pyplt.plalgorithms.backprop_tf.ActivationType.RELU` (default None).
        :type activation_functions: list of :class:`pyplt.plalgorithms.backprop_tf.ActivationType` or None, optional
        :param learn_rate: the learning rate used in the weight update step of the algorithm (default 0.1).
        :type learn_rate: float, optional
        :param error_threshold: a threshold at or below which the error of a model is considered to be
            sufficiently trained (default 0.1).
        :type error_threshold: float, optional
        :param epochs: the maximum number of iterations the algorithm should make over the entire set of
            training examples (default 10).
        :type epochs: int, optional
        """
        self._num_inputs = input_size
        self._num_output = self._num_inputs
        self._code_size = code_size
        self._ann_topology = encoder_topology + [self._code_size] + decoder_topology + [self._num_output]
        self._code_layer_idx = len(encoder_topology)
        self._learn_rate = learn_rate
        self._error_threshold = error_threshold
        self._num_epochs = epochs
        self._training_examples = None

        # activation functions
        self._activation_fns = []
        activation_fn_names = []

        # convert activation function enums to actual tensorflow functions
        # but keep ActivationType names (in activation_fn_names) for more readable printing
        if activation_functions is None:
            for _ in self._ann_topology:  # for each layer
                self._activation_fns.append(tf.nn.relu)
        else:
            for act_fn in activation_functions:
                if (act_fn == ActivationType.SIGMOID) or (act_fn == ActivationType.SIGMOID.name):  # enum or enum name
                    self._activation_fns.append(tf.nn.sigmoid)
                    activation_fn_names.append(ActivationType.SIGMOID.name)
                elif (act_fn == ActivationType.RELU) or (act_fn == ActivationType.RELU.name):
                    self._activation_fns.append(tf.nn.relu)
                    activation_fn_names.append(ActivationType.RELU.name)
                # TODO: do same for new activation functions

        print("AUTOENCODER activation functions:")
        print(self._activation_fns)

        # SET UP TENSORFLOW STUFF

        # start a new graph and session
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

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
            outs.append(self._activation_fns[0](summ, name=label))  # i.e. activation fn

            # calculate the outputs of any subsequent layers (the last being the output layer)
            for layer in range(1, len(self._ann_topology)):
                if layer == (len(self._ann_topology) - 1):
                    label = 'OUT'
                else:
                    label = 'L' + str(layer) + 'Out'
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

    def train(self, training_examples):
        """Train the autoencoder.

        :param training_examples: the input examples used to train the network.
        :type training_examples: array-like of shape n_samples x n_features
        """
        self._training_examples = training_examples

        with self._graph.as_default():
            sess = self._session
            # graph = tf.get_default_graph()

            sess.run(self._init_op)  # initialise the variables
            for epoch in range(self._num_epochs):

                # for tensorboard
                # writer = tf.summary.FileWriter("tensorboard-output", sess.graph)

                # sess.run(self._optimiser, feed_dict={self._X: self._training_examples})
                # train_loss = self._loss.eval(feed_dict={self._X: self._training_examples}, session=sess)

                # for tensorboard
                # writer.close()

                batch_size = 500
                # or do in batches...
                num_batches = len(training_examples) // batch_size
                for iteration in range(num_batches):
                    if iteration == num_batches-1:
                        # last iteration
                        X_batch = self._training_examples[iteration*batch_size:]
                    else:
                        X_batch = self._training_examples[iteration*batch_size:(iteration+1)*batch_size]
                    sess.run(self._optimiser, feed_dict={self._X: X_batch})

                train_loss = self._loss.eval(feed_dict={self._X: self._training_examples}, session=sess)

                if train_loss <= self._error_threshold:
                    print("Reached loss below or equal to error threshold. Training terminated.")
                    break

                print("AUTOENCODER epoch {} loss {}".format(epoch, train_loss))

    def predict(self, samples):
        """Run the given samples through the entire autoencoder.

        :param samples:
        :return: network output
        """
        # TODO: [AUTOENCODER] add docstring
        with self._graph.as_default():
            sess = self._session

            # run it though just the encoder
            output = sess.run(self._output_layer, feed_dict={self._X: samples})
            # output = self._output_layer.eval(feed_dict={self._X: samples}, session=sess)

        return output

    def encode(self, samples):
        """Encode the given samples by running the given samples through the encoder part of the network.

        :param samples:
        :return: the encoded sample
        """
        # TODO: [AUTOENCODER] add docstring
        with self._graph.as_default():
            sess = self._session

            # run it though just the encoder
            encoded_samples = sess.run(self._encoding, feed_dict={self._X: samples})

        return encoded_samples

    def get_code_size(self):
        """Get the value of the code size parameter.

        :return: the code size value.
        :rtype: int
        """
        return self._code_size

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
    # with Autoencoder() as autoencoder:
    #     autoencoder.train()
    # as per https://stackoverflow.com/questions/43975090/tensorflow-close-session-on-object-destruction
    # TODO: add documentation to explain this ^ for developers / API users

    def __enter__(self):
        return self

    def __exit__(self, exc_type):
        self.clean_up()
