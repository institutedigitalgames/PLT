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
import tkinter as tk
from tkinter import ttk, font

from pyplt import ROOT_PATH
from pyplt.gui.util import colours
from pyplt.util.enums import ParamType, ActivationType


class BackpropMenu(tk.Frame):
    """GUI menu for specifying parameters of the `Backpropagation` algorithm.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent, on_resize_fn):
        """Initializes the frame widget and its contents.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter widget`
        :param on_resize_fn: the function called when the parent window is resized by the user. This is required by
            this class so that the window is resized accordingly whenever the widgets for a hidden layer are added
            to or removed from the `BackpropMenu`.
        :type on_resize_fn: function
        """
        # variables (set to default values):
        self._hidden_neurons = dict()
        self._hidden_activation_functions = dict()
        self._hidden_neurons[0] = tk.IntVar(value=5)
        self._hidden_activation_functions[0] = tk.StringVar(value=ActivationType.RELU.name)
        self._output_activation_function = tk.StringVar(value=ActivationType.RELU.name)
        self._learn_rate = tk.DoubleVar(value=0.001)
        self._error_thresh = tk.DoubleVar(value=0.001)
        self._epochs = tk.IntVar(value=10)

        self._on_resize_fn = on_resize_fn

        self._parent = parent
        tk.Frame.__init__(self, parent, bg=colours.PL_OUTER)

        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        #################
        # ANN params menu
        #################

        ann_frame = tk.Frame(self, bd=2, relief=tk.GROOVE, padx=15, pady=15, bg=colours.PL_INNER)
        ann_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        # Main title
        ann_title = tk.Label(ann_frame, text="ANN Topology Parameters", bg=colours.PL_INNER,
                             font=font.Font(family='Ebrima', size=12))
        ann_title.pack(side=tk.TOP, pady=(0, 10))

        # Hidden layers
        self._num_hidden_layers = 0
        self._gui_layer_rows = []
        self.ann_hidden_layers_frame = tk.Frame(ann_frame, bg=colours.PL_INNER)
        self.ann_hidden_layers_frame.pack()

        # Parameter titles
        tk.Label(self.ann_hidden_layers_frame, text="ANN Layer", width=20,
                 justify='center', bg=colours.PL_INNER, font=font.Font(family='Ebrima', size=10,
                                                                       weight=font.BOLD)).grid(row=0, column=0)
        tk.Label(self.ann_hidden_layers_frame, text="Neurons", width=10,
                 justify='center', bg=colours.PL_INNER, font=font.Font(family='Ebrima', size=10,
                                                                       weight=font.BOLD)).grid(row=0, column=1)
        tk.Label(self.ann_hidden_layers_frame, text="Activation Function", width=20,
                 justify='center', bg=colours.PL_INNER, font=font.Font(family='Ebrima', size=10,
                                                                       weight=font.BOLD)).grid(row=0, column=2)

        self._add_hidden_layer()  # add one hidden layer by default

        # fake labels for proper grid alignment
        fake_frame = tk.Frame(ann_frame, bg=colours.PL_INNER)
        fake_frame.pack()
        tk.Label(fake_frame, text="", width=20, justify='center', bg=colours.PL_INNER).grid(row=0, column=0)
        tk.Label(fake_frame, text="", width=10, justify='center', bg=colours.PL_INNER).grid(row=0, column=1)
        tk.Label(fake_frame, text="", width=20, justify='center', bg=colours.PL_INNER).grid(row=0, column=2)

        # Buttons to Add/Remove hidden layers
        self._add_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/add_layer_128_30_01_dark_blue.png"))
        self._remove_img = tk.PhotoImage(file=os.path.join(ROOT_PATH,
                                                           "assets/buttons/remove_layer_128_30_01_dark_blue.png"))
        ann_buttons_frame = tk.Frame(ann_frame, pady=10, bg=colours.PL_INNER)
        ann_buttons_frame.pack()
        add_btn = tk.Button(ann_buttons_frame, image=self._add_img, relief='flat', bd=0,
                            highlightbackground=colours.PL_INNER, highlightcolor=colours.PL_INNER,
                            highlightthickness=0, background=colours.PL_INNER, activebackground=colours.PL_INNER,
                            command=self._add_hidden_layer)
        add_btn.pack(side=tk.LEFT, padx=5)
        remove_btn = tk.Button(ann_buttons_frame, image=self._remove_img, relief='flat', bd=0,
                               highlightbackground=colours.PL_INNER, highlightcolor=colours.PL_INNER,
                               highlightthickness=0, background=colours.PL_INNER, activebackground=colours.PL_INNER,
                               command=self._del_hidden_layer)
        remove_btn.pack(side=tk.RIGHT, padx=5)

        # Output layer
        ann_output_layer_frame = tk.Frame(ann_frame, bg=colours.PL_INNER)
        ann_output_layer_frame.pack()

        # fake labels for proper grid alignment
        fake_l1 = tk.Label(ann_output_layer_frame, text="", width=20, justify='center', bg=colours.PL_INNER)
        fake_l1.grid(row=0, column=0)
        fake_l2 = tk.Label(ann_output_layer_frame, text="", width=10, justify='center', bg=colours.PL_INNER)
        fake_l2.grid(row=0, column=1)
        fake_l3 = tk.Label(ann_output_layer_frame, text="", width=20, justify='center', bg=colours.PL_INNER)
        fake_l3.grid(row=0, column=2)

        tk.Label(ann_output_layer_frame, text="Output Layer ", bg=colours.PL_INNER).grid(row=1, column=0)
        # num neurons
        self._o_neurons = ttk.Entry(ann_output_layer_frame, width=5, style='PL.PLT.TEntry')
        self._o_neurons.insert(0, "1")
        self._o_neurons.grid(row=1, column=1)
        self._o_neurons.configure(state='disabled')  # disabled since we will always have 1 neuron as output!
        self._o_neurons.bind("<<PLTStateToggle>>", self._check_o_neurons_entry)  # bind

        # ^ n.b. ensured that 'Run Experiment' button, OutputLayer#nuerons checkbox and Steps 2-4 of BeginnerMenu
        # are re-disabled or re-enabled accordingly on close of stacked windows (help dialog or load params).
        # solution via binding state changes to method which ensures re-disable (or re-enable if appropriate time/case).

        # activation function
        options = [ActivationType.RELU.name, ActivationType.SIGMOID.name]
        o_afn = ttk.OptionMenu(ann_output_layer_frame, self._output_activation_function, options[0], *options,
                               style='Sub.PL.PLT.TMenubutton')
        o_afn.grid(row=1, column=2)

        ################
        # BP params menu
        ################

        bp_frame = tk.Frame(self, bd=2, relief=tk.GROOVE, padx=15, pady=15, bg=colours.PL_INNER)
        bp_frame.grid(row=1, column=0, sticky='nsew')

        bp_title = tk.Label(bp_frame, text="Backpropagation Parameters", bg=colours.PL_INNER,
                            font=font.Font(family='Ebrima', size=12))
        bp_title.pack(side=tk.TOP, pady=(0, 10))

        bp_params = tk.Frame(bp_frame, bg=colours.PL_INNER)
        bp_params.pack()

        self._add_entry_param(bp_params, "Learning rate", self._learn_rate, 0, ParamType.FLOAT.name)

        bp_term_frame = tk.Frame(bp_frame, bg=colours.PL_INNER, bd=2, relief='groove', padx=50, pady=5)
        bp_term_frame.pack(pady=(10, 0))

        tk.Label(bp_term_frame, text="Termination Condition", font='Ebrima 10 bold').pack()
        bp_term_params = tk.Frame(bp_term_frame, bg=colours.PL_INNER, padx=5, pady=5)
        bp_term_params.pack()

        self._add_entry_param(bp_term_params, "Error threshold", self._error_thresh, 0, ParamType.FLOAT.name)
        self._add_entry_param(bp_term_params, "Epochs", self._epochs, 1, ParamType.INT.name)

        self.columnconfigure(0, weight=1)

    def _check_o_neurons_entry(self, event):
        """Ensure that the output neurons `tkinter.Entry` widget remains disabled when a change in its
        state is detected.

        :param event: the <<PLTStateToggle>> event that triggered the call to this method.
        :type event: `tkinter Event`
        """
        new_state = str(self._o_neurons.cget('state'))
        # print("new_state: " + str(new_state))
        if (new_state == 'disable') or (new_state == 'disabled'):
            # print("Output neurons Entry state was changed (disabled)! -- ignoring...")
            return
        # print("Output neurons Entry state was changed (activated)!")
        # ALWAYS re-disable Output neurons Entry!
        # print("Setting back to disabled - as always.")
        self._o_neurons.configure(state='disable')  # set back to disabled
        # ^ n.b. state name does not always apply - check for specific widget!

    def _add_hidden_layer(self):
        """Add a new hidden layer entry in the network topology area of the GUI menu."""
        layer = self._num_hidden_layers
        # layer name
        l_name = tk.Label(self.ann_hidden_layers_frame, text="Hidden Layer " + str(layer+1), bg=colours.PL_INNER)
        l_name.grid(row=layer+1, column=0)  # +1 to account for titles in GUI
        # num neurons
        self._hidden_neurons[layer] = tk.IntVar(value=5)  # init to default value
        l_neurons = ttk.Entry(self.ann_hidden_layers_frame, width=5, textvariable=self._hidden_neurons[layer],
                              validate="all", validatecommand=self._vcmd + (ParamType.INT.name, ),
                              style='PL.PLT.TEntry')
        l_neurons.grid(row=layer+1, column=1)  # +1 to account for titles in GUI
        self.ann_hidden_layers_frame.after_idle(lambda: l_neurons.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values
        # activation functions
        options = [ActivationType.RELU.name, ActivationType.SIGMOID.name]
        self._hidden_activation_functions[layer] = tk.StringVar(value=ActivationType.RELU.name)
        # ^ init to default value
        l_afn = ttk.OptionMenu(self.ann_hidden_layers_frame, self._hidden_activation_functions[layer], options[0],
                               *options, style='Sub.PL.PLT.TMenubutton')
        l_afn.grid(row=layer+1, column=2)  # +1 to account for titles in GUI
        self._gui_layer_rows.append([l_name, l_neurons, l_afn])
        # increment number of hidden layers
        self._num_hidden_layers += 1

        self.update_idletasks()
        self._on_resize_fn(None)

    def _del_hidden_layer(self):
        """Remove the last hidden layer entry from the network topology area of the GUI menu."""
        if self._num_hidden_layers > 0:
            layer = self._num_hidden_layers-1
            # remove gui stuff for that layer
            for element in self._gui_layer_rows[layer]:
                element.destroy()
            # remove entry in self._gui_layer_rows for that layer
            del self._gui_layer_rows[layer]
            # remove entry in self._hidden_neurons and self._hidden_activation_functions
            del self._hidden_neurons[layer]
            del self._hidden_activation_functions[layer]
            # decrement number of hidden layers
            self._num_hidden_layers -= 1

        self.update_idletasks()
        self._on_resize_fn(None)

    def _add_entry_param(self, parent, name, var, row, val_type):
        """Generic method for adding parameter labels and text entries to the GUI menu.

        The method constructs a `ttk.Entry` widget preceded by a `tkinter.Label` widget displaying the
        given parameter name.

        :param parent: the parent widget of these widgets.
        :type parent: `tkinter widget`
        :param name: the name of the parameter to be used as the label of the text entry widget.
        :type name: str
        :param var: the variable in which the parameter value entered by the user is to be stored.
        :type var: `tkinter.StringVar` or `tkinter.IntVar` or `tkinter.DoubleVar` or `tkinter.BooleanVar`
        :param row: the number of the row in the parent widget's grid layout where these widgets are to be drawn.
        :type row: int
        :param val_type: specifies the type of validation to carry out on the value/s entered by the user in the
            text entry widget for the given parameter.
        :type val_type: :class:`pyplt.util.enums.ParamType`
        """
        tk.Label(parent, text=str(name)+": ", bg=colours.PL_INNER).grid(row=row, column=0, sticky='w',
                                                                        padx=(0, 10), pady=5)
        entry = ttk.Entry(parent, width=7, textvariable=var,
                          validate="all", validatecommand=self._vcmd + (val_type,), style='PL.PLT.TEntry')
        entry.grid(row=row, column=1, pady=5)
        parent.after_idle(lambda: entry.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values

    def _on_validate(self, action, index, value_if_allowed,
                     prior_value, text, validation_type, trigger_type, widget_name, val_type):
        """Validate the text input by the user in a `ttk.Entry` widget.

        All but the last of the method arguments refer to callback substitution codes describing the type of
        change made to the text of the `ttk.Entry widget`. Their description is based on existing documentation at
        http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/entry-validation.html.

        :param action: Action code: 0 for an attempted deletion, 1 for an attempted insertion, or -1 if the
            callback was called for focus in, focus out, or a change to the textvariable.
        :param index: When the user attempts to insert or delete text, this argument will be the index of the
            beginning of the insertion or deletion. If the callback was due to focus in, focus out, or a change to
            the textvariable, the argument will be -1.
        :param value_if_allowed: The value that the text will have if the change is allowed.
        :param prior_value: The text in the entry before the change.
        :param text: If the call was due to an insertion or deletion, this argument will be the text being
            inserted or deleted.
        :param validation_type: The current value of the widget's validate option.
        :param trigger_type: The reason for this callback: one of 'focusin', 'focusout', 'key', or 'forced' if
            the textvariable was changed.
        :param widget_name: The name of the widget.
        :param val_type: specifies the type of validation to carry out on the value/s entered by the user in the
            text entry widget for the given parameter.
        :type val_type: :class:`pyplt.util.enums.ParamType`
        :return: a boolean specifying whether the text is valid (True) or not (False).
        :rtype: bool
        """
        # print("Validating... " + val_type)
        # print(prior_value)
        # print(text)
        # print(value_if_allowed)
        if ParamType[val_type] == ParamType.FLOAT:
            try:
                float(text)
                return True
            except ValueError:
                try:
                    float(value_if_allowed)
                    return True
                except ValueError:
                    return False
        else:
            if text.isnumeric():
                return True
            else:
                return False

    def get_params(self):
        """Get the values for the `Backpropagation` algorithm parameters as specified by the user via the GUI.

        The parameter values are returned in the form of a dict where the keys match the keywords of the arguments
        that would be passed to the corresponding :class:`pyplt.plalgorithms.base.PLAlgorithm` constructor.

        :return: a dict containing the values for the following parameters in order:

            * ann_topology: the topology of the neurons in the network
            * learn_rate: the learning rate
            * error_threshold: the error threshold
            * epochs: the number of epochs
            * activation_functions: the activation functions for each neuron layer in the network
        :rtype: dict (size 7)
        """
        # what we have:
        # - the number of hidden layers (default 1);
        # - a dict containing the number of neurons in each hidden layer (default {0: 5});
        # - a dict specifying the name of the activation function type used for each hidden layer
        # (default {0: ActivationType.RELU.name});
        # - the name of activation function type of the output neuron (default ActivationType.RELU.name);
        # - the learning rate (default 0.1);
        # - the error threshold (default 0.1);
        # - the number of epochs (default 10).

        # get values of dicts
        _hidden_neurons = dict()
        for h_n in self._hidden_neurons:
            _hidden_neurons[h_n] = self._hidden_neurons[h_n].get()

        _hidden_activation_functions = dict()
        for h_a in self._hidden_activation_functions:
            _hidden_activation_functions[h_a] = self._hidden_activation_functions[h_a].get()

        if self._num_hidden_layers == 0:
            ann_topology = None
        else:
            ann_topology = list(_hidden_neurons.values()) + [1]  # (1 = output layer)
        activation_fns = list(_hidden_activation_functions.values()) + [self._output_activation_function.get()]

        print("topology: " + str(ann_topology))
        print("activation_fns: " + str(activation_fns))

        # construct dict ready to be passed on as kwargs to PLAlgorithm class (BackpropagationTF):
        # ann_topology - the topology of the neurons in the network (default [5, 1])
        # learn_rate - the learning rate (default 0.1)
        # error_threshold - the error threshold (default 0.1)
        # epochs - the number of epochs (default 10)
        # activation_functions - the activation functions for each neuron layer in the network
        # ^ (default [ActivationType.RELU.name, ActivationType.RELU.name])

        params_dict = {'ann_topology': ann_topology,
                       'learn_rate': self._learn_rate.get(),
                       'error_threshold': self._error_thresh.get(),
                       'epochs': self._epochs.get(),
                       'activation_functions': activation_fns}
        return params_dict
