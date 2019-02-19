import os
import tkinter as tk
from tkinter import ttk, font

# from ttkthemes import ThemedStyle

from pyplt import ROOT_PATH
from pyplt.gui.util import colours  # , styles
from pyplt.plalgorithms.backprop_tf import ActivationType
from pyplt.util.enums import ParamType


class AutoencoderSettings(tk.Frame):
    """Tkinter Frame widget for setting up an autoencoder and its parameter values."""

    def __init__(self, parent, input_size, on_resize_fn):
        self._parent = parent
        tk.Frame.__init__(self, self._parent)

        # SET UP VARIABLES (set to default values):

        # - input variables
        self._input_size = input_size

        # - encoder variables
        self._encoder_hidden_neurons = dict()
        self._encoder_hidden_activation_functions = dict()
        # self._hidden_neurons[0] = tk.IntVar(value=5)
        # self._hidden_activation_functions[0] = tk.StringVar(value=ActivationType.SIGMOID.name)
        self._num_encoder_hidden_layers = 0
        self._gui_encoder_layer_rows = []

        # - code layer variables
        self._code_size = tk.IntVar()  # no default value bc depends on input size...
        self._code_activation_function = tk.StringVar(value=ActivationType.SIGMOID.name)

        # - decoder variables
        self._decoder_hidden_neurons = dict()
        self._decoder_hidden_activation_functions = dict()
        self._num_decoder_hidden_layers = 0
        self._gui_decoder_layer_rows = []

        # - output layer variables
        self._output_size = self._input_size
        self._output_activation_function = tk.StringVar(value=ActivationType.SIGMOID.name)

        # - total / general variables
        self._decoder_labels = dict()
        self._num_total_hidden_layers = 0
        self._actf_options = [ActivationType.SIGMOID.name]

        # - backpropagation variables
        self._learn_rate = tk.DoubleVar(value=0.1)
        self._error_thresh = tk.DoubleVar(value=0.1)
        self._epochs = tk.IntVar(value=10)

        # - presentation variables
        self._label_width = 13
        self._neurons_padx = 65
        sections_padx = 25

        self._on_resize_fn = on_resize_fn

        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        #################
        # ANN params menu
        #################

        ann_frame = tk.Frame(self, bd=2, relief=tk.GROOVE, padx=15, pady=15, bg=colours.PL_INNER)
        ann_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        # Main title
        ann_title = tk.Label(ann_frame, text="Autoencoder ANN Topology Parameters", bg=colours.PL_INNER,
                             font=font.Font(family='Ebrima', size=12))
        ann_title.pack(side=tk.TOP)

        # Global Parameter titles
        topology_titles = tk.Frame(ann_frame, bg=colours.PL_INNER)
        topology_titles.pack(pady=10, padx=sections_padx)
        # also add a canvas to fill up space so that code layer frame is aligned with encoder, code and decoder frames!
        code_title_canvas = tk.Canvas(topology_titles, bg=colours.PL_INNER, width=15, height=5)
        code_title_canvas.pack(side=tk.LEFT, padx=2)
        topology_titles_sub_frame = tk.Frame(topology_titles)
        topology_titles_sub_frame.pack()
        tk.Label(topology_titles_sub_frame, text="ANN Layer", width=20,
                 justify='center', bg=colours.PL_INNER, font=font.Font(family='Ebrima', size=11,
                                                                       weight=font.BOLD)).grid(row=0, column=0)
        tk.Label(topology_titles_sub_frame, text="Neurons", width=10,
                 justify='center', bg=colours.PL_INNER, font=font.Font(family='Ebrima', size=11,
                                                                       weight=font.BOLD)).grid(row=0, column=1)
        tk.Label(topology_titles_sub_frame, text="Activation Function", width=20,
                 justify='center', bg=colours.PL_INNER, font=font.Font(family='Ebrima', size=11,
                                                                       weight=font.BOLD)).grid(row=0, column=2)

        # ---------------------------------------- ENCODER ----------------------------------------

        encoder_sub_menu = tk.Frame(ann_frame, bg=colours.PL_INNER)
        encoder_sub_menu.pack(fill=tk.X, expand=True, padx=sections_padx)

        # Encoder Subtitle
        encoder_title_canvas = tk.Canvas(encoder_sub_menu, bg=colours.PL_INNER, width=15, height=65)
        encoder_title_canvas.create_text((0, 65), text="ENCODER",
                                         font=font.Font(family='Ebrima', size=10), anchor=tk.NW, angle=90)
        encoder_title_canvas.pack(side=tk.LEFT, padx=2)

        encoder_frame = tk.Frame(encoder_sub_menu, bd=2, relief=tk.GROOVE, bg=colours.PL_INNER, padx=10, pady=10)
        encoder_frame.pack(fill=tk.X, expand=True)

        # Input layer
        ann_input_layer_frame = tk.Frame(encoder_frame, bg=colours.PL_INNER)
        ann_input_layer_frame.pack()

        tk.Label(ann_input_layer_frame, text="Input Layer", bg=colours.PL_INNER,
                 width=self._label_width, anchor=tk.W).grid(row=0, column=0)
        # num neurons
        self._i_neurons = ttk.Entry(ann_input_layer_frame, width=5, style='PL.PLT.TEntry')
        self._i_neurons.insert(0, str(self._output_size))
        self._i_neurons.grid(row=0, column=1, padx=self._neurons_padx)
        self._i_neurons.configure(state='disabled')  # disabled since we will always have 1 neuron as output!
        self._i_neurons.bind("<<PLTStateToggle>>", self._check_i_neurons_entry)  # bind
        # ^ n.b. ensured that 'Run Experiment' button, OutputLayer#nuerons checkbox and Steps 2-4 of BeginnerMenu
        # are re-disabled or re-enabled accordingly on close of stacked windows (help dialog or load params).
        # solution via binding state changes to method which ensures re-disable (or re-enable if appropriate time/case).
        # finally, add carefully-designed fake third column to align with the other hidden layers
        tk.Label(ann_input_layer_frame, bg=colours.PL_INNER, width=12, anchor=tk.E).grid(row=0, column=2, padx=(0, 3))

        # Hidden layers
        self._encoder_hidden_layers_frame = tk.Frame(encoder_frame, bg=colours.PL_INNER)
        self._encoder_hidden_layers_frame.pack()

        self._add_hidden_layer('encoder')  # add one hidden layer by default

        # Buttons to Add/Remove hidden layers
        self._add_img = tk.PhotoImage(file=os.path.join(ROOT_PATH, "assets/buttons/add_layer_128_30_01_dark_blue.png"))
        self._remove_img = tk.PhotoImage(file=os.path.join(ROOT_PATH,
                                                           "assets/buttons/remove_layer_128_30_01_dark_blue.png"))
        encoder_buttons_frame = tk.Frame(encoder_frame, pady=10, bg=colours.PL_INNER)
        encoder_buttons_frame.pack()
        encoder_add_btn = tk.Button(encoder_buttons_frame, image=self._add_img, relief='flat', bd=0,
                                    highlightbackground=colours.PL_INNER, highlightcolor=colours.PL_INNER,
                                    highlightthickness=0, background=colours.PL_INNER, activebackground=colours.PL_INNER,
                                    command=lambda s='encoder': self._add_hidden_layer(s))
        encoder_add_btn.pack(side=tk.LEFT, padx=5)
        encoder_remove_btn = tk.Button(encoder_buttons_frame, image=self._remove_img, relief='flat', bd=0,
                                       highlightbackground=colours.PL_INNER, highlightcolor=colours.PL_INNER,
                                       highlightthickness=0, background=colours.PL_INNER, activebackground=colours.PL_INNER,
                                       command=lambda s='encoder': self._del_hidden_layer(s))
        encoder_remove_btn.pack(side=tk.RIGHT, padx=5)

        # ---------------------------------------- CODE LAYER ----------------------------------------

        code_sub_menu = tk.Frame(ann_frame, bg=colours.PL_INNER)
        code_sub_menu.pack(fill=tk.X, expand=True, padx=sections_padx)

        # also add a canvas to fill up space so that code layer frame is aligned with encoder and decoder frames!
        code_title_canvas = tk.Canvas(code_sub_menu, bg=colours.PL_INNER, width=15, height=65)
        code_title_canvas.pack(side=tk.LEFT, padx=2)

        # self._code_layer_frame = tk.Frame(ann_frame, bd=2, relief=tk.GROOVE, bg=colours.PL_INNER)
        code_layer_frame = tk.Frame(code_sub_menu, bg=colours.PL_INNER, padx=10, pady=10)  # , relief=tk.GROOVE, bd=2
        code_layer_frame.pack(fill=tk.X, expand=True)

        # create another sub frame for alignment with encoder and decoder frames menus
        code_layer_sub_frame = tk.Frame(code_layer_frame, bg=colours.PL_INNER)
        code_layer_sub_frame.pack()

        tk.Label(code_layer_sub_frame, text="CODE LAYER", bg=colours.PL_INNER,  # font='Ebrima 10 bold',
                 width=self._label_width, anchor=tk.W).grid(row=0, column=0)
        # num neurons
        c_neurons = ttk.Entry(code_layer_sub_frame, width=5,
                              textvariable=self._code_size,
                              validate="all", validatecommand=self._vcmd + (ParamType.INT.name, ),
                              style='PL.PLT.TEntry')
        c_neurons.grid(row=0, column=1, padx=self._neurons_padx)

        # activation function
        c_afn = ttk.OptionMenu(code_layer_sub_frame, self._code_activation_function,
                               self._actf_options[0], *self._actf_options, style='Sub.PL.PLT.TMenubutton')
        c_afn.grid(row=0, column=2)

        # ---------------------------------------- DECODER ----------------------------------------

        decoder_sub_menu = tk.Frame(ann_frame, bg=colours.PL_INNER)
        decoder_sub_menu.pack(fill=tk.X, expand=True, padx=sections_padx)

        # Decoder Subtitle
        decoder_title_canvas = tk.Canvas(decoder_sub_menu, bg=colours.PL_INNER, width=15, height=65)
        decoder_title_canvas.create_text((0, 65), text="DECODER",
                                         font=font.Font(family='Ebrima', size=10), anchor=tk.NW, angle=90)
        decoder_title_canvas.pack(side=tk.LEFT, padx=2)

        decoder_frame = tk.Frame(decoder_sub_menu, bd=2, relief=tk.GROOVE, bg=colours.PL_INNER, padx=10, pady=10)
        decoder_frame.pack(fill=tk.X, expand=True)

        # Hidden layers
        self._decoder_hidden_layers_frame = tk.Frame(decoder_frame, bg=colours.PL_INNER)
        self._decoder_hidden_layers_frame.pack()

        self._add_hidden_layer('decoder')  # add one hidden layer by default

        # Buttons to Add/Remove hidden layers
        decoder_buttons_frame = tk.Frame(decoder_frame, pady=10, bg=colours.PL_INNER)
        decoder_buttons_frame.pack()
        decoder_add_btn = tk.Button(decoder_buttons_frame, image=self._add_img, relief='flat', bd=0,
                                    highlightbackground=colours.PL_INNER, highlightcolor=colours.PL_INNER,
                                    highlightthickness=0, background=colours.PL_INNER, activebackground=colours.PL_INNER,
                                    command=lambda s='decoder': self._add_hidden_layer(s))
        decoder_add_btn.pack(side=tk.LEFT, padx=5)
        decoder_remove_btn = tk.Button(decoder_buttons_frame, image=self._remove_img, relief='flat', bd=0,
                                       highlightbackground=colours.PL_INNER, highlightcolor=colours.PL_INNER,
                                       highlightthickness=0, background=colours.PL_INNER, activebackground=colours.PL_INNER,
                                       command=lambda s='decoder': self._del_hidden_layer(s))
        decoder_remove_btn.pack(side=tk.RIGHT, padx=5)

        # Output layer
        ann_output_layer_frame = tk.Frame(decoder_frame, bg=colours.PL_INNER)
        ann_output_layer_frame.pack()

        tk.Label(ann_output_layer_frame, text="Output Layer", bg=colours.PL_INNER,
                 width=self._label_width, anchor=tk.W).grid(row=1, column=0)
        # num neurons
        self._o_neurons = ttk.Entry(ann_output_layer_frame, width=5, style='PL.PLT.TEntry')
        self._o_neurons.insert(0, str(self._output_size))
        self._o_neurons.grid(row=1, column=1, padx=self._neurons_padx)
        self._o_neurons.configure(state='disabled')  # disabled since we will always have 1 neuron as output!
        self._o_neurons.bind("<<PLTStateToggle>>", self._check_o_neurons_entry)  # bind

        # ^ n.b. ensured that 'Run Experiment' button, OutputLayer#nuerons checkbox and Steps 2-4 of BeginnerMenu
        # are re-disabled or re-enabled accordingly on close of stacked windows (help dialog or load params).
        # solution via binding state changes to method which ensures re-disable (or re-enable if appropriate time/case).

        # activation function
        o_afn = ttk.OptionMenu(ann_output_layer_frame, self._output_activation_function,
                               self._actf_options[0], *self._actf_options,
                               style='Sub.PL.PLT.TMenubutton')
        o_afn.grid(row=1, column=2)

        ################
        # BP params menu
        ################

        bp_frame = tk.Frame(self, bd=2, relief=tk.GROOVE, padx=15, pady=15, bg=colours.PL_INNER)
        bp_frame.grid(row=1, column=0, sticky='nsew')

        bp_title = tk.Label(bp_frame, text="Autoencoder Backpropagation Parameters", bg=colours.PL_INNER,
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

    def _check_i_neurons_entry(self, event):
        """Ensure that the input neurons `tkinter.Entry` widget remains disabled when a change in its
        state is detected.

        :param event: the <<PLTStateToggle>> event that triggered the call to this method.
        :type event: `tkinter Event`
        """
        new_state = str(self._i_neurons.cget('state'))
        # print("new_state: " + str(new_state))
        if (new_state == 'disable') or (new_state == 'disabled'):
            # print("Input neurons Entry state was changed (disabled)! -- ignoring...")
            return
        # print("Input neurons Entry state was changed (activated)!")
        # ALWAYS re-disable Input neurons Entry!
        # print("Setting back to disabled - as always.")
        self._i_neurons.configure(state='disable')  # set back to disabled
        # ^ n.b. state name does not always apply - check for specific widget!

    def _add_hidden_layer(self, part):
        """Add a new hidden layer entry in the network topology area of the GUI menu.

        :param part: specifies which part of the network (encoder or decoder) to add the layer to.
        :type part: 'encoder' or 'decoder'
        """
        if part == 'encoder':
            num_local_hidden_layers = self._num_encoder_hidden_layers
            hidden_neurons = self._encoder_hidden_neurons
            hidden_activation_functions = self._encoder_hidden_activation_functions
            frame = self._encoder_hidden_layers_frame
            gui_layers = self._gui_encoder_layer_rows
            grid_increment = 1  # to make up for Input Layer in grid layout !
        else:  # i.e. decoder
            num_local_hidden_layers = self._num_decoder_hidden_layers
            hidden_neurons = self._decoder_hidden_neurons
            hidden_activation_functions = self._decoder_hidden_activation_functions
            frame = self._decoder_hidden_layers_frame
            gui_layers = self._gui_decoder_layer_rows
            grid_increment = 0

        layer_local = num_local_hidden_layers
        # layer name
        label = "Hidden Layer " + str(layer_local+1)
        # if the layer was added to the decoder, add new decoder label StringVar and use that instead
        if part == 'decoder':
            label = "Hidden Layer " + str(self._num_encoder_hidden_layers + layer_local + 2)
            # ^ the 2 includes code layer and new layer
            self._decoder_labels[num_local_hidden_layers] = tk.StringVar(value=label)
            l_name = tk.Label(frame, textvariable=self._decoder_labels[num_local_hidden_layers],
                              bg=colours.PL_INNER, width=self._label_width, anchor=tk.W)
        else:
            l_name = tk.Label(frame, text=label, bg=colours.PL_INNER, width=self._label_width, anchor=tk.W)
        # ^ the +1 is for label only
        l_name.grid(row=layer_local+grid_increment, column=0)  # +1 to account for titles in GUI
        # num neurons
        hidden_neurons[layer_local] = tk.IntVar(value=5)  # init to default value
        l_neurons = ttk.Entry(frame, width=5, textvariable=hidden_neurons[layer_local],
                              validate="all", validatecommand=self._vcmd + (ParamType.INT.name, ),
                              style='PL.PLT.TEntry')
        l_neurons.grid(row=layer_local+grid_increment, column=1, padx=self._neurons_padx)
        # ^ +1 to account for titles in GUI
        frame.after_idle(lambda: l_neurons.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values
        # activation functions
        hidden_activation_functions[layer_local] = tk.StringVar(value=ActivationType.SIGMOID.name)
        # ^ init to default value
        l_afn = ttk.OptionMenu(frame, hidden_activation_functions[layer_local], self._actf_options[0],
                               *self._actf_options, style='Sub.PL.PLT.TMenubutton')
        l_afn.grid(row=layer_local+grid_increment, column=2)  # +1 to account for titles in GUI
        gui_layers.append([l_name, l_neurons, l_afn])

        # finally, increment number of hidden layers...
        if part == 'encoder':
            self._num_encoder_hidden_layers += 1
            # and if the layer was added to the encoder, update the decoder labels
            self._update_labels()
        else:
            self._num_decoder_hidden_layers += 1
        self._num_total_hidden_layers += 1

        self.update_idletasks()
        if self._on_resize_fn is not None:
            self._on_resize_fn(None)

    def _del_hidden_layer(self, part):
        """Remove the last hidden layer entry from the network topology area of the GUI menu.

        :param part: specifies which part of the network (encoder or decoder) to remove the layer from.
        :type part: 'encoder' or 'decoder'
        """
        if part == 'encoder':
            num_local_hidden_layers = self._num_encoder_hidden_layers
            hidden_neurons = self._encoder_hidden_neurons
            hidden_activation_functions = self._encoder_hidden_activation_functions
            # frame = self._encoder_hidden_layers_frame
            gui_layers = self._gui_encoder_layer_rows
        else:
            num_local_hidden_layers = self._num_decoder_hidden_layers
            hidden_neurons = self._decoder_hidden_neurons
            hidden_activation_functions = self._decoder_hidden_activation_functions
            # frame = self._decoder_hidden_layers_frame
            gui_layers = self._gui_decoder_layer_rows

        if num_local_hidden_layers > 0:
            layer_local = num_local_hidden_layers - 1
            # remove gui stuff for that layer
            for element in gui_layers[layer_local]:
                element.destroy()
            # remove entry in self._gui_layer_rows for that layer
            del gui_layers[layer_local]
            # remove entry in self._hidden_neurons and self._hidden_activation_functions
            del hidden_neurons[layer_local]
            del hidden_activation_functions[layer_local]

            # if the layer was removed from the decoder, delete the decoder label
            if part == 'decoder':
                del self._decoder_labels[layer_local]

            # finally, decrement number of hidden layers
            if part == 'encoder':
                self._num_encoder_hidden_layers -= 1
                # and if the layer was removed from the encoder, update the decoder labels
                self._update_labels()
            else:
                self._num_decoder_hidden_layers -= 1
            self._num_total_hidden_layers -= 1

        self.update_idletasks()
        if self._on_resize_fn is not None:
            self._on_resize_fn(None)

    def _update_labels(self):
        for position, label in self._decoder_labels.items():
            new_pos = self._num_encoder_hidden_layers + position + 2  # the 2 includes code layer and starting from 0
            label.set("Hidden Layer " + str(new_pos))

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

    def print_topology(self):
        ehn = self._encoder_hidden_neurons
        eaf = self._encoder_hidden_activation_functions
        dhn = self._decoder_hidden_neurons
        daf = self._decoder_hidden_activation_functions
        for layer in range(len(ehn)):
            print("ENCODER LAYER " + str(layer) + ": " + str(ehn[layer].get()) + " neurons; "
                  + str(eaf[layer].get()) + "activation.")
        for layer in range(len(dhn)):
            print("DECODER LAYER " + str(layer) + ": " + str(dhn[layer].get()) + " neurons; "
                  + str(daf[layer].get()) + "activation.")


# test it...
# root = tk.Tk()
# root.style = ThemedStyle(root)  # Like other Tkinter classes, a Style can take a master argument
# root.style.set_theme("arc")
#
# root.title("Preference Learning Toolbox")
# img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
# root.tk.call('wm', 'iconphoto', root._w, img)
#
# print("PLT  Copyright (C) 2018  Institute of Digital Games, University of Malta \n" +
#       "This program comes with ABSOLUTELY NO WARRANTY. \n" +
#       "This is free software, and you are welcome to redistribute it \n" +
#       "under certain conditions. \n" +
#       "For more details see the GNU General Public License v3 which may be \n" +
#       "downloaded at http://plt.institutedigitalgames.com/download.php.")
#
# # set default font to Ebrima
# default_font = font.nametofont("TkDefaultFont")
# default_font.configure(family='Ebrima', size=10, weight=font.NORMAL)  # i.e., ebrima_small
# root.option_add("*Font", default_font)
#
# # set default widget background to colours.BACKGROUND
# root.tk_setPalette(background=colours.BACKGROUND)
#
# # configure ttk styles
# styles.configure_styles(root.style)
#
# gui = AutoencoderSettings(root, 784, None)
# gui.pack()
#
# root.mainloop()
