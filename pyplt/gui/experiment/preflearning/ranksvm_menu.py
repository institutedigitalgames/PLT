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

import tkinter as tk
from tkinter import ttk, font

from pyplt.gui.util import colours
from pyplt.util.enums import ParamType, KernelType


class RankSVMMenu(tk.Frame):
    """GUI menu for specifying parameters of the `RankSVM` algorithm.

    Extends the class `tkinter.Frame`.
    """

    def __init__(self, parent):
        """Initializes the frame widget and its contents.

        :param parent: the parent widget of this frame widget.
        :type parent: `tkinter widget`
        """
        # variables (set to default values):
        self._kernel = tk.StringVar(value=KernelType.LINEAR.name)
        self._gamma = tk.DoubleVar(value=1)
        self._degree = tk.DoubleVar(value=2)

        self._parent = parent
        tk.Frame.__init__(self, parent, bg=colours.PL_OUTER)

        # for validation:
        self._vcmd = (self.register(self._on_validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        #################
        # ANN params menu
        #################

        rsvm_frame = tk.Frame(self, bd=2, relief=tk.GROOVE, padx=15, pady=15, bg=colours.PL_INNER)
        rsvm_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        # Main title
        title = tk.Label(rsvm_frame, text="Rank SVM Parameters", bg=colours.PL_INNER,
                         font=font.Font(family='Ebrima', size=12))
        title.pack(side=tk.TOP, pady=(0, 10))

        inner_frame = tk.Frame(rsvm_frame, bg=colours.PL_INNER)
        inner_frame.pack()

        # kernel (pack after creating gamma & degree widgets)
        kernel_frame = tk.Frame(inner_frame, bg=colours.PL_INNER)
        kernel_frame.pack(pady=(0, 5))
        tk.Label(kernel_frame, text="Kernel: ", bg=colours.PL_INNER).grid(row=0, column=0, padx=(0, 5))
        options = [KernelType.LINEAR.name, KernelType.RBF.name, KernelType.POLY.name]
        self._kernel_menu = ttk.OptionMenu(kernel_frame, self._kernel, options[0],
                                           *options, style='Sub.PL.PLT.TMenubutton',
                                           command=lambda _: self._show_kernel_params())

        self._kernel_params_frame = tk.Frame(inner_frame)
        self._kernel_params_frame.pack()
        # gamma & degree (but don't pack!)
        self._gamma_gui = self._add_entry_param(self._kernel_params_frame, "\u03b3 ", self._gamma, 0,
                                                ParamType.FLOAT.name)
        self._degree_gui = self._add_entry_param(self._kernel_params_frame, "Degree", self._degree, 1,
                                                 ParamType.FLOAT.name)

        self._kernel_menu.grid(row=0, column=1)

        self.columnconfigure(0, weight=1)

    def _show_kernel_params(self):
        """Display the appropriate kernel parameter widgets according to the kernel chosen by the user."""
        # print("displaying the correct kernel parameters...")
        if self._kernel.get() == KernelType.RBF.name:
            self._degree_gui[0].grid_forget()  # remove degree label
            self._degree_gui[1].grid_forget()  # remove degree entry
            self._gamma_gui[0].grid(row=0, column=0, sticky='e', padx=(0, 10), pady=5)  # include gamma label
            self._gamma_gui[1].grid(row=0, column=1, pady=5, sticky='e')  # include gamma entry
        elif self._kernel.get() == KernelType.POLY.name:
            self._gamma_gui[0].grid(row=0, column=0, sticky='e', padx=(0, 10), pady=5)  # include gamma label
            self._gamma_gui[1].grid(row=0, column=1, pady=5, sticky='e')  # include gamma entry
            self._degree_gui[0].grid(row=1, column=0, sticky='e', padx=(0, 10), pady=5)  # include degree label
            self._degree_gui[1].grid(row=1, column=1, pady=5, sticky='e')  # include degree entry
            self._kernel_params_frame.columnconfigure(0, weight=1)
            self._kernel_params_frame.columnconfigure(1, weight=1)
        else:
            self._degree_gui[0].grid_forget()  # remove degree label
            self._degree_gui[1].grid_forget()  # remove degree entry
            self._gamma_gui[0].grid_forget()  # remove gamma label
            self._gamma_gui[1].grid_forget()  # remove gamma entry

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
        label = tk.Label(parent, text=str(name)+": ", bg=colours.PL_INNER)
        entry = ttk.Entry(parent, width=7, textvariable=var,
                         validate="all", validatecommand=self._vcmd + (val_type,), style='PL.PLT.TEntry')
        parent.after_idle(lambda: entry.config(validate='all'))
        # ^ re-enables validation after using .set() with the Vars to initialize them with default values
        return label, entry

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
        """Get the values for the `RankSVM` algorithm parameters as specified by the user via the GUI.

        The parameter values are returned in the form of a dict where the keys match the keywords of the arguments
        that would be passed to the corresponding :class:`pyplt.plalgorithms.base.PLAlgorithm` constructor.

        :return: a dict containing the values for the following parameters in order:

            * kernel: the kernel name
            * gamma: the gamma kernel parameter value
            * degree: the degree kernel parameter value
        :rtype: dict (size 3)
        """
        kernel = KernelType[str(self._kernel.get())]
        gamma = float(self._gamma.get())
        degree = float(self._degree.get())

        # construct dict ready to be passed on as kwargs to PLAlgorithm class (RankSVM):
        # - the kernel name (default RBF i.e. 1);
        # - the gamma kernel parameter value (default 1.0);
        # - the degree kernel parameter value (default 2.0).
        params_dict = {'kernel': kernel,
                       'gamma': gamma,
                       'degree': degree}
        return params_dict
