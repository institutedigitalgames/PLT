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

"""This module contains a number of classes specifying the content for each of the help/about dialogs throughout PLT."""

import os
import tkinter as tk
from tkinter import font
from tkinter import ttk
import webbrowser

from pyplt import ROOT_PATH
from pyplt.gui.util import windowstacking as ws, colours


class HelpDialog(tk.Toplevel):
    """Base class for help dialog windows used to assist the user throughout the GUI."""

    def __init__(self, parent_window):
        """Initializes the window widget.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        self._parent_window = parent_window

        tk.Toplevel.__init__(self, self._parent_window)
        img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
        self.tk.call('wm', 'iconphoto', self._w, img)
        ws.disable_parent(self, self._parent_window, ws.Mode.WITH_CLOSE)
        ws.stack_window(self, self._parent_window)
        self.title("Help")

        self.main_frame = tk.Frame(self, bg=colours.HELP_BACKGROUND)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # set up scrollbar skeleton
        self._v_scroll = ttk.Scrollbar(self.main_frame, orient="vertical", style="Help.PLT.Vertical.TScrollbar")

        # define fonts
        self.ebrima_h1 = font.Font(family='Ebrima', size=12, weight=font.BOLD)
        self.ebrima_h2 = font.Font(family='Ebrima', size=10, weight=font.BOLD)
        self.ebrima_bold = font.Font(family='Ebrima', size=10, weight=font.BOLD)

    def _show_link_cursor(self, event, text_widget):
        """Display the hand cursor over the given text widget.

        :param event: the event that triggered this method to be called.
        :type event: `tkinter Event`
        :param text_widget: the Text widget over which the cursor change is to be applied.
        :type text_widget: `tkinter.Text`
        """
        text_widget.config(cursor='hand2')

    def _show_normal_cursor(self, event, text_widget):
        """Display the normal (arrow) cursor over given text widget.

        :param event: the event that triggered this method to be called.
        :type event: `tkinter Event`
        :param text_widget: the Text widget over which the cursor change is to be applied.
        :type text_widget: `tkinter.Text`
        """
        text_widget.config(cursor='arrow')

    def _open_link(self, event, link):
        """Open the given hyperlink in the user's web browser.

        :param event: the event that triggered this method to be called.
        :type event: `tkinter Event`
        :param link: the text specifying the hyperlink to be opened.
        :param link: str
        """
        webbrowser.open(str(link))


class AboutBox(HelpDialog):
    """'About' window containing text on the details of the PLT software and its license.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the :class:`pyplt.gui.mainmenu.MainMenu` window.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)
        self.title("About")  # override 'Help' title

        # override blue background
        self.main_frame.config(bg=colours.BACKGROUND)

        tk.Label(self.main_frame, text="About Preference Learning Toolbox (PLT)", font=self.ebrima_h1).pack()

        text = "PLT  Copyright (C) 2018  Institute of Digital Games, University of Malta \n" \
               "This program comes with ABSOLUTELY NO WARRANTY. \n" \
               "This is free software, and you are welcome to redistribute it " \
               "under certain conditions. \n" \
               "For more details see the GNU General Public License v3 which may be " \
               "downloaded at "

        license_link = r"http://plt.institutedigitalgames.com/download.php"

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=5)
        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, text)
        text_area.insert(tk.END, license_link, "link")
        text_area.insert(tk.END, ".")

        # make hyperlinks actual hyperlinks
        text_area.tag_config("link", foreground='blue', underline=1, font='Consolas 10 normal')  # , cursor='hand1'
        text_area.tag_bind("link", '<Enter>', lambda _, tw=text_area: self._show_link_cursor(_, tw))
        text_area.tag_bind("link", '<Leave>', lambda _, tw=text_area: self._show_normal_cursor(_, tw))
        text_area.tag_bind("link", '<Button-1>', lambda _, link=license_link: self._open_link(_, link))

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class MainHelpDialog(HelpDialog):
    """Help dialog window to assist the user in the :class:`pyplt.gui.mainmenu.MainMenu` window.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the MainMenu.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Main Menu - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        intro_text = "This menu allows you to select between two modes of operation: a beginner mode, and an " \
                     "advanced mode.\n\n"

        beginner_heading = "Beginner Mode\n"

        beginner_text = "The beginner mode simplifies the experiment setup process into 4 easy and quick steps: " \
                        "loading the data set, choosing whether to apply feature selection, choosing a preference " \
                        "learning algorithm, and finally running the experiment.\n\n"

        advanced_heading = "Advanced Mode\n"

        advanced_text = "The advanced mode on the other hand involves 5 steps, all but the last of which are " \
                        "encapsulated in their own detailed tab: loading the data set, data pre-processing, " \
                        "feature selection, preference learning, and running the experiment. In this case, each of " \
                        "the three tabs for steps 2-4 provides a set of options or parameters through which you " \
                        "may fine-tune the experiment setup."

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, intro_text)
        text_area.insert(tk.END, beginner_heading, "heading")
        text_area.insert(tk.END, beginner_text)
        text_area.insert(tk.END, advanced_heading, "heading")
        text_area.insert(tk.END, advanced_text)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h2)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class LoadDataHelpDialog(HelpDialog):
    """Help dialog window to assist the user in the `Load Data` tab.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the `Load Data` tab.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=24, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Load Data - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        advanced_text = "In this advanced mode of PLT the experiment set-up process is divided into 5 steps, all but " \
                        "the last of which are encapsulated in their own detailed tab: loading the data set, " \
                        "data pre-processing, feature selection, preference learning, and running the experiment. " \
                        "Each of the three tabs for steps 2-4 provides a set of options or parameters through which " \
                        "you may fine-tune the experiment setup. Initially, the interface areas for steps 2-5 " \
                        "are locked. Once a dataset is correctly loaded, they are enabled and accessible. The " \
                        "interface allows you to set up the parameters for steps 2-4 in any order and in fact, " \
                        "pre-processing and feature selection are optional. Once all the options are set, you can " \
                        "run the experiment from the last tab.\n\n"

        tab_heading = "Loading the Dataset\n"

        tab_text_01 = "In this tab you are asked to load your data. " \
                      "A dataset needs to contain two elements: a set of objects (input) and the relation or order " \
                      "among them (output). In PLT, the dataset may be loaded in one of two formats: a "
        tab_text_02 = "single file format"  # bold
        tab_text_03 = " for problems where a total order of objects exists and a "
        tab_text_04 = "dual file format"  # bold
        tab_text_05 = " for problems where a partial order of objects exists. A total order of objects is a "
        tab_text_06 = "rating"  # bold
        tab_text_07 = " given for each object. A partial order is a set of "
        tab_text_08 = "pairwise preferences"  # bold
        tab_text_09 = " given for a number of objects in the dataset. In the single file format, a single " \
                      "Comma-Separated-Value (.csv) file must be uploaded. This file should contain the " \
                      "objects together with their individual ratings. On the other hand, in the dual file " \
                      "format, two Comma-Separated-Value (.csv) files must be uploaded: a file containing the " \
                      "objects and a file containing the pairwise preferences.\n\n"

        objects_heading = "Objects\n"

        objects_text_01 = "In PLT, all objects in the "
        objects_text_02 = "objects file or single dataset file"  # bold
        objects_text_03 = " have to be represented by the same list of features or attributes. Each line/row of the " \
                          "file contains the feature values* of one object separated by a single character (comma by " \
                          "default).\n(Optional: the first line/row of the file can contain the name of the " \
                          "features.)\n(Optional: the first feature (column) of each object (line/row) can be used " \
                          "as object ID. ID values must be unique integers.)\n*Please note that at the time being, " \
                          "the Python implementation of PLT does not yet support nominal data and therefore only " \
                          "numeric feature values in integer (e.g. 1), floating point (e.g. 0.01) or scientific " \
                          "(e.g 1e-10) format are permitted.\n\n"

        total_order_heading = "Ratings: Total Order\n"

        total_order_text_01 = "When the available order among objects is total (i.e., the relation between any pair " \
                              "of objects is known) and given as a numeric value** assigned to each object, this " \
                              "value can be included as the last feature (column) in the "
        total_order_text_02 = "objects file"  # bold
        total_order_text_03 = " but uploaded as a "
        total_order_text_04 = "single data file"  # bold
        total_order_text_05 = ".\n**Only numeric values in integer (e.g. 1), floating point (e.g. 0.01) or " \
                              "scientific (e.g 1e-10) format are permitted.\n\n"

        partial_order_heading = "Pairwise Preferences: Partial Order\n"

        partial_order_text_01 = "When the available order among objects is partial (i.e., only the relation between " \
                                "some pairs of objects is known), this information should be included in a "
        partial_order_text_02 = "separate order (ranks) file"  # bold
        partial_order_text_03 = ".\nEach line/row of the order file contains a pair of object IDs, the first being " \
                                "that of the preferred object in the pair and the second being that of the other " \
                                "(non-preferred) object in the pair. Note that when the "
        partial_order_text_04 = "objects file"  # bold
        partial_order_text_05 = " does not contain object IDs, the line/row number is used as ID (starting at 0 and " \
                                "excluding the optional labels line/row).\n(Optional: the first line/row of the file " \
                                "can contain the name of columns; e.g., ‘PreferredObject’, and " \
                                "‘NonPreferredObject’.)\n(Optional: the first column of each object (line/row) can " \
                                "be used as a rank ID. ID values must be unique integers.)"

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, advanced_text)
        text_area.insert(tk.END, tab_heading, "heading")
        text_area.insert(tk.END, tab_text_01)
        text_area.insert(tk.END, tab_text_02, "bold")
        text_area.insert(tk.END, tab_text_03)
        text_area.insert(tk.END, tab_text_04, "bold")
        text_area.insert(tk.END, tab_text_05)
        text_area.insert(tk.END, tab_text_06, "bold")
        text_area.insert(tk.END, tab_text_07)
        text_area.insert(tk.END, tab_text_08, "bold")
        text_area.insert(tk.END, tab_text_09)

        text_area.insert(tk.END, objects_heading, "heading")
        text_area.insert(tk.END, objects_text_01)
        text_area.insert(tk.END, objects_text_02, "bold")
        text_area.insert(tk.END, objects_text_03)

        text_area.insert(tk.END, total_order_heading, "heading")
        text_area.insert(tk.END, total_order_text_01)
        text_area.insert(tk.END, total_order_text_02, "bold")
        text_area.insert(tk.END, total_order_text_03)
        text_area.insert(tk.END, total_order_text_04, "bold")
        text_area.insert(tk.END, total_order_text_05)

        text_area.insert(tk.END, partial_order_heading, "heading")
        text_area.insert(tk.END, partial_order_text_01)
        text_area.insert(tk.END, partial_order_text_02, "bold")
        text_area.insert(tk.END, partial_order_text_03)
        text_area.insert(tk.END, partial_order_text_04, "bold")
        text_area.insert(tk.END, partial_order_text_05)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class PreprocHelpDialog(HelpDialog):
    """Help dialog window to assist the user in the `Preprocessing` tab.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the `Preprocessing` tab.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Preprocessing - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        tab_text_01 = "This tab allows you to pre-process your dataset in several ways: feature extraction, " \
                      "including/excluding features, feature normalization, and dataset shuffling.\n\n"
        #               "The pre-processing settings you choose will be reflected immediately " \
        #               "in the '"
        # tab_text_02 = "Data Preview"  # bold
        # tab_text_03 = "' area of the tab which provides a preview of your dataset.\n\n"

        ae_heading = "Feature Extraction\n"
        ae_text_01 = "If your dataset does not include predetermined features, you may choose to apply "
        ae_text_02 = "automatic feature extraction (AFE)"  # bold
        ae_text_03 = " by selecting the corresponding radiobutton at the top of the tab. AFE uses an "
        ae_text_04 = "autoencoder"  # bold
        ae_text_05 = " to extract features from your data. The autoencoder may also be considered a form of "
        ae_text_06 = "dimensionality reduction or data compression"  # NOT bold
        ae_text_07 = ". The autoencoder consists of two parts: the "
        ae_text_08 = "encoder"  # bold
        ae_text_09 = " and the "
        ae_text_10 = "decoder"  # bold
        ae_text_11 = ". The encoder compresses the input data whereas the decoder decompresses the " \
                     "compressed version the data to create as accurate a reconstruction of the input as " \
                     "possible. The layer in between the encoder and the decoder (i.e., the "
        ae_text_12 = "code layer"  # bold
        ae_text_13 = ") stores the compressed (encoded) version of the input. The topology of both the encoder and " \
                     "the decoder (which are generally symmetrical) as well as the "
        ae_text_14 = "code size"  # bold
        ae_text_15 = " (i.e., the number of neurons in the code layer) are to be specified by the user. Each layer " \
                     "in the encoder should be smaller (i.e., have less neurons) than the previous one, whereas each " \
                     "layer in the decoder should be larger (i.e., have more neurons) than the previous one. The " \
                     "learning rate, error threshold and epoch parameters of the backpropagation algorithm which " \
                     "trains the autoencoder may also be specified by the user. The autoencoder is optimized using " \
                     "the Adam Optimizer and its loss/performance is determined via the Mean Squared Error " \
                     "function.\n\n"

        include_heading = "Including/Excluding Features\n"
        include_text_01 = "You may choose which features to include in or exclude out of the preference learning " \
                          "experiment via the checkbutton toggles next to each of the features in the '"
        include_text_02 = "Pre-processing Settings"  # bold
        include_text_03 = "' area of the tab.\n\n"

        norm_heading = "Feature Normalization\n"
        norm_text_01 = "You may choose to apply one of two kinds of normalization to the features in your dataset: "
        norm_text_02 = "Min-Max"  # bold
        norm_text_03 = " or "
        norm_text_04 = "Z-Score"  # bold
        norm_text_05 = ". The "
        norm_text_06 = "Min-Max"  # bold
        norm_text_07 = " method transposes the values of the selected features to fit the range of 0 to 1." \
                       " On the other hand, the "
        norm_text_08 = "Z-Score"
        norm_text_09 = " method transforms the values of the selected feature such that the average value of " \
                       "the feature is zero and the standard deviation is one.\n\n"

        shuffle_heading = "Dataset Shuffling\n"
        shuffle_text_01 = "You may choose whether or not your dataset is shuffled at the start of the experiment " \
                          "execution (i.e., prior to fold splitting, rank derivation, and normalization " \
                          "(if applicable)) via the tickbox at the bottom of the tab. " \
                          "If the dual file format is being used, it is the order of the ranks (pairwise " \
                          "preferences) that is randomized if you choose to shuffle. " \
                          "If the single file format is being used, it is " \
                          "the order of the samples that is randomized if you choose to shuffle. " \
                          "The randomization may be controlled via a "
        shuffle_text_02 = "random seed"  # bold
        shuffle_text_03 = " number (integer) which you may " \
                          "optionally enter in the corresponding text box that appears when you choose to shuffle."

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, tab_text_01)
        # text_area.insert(tk.END, tab_text_02, "bold")
        # text_area.insert(tk.END, tab_text_03)

        text_area.insert(tk.END, ae_heading, "heading")
        text_area.insert(tk.END, ae_text_01)
        text_area.insert(tk.END, ae_text_02, "bold")
        text_area.insert(tk.END, ae_text_03)
        text_area.insert(tk.END, ae_text_04, "bold")
        text_area.insert(tk.END, ae_text_05)
        text_area.insert(tk.END, ae_text_06)  # NOT bold
        text_area.insert(tk.END, ae_text_07)
        text_area.insert(tk.END, ae_text_08, "bold")
        text_area.insert(tk.END, ae_text_09)
        text_area.insert(tk.END, ae_text_10, "bold")
        text_area.insert(tk.END, ae_text_11)
        text_area.insert(tk.END, ae_text_12, "bold")
        text_area.insert(tk.END, ae_text_13)
        text_area.insert(tk.END, ae_text_14, "bold")
        text_area.insert(tk.END, ae_text_15)

        text_area.insert(tk.END, include_heading, "heading")
        text_area.insert(tk.END, include_text_01)
        text_area.insert(tk.END, include_text_02, "bold")
        text_area.insert(tk.END, include_text_03)

        text_area.insert(tk.END, norm_heading, "heading")
        text_area.insert(tk.END, norm_text_01)
        text_area.insert(tk.END, norm_text_02, "bold")
        text_area.insert(tk.END, norm_text_03)
        text_area.insert(tk.END, norm_text_04, "bold")
        text_area.insert(tk.END, norm_text_05)
        text_area.insert(tk.END, norm_text_06, "bold")
        text_area.insert(tk.END, norm_text_07)
        text_area.insert(tk.END, norm_text_08, "bold")
        text_area.insert(tk.END, norm_text_09)

        text_area.insert(tk.END, shuffle_heading, "heading")
        text_area.insert(tk.END, shuffle_text_01)
        text_area.insert(tk.END, shuffle_text_02, "bold")
        text_area.insert(tk.END, shuffle_text_03)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h2)  # , underline=1
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class FSHelpDialog(HelpDialog):
    """Help dialog window to assist the user in the `Feature Selection` tab.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the `Feature Selection` tab.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Feature Selection - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        tab_text = "This tab allows you to apply a feature selection method (optional) which automatically finds " \
                   "the most relevant subset of input features for preference models derived from " \
                   "your data. PLT offers the following feature selection methods:\n\n"

        sfs_heading = "Sequential Forward Selection (SFS)\n"  # heading
        sfs_text_01 = "SFS is a bottom-up hill-climbing algorithm where one feature is added at a time to the " \
                      "current feature set. The feature to be added is selected from the subset of the remaining " \
                      "features such that the new feature set generates the maximum value of the performance " \
                      "function over all candidate features for addition. The selection procedure begins with an " \
                      "empty feature set and terminates when an added feature yields equal or lower " \
                      "performance to the performance obtained without it. The performance of each subset of " \
                      "features considered is computed as the prediction accuracy of a model trained using " \
                      "that subset of features as input. All of the preference learning algorithms implemented " \
                      "in the tool can be used to train this model; i.e., "
        sfs_text_02 = "RankSVM"  # bold
        sfs_text_03 = " and "
        sfs_text_04 = "Backpropagation"  # bold
        sfs_text_05 = " (see Preference Learning tab Help dialog for more detail on each algorithm). Furthermore, " \
                      "you can choose to either train the model using the complete dataset (no validation) " \
                      "and therefore assessing the performance as the percentage of correctly classified " \
                      "training pairs or can test the generality of the results by using one of the " \
                      "following methods:\n\n"

        holdout_heading = "> Holdout Validation\n"  # bold
        holdout_text_01 = "This method will train the model on a given proportion of the dataset (e.g., 70%) " \
                          "and then test the model on the remaining proportion of the dataset (e.g., 30%).\n\n"

        kfcv_heading = "> K-Fold Cross Validation\n"  # bold
        kfcv_text_01 = "This method will train "
        kfcv_text_02 = "k"  # bold
        kfcv_text_03 = " models using different folds (partitions) of the data and return the percentage of " \
                       "correctly classified pairs not used for training (validation accuracy). Note that the " \
                       "dataset may be split into folds in two ways: automatically or manually. In the "
        kfcv_text_04 = "automatic"  # bold
        kfcv_text_05 = " approach , the '"
        kfcv_text_06 = "k"  # bold
        kfcv_text_07 = "' argument should be specified by the user (k=3 by default). On the other hand, in the "
        kfcv_text_08 = "manual"  # bold
        kfcv_text_09 = " approach, the user must upload a file specifying the "
        kfcv_text_10 = "test fold index"  # bold
        kfcv_text_11 = " for each sample in the dataset. Each row in the file should contain the index of the test " \
                       "fold which the corresponding sample in the dataset should be allocated to. If the " \
                       "single file format is used, the indices should be given with respect to objects. " \
                       "Otherwise (if the dual file format is used), the indices should be given with " \
                       "respect to the ranks. The file may " \
                       "optionally include a sample ID column as the first column of the file. In this case, the IDs " \
                       "in this column should correspond to the IDs of the data samples loaded at an earlier stage " \
                       "of PLT. Otherwise (i.e., there is only one column in the file indicating the fold indices), " \
                       "the ID of the sample in the given row is determined by the row number. The file may also " \
                       "optionally include column headers in the first row of the file. These file parameters " \
                       "should be specified by the user via the provided dialog when loading the file.\n\n"

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, tab_text)
        text_area.insert(tk.END, sfs_heading, "heading")
        text_area.insert(tk.END, sfs_text_01)
        text_area.insert(tk.END, sfs_text_02, "bold")
        text_area.insert(tk.END, sfs_text_03)
        text_area.insert(tk.END, sfs_text_04, "bold")
        text_area.insert(tk.END, sfs_text_05)
        text_area.insert(tk.END, holdout_heading, "bold")
        text_area.insert(tk.END, holdout_text_01)
        text_area.insert(tk.END, kfcv_heading, "bold")
        text_area.insert(tk.END, kfcv_text_01)
        text_area.insert(tk.END, kfcv_text_02, "bold")
        text_area.insert(tk.END, kfcv_text_03)
        text_area.insert(tk.END, kfcv_text_04, "bold")
        text_area.insert(tk.END, kfcv_text_05)
        text_area.insert(tk.END, kfcv_text_06, "bold")
        text_area.insert(tk.END, kfcv_text_07)
        text_area.insert(tk.END, kfcv_text_08, "bold")
        text_area.insert(tk.END, kfcv_text_09)
        text_area.insert(tk.END, kfcv_text_10, "bold")
        text_area.insert(tk.END, kfcv_text_11)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class PLHelpDialog(HelpDialog):
    """Help dialog window to assist the user in the `Preference Learning` tab.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the `Preference Learning` tab.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Preference Learning - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        pl_heading = "Preference Learning\n"  # heading

        tab_text = "This tab allows you to choose which preference learning algorithm will be used to infer " \
                   "the final preference model from your data. PLT offers the following preference learning " \
                   "algorithms:\n\n"

        rsvm_heading2 = "> RankSVM\n"  # heading2
        rsvm_text_01 = "The RankSVM algorithm is a rank-based version of the traditional Support Vector " \
                       "Machine (SVM) algorithm. Provided with data examples with annotated classes or categories " \
                       "as a form of output, an SVM maps the data instances onto geometric points in a high-" \
                       "dimensional space according to the input features that define them. This mapping process " \
                       "is carried out via a pre-defined kernel function. The SVM then attempts to split the " \
                       "instances according to their annotated category as accurately as possible via a " \
                       "hyperplane. A hyperplane is a geometric element of n − 1 dimensions which divides a " \
                       "space of n dimensions into two just as a 3-dimensional space is divided into two by a " \
                       "plane (2-dimensional) which, in turn, is divided into two by a line (1-dimensional). " \
                       "The goal of an SVM is therefore to find the optimal dimension in which the " \
                       "corresponding hyperplane best separates the data into different categories. Unseen " \
                       "instances may then be mapped to the space according to their features and an output " \
                       "(i.e., a category) is produced based on which sub-space they correspond to according " \
                       "to the hyperplane. In PLT, the algorithm was implemented using the "
        rsvm_text_02 = "scikit-learn"  # bold
        rsvm_text_03 = " library. For more information refer to [1-3].\n\n"

        bp_heading2 = "> Backpropagation\n"  # heading2
        bp_text_01 = "This is a gradient-descent algorithm that iteratively (over a given number of epochs) " \
                     "optimizes an error function by adjusting the weights of an artificial neural network (ANN) " \
                     "model proportionally to the gradient of the error with respect to the current value of " \
                     "the weights and current data samples. The proportion and therefore the strength of each " \
                     "update is regulated by the given learning rate. The error function used is the Rank Margin " \
                     "function which for a given pair of data samples (A and B, with A preferred over B) " \
                     "yields 0 if the network output for A (fA) is more than one unit larger than the network " \
                     "output for B (fB) and 1.0-((fA)-(fB)) otherwise. The total error is averaged over the " \
                     "complete set of pairs in the training set. If the error is below a given threshold, training " \
                     "stops before reaching the specified number of epochs, and the current weight values are " \
                     "returned as the final model. In PLT, the algorithm was implemented using the "
        bp_text_02 = "tensorflow"  # bold
        bp_text_03 = " library. For more information refer to [4-5].\n\n"

        eval_heading = "Evaluation\n"  # heading
        eval_text_01 = "You can choose to either train the model using the complete dataset (no validation) " \
                       "and therefore assessing the performance as the percentage of correctly classified " \
                       "training pairs or can test the generality of the results by using one of the " \
                       "following methods:\n\n"

        holdout_heading = "> Holdout Validation\n"  # bold
        holdout_text_01 = "This method will train the model on a given proportion of the dataset (e.g., 70%) " \
                          "and then test the model on the remaining proportion of the dataset (e.g., 30%).\n\n"

        kfcv_heading = "> K-Fold Cross Validation\n"  # bold
        kfcv_text_01 = "This method will train "
        kfcv_text_02 = "k"  # bold
        kfcv_text_03 = " models using different folds (partitions) of the data and return the percentage of " \
                       "correctly classified pairs not used for training (validation accuracy). Note that the " \
                       "dataset may be split into folds in two ways: automatically or manually. In the "
        kfcv_text_04 = "automatic"  # bold
        kfcv_text_05 = " approach , the '"
        kfcv_text_06 = "k"  # bold
        kfcv_text_07 = "' argument should be specified by the user (k=3 by default). On the other hand, in the "
        kfcv_text_08 = "manual"  # bold
        kfcv_text_09 = " approach, the user must upload a file specifying the "
        kfcv_text_10 = "test fold index"  # bold
        kfcv_text_11 = " for each sample in the dataset. Each row in the file should contain the index of the test " \
                       "fold which the corresponding sample in the dataset should be allocated to. If the " \
                       "single file format is used, the indices should be given with respect to objects. " \
                       "Otherwise (if the dual file format is used), the indices should be given with " \
                       "respect to the ranks. The file may " \
                       "optionally include a sample ID column as the first column of the file. In this case, the IDs " \
                       "in this column should correspond to the IDs of the data samples loaded at an earlier stage " \
                       "of PLT. Otherwise (i.e., there is only one column in the file indicating the fold indices), " \
                       "the ID of the sample in the given row is determined by the row number. The file may also " \
                       "optionally include column headers in the first row of the file. These file parameters " \
                       "should be specified by the user via the provided dialog when loading the file.\n\n"

        run_heading = "Execution and Results\n"
        run_text_01 = "With the algorithm configured, pressing the "
        run_text_02 = "Run Experiment"  # bold
        run_text_03 = " button will start the feature selection and preference learning (model training) " \
                      "algorithms and show an approximated progress report. Once the whole process is completed, " \
                      "a summary screen with the configuration of each step as well as the training and validation " \
                      "accuracies of the final model are displayed. You will be able to save the experiment " \
                      "report and the final model to file.\n\n"

        refs_heading = "References\n"
        refs_text = "[1] T. Joachims, \"Optimizing search engines using clickthrough data,\" in Proceedings of the " \
                    "Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. " \
                    "New York, NY, USA: ACM, 2002, pp. 133-142. [Online]. Available: http://doi.acm.org/" \
                    "10.1145/775047.775067.\n" \
                    "[2] R. Herbrich, T. Graepel and K. Obermayer, \"Support vector learning for ordinal " \
                    "regression,\" in Artificial Neural Networks, 1999. ICANN 99. Ninth International Conference " \
                    "on (Conf. Publ. no. 470), 1999.\n" \
                    "[3] scikit-learn Machine Learning in Python. Available: http://scikit-learn.org/.\n" \
                    "[4] H. P. Martinez, Y. Bengio and G. N. Yannakakis, \"Learning deep physiological models " \
                    "of affect,\" IEEE Computational Intelligence Magazine, vol. 8, (2), pp. 20-33, 2013.\n" \
                    "[5] TensorFlow. Available: https://www.tensorflow.org/."

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, pl_heading, "heading")
        text_area.insert(tk.END, tab_text)
        text_area.insert(tk.END, rsvm_heading2, "heading2")
        text_area.insert(tk.END, rsvm_text_01)
        text_area.insert(tk.END, rsvm_text_02, "bold")
        text_area.insert(tk.END, rsvm_text_03)
        text_area.insert(tk.END, bp_heading2, "heading2")
        text_area.insert(tk.END, bp_text_01)
        text_area.insert(tk.END, bp_text_02, "bold")
        text_area.insert(tk.END, bp_text_03)
        text_area.insert(tk.END, eval_heading, "heading")
        text_area.insert(tk.END, eval_text_01)
        text_area.insert(tk.END, holdout_heading, "bold")
        text_area.insert(tk.END, holdout_text_01)
        text_area.insert(tk.END, kfcv_heading, "bold")
        text_area.insert(tk.END, kfcv_text_01)
        text_area.insert(tk.END, kfcv_text_02, "bold")
        text_area.insert(tk.END, kfcv_text_03)
        text_area.insert(tk.END, kfcv_text_04, "bold")
        text_area.insert(tk.END, kfcv_text_05)
        text_area.insert(tk.END, kfcv_text_06, "bold")
        text_area.insert(tk.END, kfcv_text_07)
        text_area.insert(tk.END, kfcv_text_08, "bold")
        text_area.insert(tk.END, kfcv_text_09)
        text_area.insert(tk.END, kfcv_text_10, "bold")
        text_area.insert(tk.END, kfcv_text_11)
        text_area.insert(tk.END, run_heading, "heading")
        text_area.insert(tk.END, run_text_01)
        text_area.insert(tk.END, run_text_02, "bold")
        text_area.insert(tk.END, run_text_03)

        text_area.insert(tk.END, refs_heading, "heading")
        text_area.insert(tk.END, refs_text)

        # n.b. ensured that 'Run Experiment' button, OutputLayer#nuerons checkbox and Steps 2-4 of BeginnerMenu
        # are re-disabled or re-enabled accordingly on close of stacked windows (help dialog or load params).
        # solution via binding state changes to method which ensures re-disable (or re-enable if appropriate time/case).

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("heading2", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class ResultsHelpDialog(HelpDialog):
    """Help dialog window to assist the user in the experiment report window.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the experiment report window.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Experiment Report - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        intro_text = "This report presents a summary of the details and results of your completed modeling " \
                     "experiment.\n\n"

        save_log_heading = "Saving Experiment Reports\n"

        save_log_text_01 = "The details and results of the experiment may be saved to a Comma-Separated Value (CSV) " \
                           "file in a human-readable manner via the ‘"
        save_log_text_02 = "Save Report"  # bold
        save_log_text_03 = "’ button at the bottom of the Experiment Report window.\n\n"

        save_model_heading = "Saving Models\n"

        save_model_text_01 = "It is also possible to save the computational model inferred from your data via " \
                             "PLT to a Comma-Separated Value (CSV) file via the ‘"
        save_model_text_02 = "Save Model"  # bold
        save_model_text_03 = "’ button at the bottom of the Experiment Report window. The model is saved in a " \
                             "human-readable format.\n\nFor "
        save_model_text_04 = "RankSVM models"  # bold
        save_model_text_05 = ", support vectors are stored in the first column and the corresponding alpha values " \
                             "are stored in the second column. In PLT, each support vector is in the form of a " \
                             "preference pair (two object IDs separated by a comma in round brackets). The first " \
                             "object ID in the tuple refers to the preferred object in the pair whereas the second " \
                             "refers to the non-preferred object in the pair.\n\nFor "
        save_model_text_06 = "Artificial Neural Network models"  # bold
        save_model_text_07 = " (inferred via the "
        save_model_text_08 = "Backpropagation"  # bold
        save_model_text_09 = " algorithm), the weights of the network are stored as follows:\n" \
                             "> The ‘Layer’ column stores the name of the layer (the letter ‘h’ signifies a hidden " \
                             "layer).\n" \
                             "> The ‘Neuron’ column stores the index of the neuron in the given layer.\n" \
                             "> Columns starting with the letter ‘w’ store the weights of incoming edges to the " \
                             "given neuron (indicated by the ‘Neuron’ column) in the given layer (indicated by the " \
                             "‘Layer’ column). The number of such columns is based on the largest number of " \
                             "incoming edges present with respect to the neurons in the network. In the case of " \
                             "neurons with less than this amount of incoming edges, the excess column values are " \
                             "empty.\n" \
                             "> The ‘bias’ column stores the bias value of the neuron in the given layer.\n" \
                             "> The ‘activation_fn’ column stores the name of the activation function used by the " \
                             "given neuron in the given layer."

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, intro_text)
        text_area.insert(tk.END, save_log_heading, "heading")
        text_area.insert(tk.END, save_log_text_01)
        text_area.insert(tk.END, save_log_text_02, "bold")
        text_area.insert(tk.END, save_log_text_03)
        text_area.insert(tk.END, save_model_heading, "heading")
        text_area.insert(tk.END, save_model_text_01)
        text_area.insert(tk.END, save_model_text_02, "bold")
        text_area.insert(tk.END, save_model_text_03)
        text_area.insert(tk.END, save_model_text_04, "bold")
        text_area.insert(tk.END, save_model_text_05)
        text_area.insert(tk.END, save_model_text_06, "bold")
        text_area.insert(tk.END, save_model_text_07)
        text_area.insert(tk.END, save_model_text_08, "bold")
        text_area.insert(tk.END, save_model_text_09)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class BeginnerStep1HelpDialog(HelpDialog):
    """Help dialog window to assist the user in Step 1 of the BeginnerMenu.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for Step 1 of the BeginnerMenu.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Beginner Mode - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        beginner_text = "This beginner mode simplifies the experiment setup process into 4 easy and quick steps: " \
                        "loading the data set, choosing whether to apply feature selection, choosing a preference " \
                        "learning algorithm, and finally running the experiment. Initially, the interface areas " \
                        "for steps 2-4 are locked. Once a dataset is correctly loaded, they are enabled and " \
                        "accessible. The interface allows you to set up the parameters for steps 2-3 in any order. " \
                        "Once all the options are set, you can run the experiment via the button at the bottom " \
                        "of the window.\n\n"

        step1_heading = "Step 1: Load Data\n"

        step1_tab_text_01 = "In this step you are asked to load your data. " \
                            "A dataset needs to contain two elements: a set of objects (input) and the relation " \
                            "or order among them (output). In PLT, the dataset may be loaded in one of two formats: a "
        step1_tab_text_02 = "single file format"  # bold
        step1_tab_text_03 = " for problems where a total order of objects exists and a "
        step1_tab_text_04 = "dual file format"  # bold
        step1_tab_text_05 = " for problems where a partial order of objects exists. A total order of objects is a "
        step1_tab_text_06 = "rating"  # bold
        step1_tab_text_07 = " given for each object. A partial order is a set of "
        step1_tab_text_08 = "pairwise preferences"  # bold
        step1_tab_text_09 = " given for a number of objects in the dataset. In the single file format, a single " \
                            "Comma-Separated-Value (.csv) file must be uploaded. This file should contain the " \
                            "objects together with their individual ratings. On the other hand, in the dual file " \
                            "format, two Comma-Separated-Value (.csv) files must be uploaded: a file containing the " \
                            "objects and a file containing the pairwise preferences.\n\n"

        objects_heading = "Objects\n"

        objects_text_01 = "In PLT, all objects in the "
        objects_text_02 = "objects file or single dataset file"  # bold
        objects_text_03 = " have to be represented by the same list of features or attributes. Each line/row of the " \
                          "file contains the feature values* of one object separated by a single character (comma by " \
                          "default).\n(Optional: the first line/row of the file can contain the name of the " \
                          "features.)\n(Optional: the first feature (column) of each object (line/row) can be used " \
                          "as object ID. ID values must be unique integers.)\n*Please note that at the time being, " \
                          "the Python implementation of PLT does not yet support nominal data and therefore only " \
                          "numeric feature values in integer (e.g. 1), floating point (e.g. 0.01) or scientific " \
                          "(e.g 1e-10) format are permitted.\n\n"

        total_order_heading = "Ratings: Total Order\n"

        total_order_text_01 = "When the available order among objects is total (i.e., the relation between any pair " \
                              "of objects is known) and given as a numeric value** assigned to each object, this " \
                              "value can be included as the last feature (column) in the "
        total_order_text_02 = "objects file"  # bold
        total_order_text_03 = " but uploaded as a "
        total_order_text_04 = "single data file"  # bold
        total_order_text_05 = ".\n**Only numeric values in integer (e.g. 1), floating point (e.g. 0.01) or " \
                              "scientific (e.g 1e-10) format are permitted.\n\n"

        partial_order_heading = "Pairwise Preferences: Partial Order\n"

        partial_order_text_01 = "When the available order among objects is partial (i.e., only the relation between " \
                                "some pairs of objects is known), this information should be included in a "
        partial_order_text_02 = "separate order (ranks) file"  # bold
        partial_order_text_03 = ".\nEach line/row of the order file contains a pair of object IDs, the first being " \
                                "that of the preferred object in the pair and the second being that of the other " \
                                "(non-preferred) object in the pair. Note that when the "
        partial_order_text_04 = "objects file"  # bold
        partial_order_text_05 = " does not contain object IDs, the line/row number is used as ID (starting at 0 and " \
                                "excluding the optional labels line/row).\n(Optional: the first line/row of the file " \
                                "can contain the name of columns; e.g., ‘PreferredObject’, and " \
                                "‘NonPreferredObject’.)\n(Optional: the first column of each object (line/row) can " \
                                "be used as a rank ID. ID values must be unique integers.)\n\n"

        norm_heading = "Feature Normalization\n"
        norm_text_01 = "By default, all features in the dataset are normalized using the "
        norm_text_02 = "Min-Max"  # bold
        norm_text_03 = " method which transposes the values of the given feature to fit the range of 0 to 1.\n\n"

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, beginner_text)
        text_area.insert(tk.END, step1_heading, "heading")
        text_area.insert(tk.END, step1_tab_text_01)
        text_area.insert(tk.END, step1_tab_text_02, "bold")
        text_area.insert(tk.END, step1_tab_text_03)
        text_area.insert(tk.END, step1_tab_text_04, "bold")
        text_area.insert(tk.END, step1_tab_text_05)
        text_area.insert(tk.END, step1_tab_text_06, "bold")
        text_area.insert(tk.END, step1_tab_text_07)
        text_area.insert(tk.END, step1_tab_text_08, "bold")
        text_area.insert(tk.END, step1_tab_text_09)

        text_area.insert(tk.END, objects_heading, "heading2")
        text_area.insert(tk.END, objects_text_01)
        text_area.insert(tk.END, objects_text_02, "bold")
        text_area.insert(tk.END, objects_text_03)

        text_area.insert(tk.END, total_order_heading, "heading2")
        text_area.insert(tk.END, total_order_text_01)
        text_area.insert(tk.END, total_order_text_02, "bold")
        text_area.insert(tk.END, total_order_text_03)
        text_area.insert(tk.END, total_order_text_04, "bold")
        text_area.insert(tk.END, total_order_text_05)

        text_area.insert(tk.END, partial_order_heading, "heading2")
        text_area.insert(tk.END, partial_order_text_01)
        text_area.insert(tk.END, partial_order_text_02, "bold")
        text_area.insert(tk.END, partial_order_text_03)
        text_area.insert(tk.END, partial_order_text_04, "bold")
        text_area.insert(tk.END, partial_order_text_05)

        text_area.insert(tk.END, norm_heading, "heading2")
        text_area.insert(tk.END, norm_text_01)
        text_area.insert(tk.END, norm_text_02, "bold")
        text_area.insert(tk.END, norm_text_03)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("heading2", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


# TODO: add help info about automatic feature extraction (vs 'manual') & autoencoder in Beginner Menu!!!
class BeginnerStep2HelpDialog(HelpDialog):
    """Help dialog window to assist the user in Step 2 of the BeginnerMenu.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for Step 2 of the BeginnerMenu.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Beginner Mode - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        step2_heading = "Step 2: Feature Selection\n"

        step2_tab_text = "This step allows you to apply a feature selection method (optional) which automatically " \
                         "finds the most relevant subset of input features for preference models derived from " \
                         "your data. PLT uses the Sequential Forward Selection method.\n\n"

        sfs_heading = "Sequential Forward Selection (SFS)\n"  # heading
        sfs_text_01 = "SFS is a bottom-up hill-climbing algorithm where one feature is added at a time to the " \
                      "current feature set. The feature to be added is selected from the subset of the remaining " \
                      "features such that the new feature set generates the maximum value of the performance " \
                      "function over all candidate features for addition. The selection procedure begins with an " \
                      "empty feature set and terminates when an added feature yields equal or lower " \
                      "performance to the performance obtained without it. The performance of each subset of " \
                      "features considered is computed as the prediction accuracy of a model trained using " \
                      "that subset of features as input. In the Beginner Mode, the "
        sfs_text_02 = "RankSVM"  # bold
        sfs_text_03 = " preference learning algorithm is used to train this model using a linear kernel function " \
                      "(see Preference Learning section for more detail on the algorithm). Furthermore, the model " \
                      "is tested for the generality of its results by using "
        sfs_text_04 = "Holdout"  # bold
        sfs_text_05 = " validation. This means that the model is trained on a proportion of the " \
                      "dataset (70% in the Beginner Mode) and then tested on the remaining proportion " \
                      "of the dataset (30%).\n\n"

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, step2_heading, "heading")
        text_area.insert(tk.END, step2_tab_text)
        text_area.insert(tk.END, sfs_heading, "heading2")
        text_area.insert(tk.END, sfs_text_01)
        text_area.insert(tk.END, sfs_text_02, "bold")
        text_area.insert(tk.END, sfs_text_03)
        text_area.insert(tk.END, sfs_text_04, "bold")
        text_area.insert(tk.END, sfs_text_05)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("heading2", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class BeginnerStep3HelpDialog(HelpDialog):
    """Help dialog window to assist the user in Step 3 of the BeginnerMenu.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for Step 3 of the BeginnerMenu.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Beginner Mode - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        step3_heading = "Step 3: Preference Learning\n"

        step3_tab_text = "This step allows you to choose which preference learning algorithm will be used to infer " \
                         "the final preference model from your data. PLT offers the following preference learning " \
                         "algorithms:\n\n"

        rsvm_heading2 = "> RankSVM\n"  # heading2
        rsvm_text_01 = "The RankSVM algorithm is a rank-based version of the traditional Support Vector " \
                       "Machine (SVM) algorithm. Provided with data examples with annotated classes or categories " \
                       "as a form of output, an SVM maps the data instances onto geometric points in a high-" \
                       "dimensional space according to the input features that define them. This mapping process " \
                       "is carried out via a pre-defined kernel function (a linear function is used in " \
                       "the Beginner Mode). The SVM then attempts to split the " \
                       "instances according to their annotated category as accurately as possible via a " \
                       "hyperplane. A hyperplane is a geometric element of n − 1 dimensions which divides a " \
                       "space of n dimensions into two just as a 3-dimensional space is divided into two by a " \
                       "plane (2-dimensional) which, in turn, is divided into two by a line (1-dimensional). " \
                       "The goal of an SVM is therefore to find the optimal dimension in which the " \
                       "corresponding hyperplane best separates the data into different categories. Unseen " \
                       "instances may then be mapped to the space according to their features and an output " \
                       "(i.e., a category) is produced based on which sub-space they correspond to according " \
                       "to the hyperplane. In PLT, the algorithm was implemented using the "
        rsvm_text_02 = "scikit-learn"  # bold
        rsvm_text_03 = " library. For more information refer to [1-3].\n\n"

        bp_heading2 = "> Backpropagation\n"  # heading2
        bp_text_01 = "This is a gradient-descent algorithm that iteratively (over 10 epochs in the Beginner Mode) " \
                     "optimizes an error function by adjusting the weights of an artificial neural network (ANN) " \
                     "model proportionally to the gradient of the error with respect to the current value of the " \
                     "weights and current data samples. The proportion and therefore the strength of each update " \
                     "is regulated by the learning rate (0.1 in the Beginner Mode). The error function used " \
                     "is the Rank Margin function which for a given pair of data samples (A and B, with A preferred " \
                     "over B) yields 0 if the network output for A (fA) is more than one unit larger than the " \
                     "network output for B (fB) and 1.0-((fA)-(fB)) otherwise. The total error is averaged over " \
                     "the complete set of pairs in the training set. If the error is below a given threshold " \
                     "(0.1 in the Beginner Mode), training stops before reaching the specified number of epochs, " \
                     "and the current weight values are returned as the final model. In the Beginner Mode, the " \
                     "network topology is set up such that it contains one hidden layer of 5 neurons and uses " \
                     "the ReLU activation function for each neuron. In PLT, the algorithm was implemented using the "
        bp_text_02 = "tensorflow"  # bold
        bp_text_03 = " library. For more information refer to [4-5].\n\n"

        eval_heading = "Evaluation\n"  # heading
        eval_text_01 = "The generality of the results is tested by using "
        eval_text_02 = "Holdout"
        eval_text_03 = " validation. This means that the model is trained on a proportion of the " \
                       "dataset (70% in the Beginner Mode) and then tested on the remaining proportion " \
                       "of the dataset (30%).\n\n"

        refs_heading = "References\n"
        refs_text = "[1] T. Joachims, \"Optimizing search engines using clickthrough data,\" in Proceedings of the " \
                    "Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. " \
                    "New York, NY, USA: ACM, 2002, pp. 133-142. [Online]. Available: http://doi.acm.org/" \
                    "10.1145/775047.775067.\n" \
                    "[2] R. Herbrich, T. Graepel and K. Obermayer, \"Support vector learning for ordinal " \
                    "regression,\" in Artificial Neural Networks, 1999. ICANN 99. Ninth International Conference " \
                    "on (Conf. Publ. no. 470), 1999.\n" \
                    "[3] scikit-learn Machine Learning in Python. Available: http://scikit-learn.org/.\n" \
                    "[4] H. P. Martinez, Y. Bengio and G. N. Yannakakis, \"Learning deep physiological models " \
                    "of affect,\" IEEE Computational Intelligence Magazine, vol. 8, (2), pp. 20-33, 2013.\n" \
                    "[5] TensorFlow. Available: https://www.tensorflow.org/."

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, step3_heading, "heading")
        text_area.insert(tk.END, step3_tab_text)
        text_area.insert(tk.END, rsvm_heading2, "heading2")
        text_area.insert(tk.END, rsvm_text_01)
        text_area.insert(tk.END, rsvm_text_02, "bold")
        text_area.insert(tk.END, rsvm_text_03)
        text_area.insert(tk.END, bp_heading2, "heading2")
        text_area.insert(tk.END, bp_text_01)
        text_area.insert(tk.END, bp_text_02, "bold")
        text_area.insert(tk.END, bp_text_03)
        text_area.insert(tk.END, eval_heading, "heading")
        text_area.insert(tk.END, eval_text_01)
        text_area.insert(tk.END, eval_text_02, "bold")
        text_area.insert(tk.END, eval_text_03)

        text_area.insert(tk.END, refs_heading, "heading")
        text_area.insert(tk.END, refs_text)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("heading2", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class BeginnerStep4HelpDialog(HelpDialog):
    """Help dialog window to assist the user in Step 4 of the BeginnerMenu.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for Step 4 of the BeginnerMenu.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Beginner Mode - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        step4_heading = "Step 4: Execution and Results\n"

        run_text_01 = "With the algorithm configured, pressing the "
        run_text_02 = "Run Experiment"  # bold
        run_text_03 = " button will start the feature selection (if applicable) and preference learning " \
                      "(model training) algorithms and show an approximated progress report. Once the whole " \
                      "process is completed, a summary screen with the configuration of each step as well as " \
                      "the training and validation accuracies of the final model are displayed. You will be able " \
                      "to save the experiment report and the final model to file.\n\n"

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, step4_heading, "heading")
        text_area.insert(tk.END, run_text_01)
        text_area.insert(tk.END, run_text_02, "bold")
        text_area.insert(tk.END, run_text_03)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("heading2", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')


class RankDerivationHelpDialog(HelpDialog):
    """Help dialog window to assist the user with the rank derivation methods when loading the dataset.

    Extends :class:`pyplt.gui.util.help.HelpDialog`.
    """
    def __init__(self, parent_window):
        """Initializes the window widget with the help text for the LoadingParamsWindow.

        :param parent_window: the window which this window widget will be stacked on top of.
        :type parent_window: `tkinter.Toplevel`
        """
        # call super init method
        HelpDialog.__init__(self, parent_window)

        text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
                            cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)

        # add scrollbar
        self._v_scroll.config(command=text_area.yview)
        self._v_scroll.pack(side='right', fill='y')
        text_area.configure(yscrollcommand=self._v_scroll.set)

        tk.Label(self.main_frame, text="Preference Derivation Parameters - Help", font=self.ebrima_h1,
                 bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))

        intro_text = "When loading a dataset in the single file format (i.e., ratings-based), you are given the " \
                     "option to control how the pairwise preferences (ranks) are derived from your ratings-based " \
                     "data via the following parameters:\n\n"

        mdm_heading = "Minimum Distance Margin\n"

        text_01 = "The minimum distance margin is the minimum difference between the ratings of a given pair " \
                  "of objects/samples that is required for the object pair to be considered a valid and clear " \
                  "preference. This is set to 0.0 by default.\n\n"

        memory_heading = "Memory\n"

        text_02 = "The memory parameter specifies how many neighbouring objects/samples are to be compared with a " \
                  "given object/sample when constructing the pairwise ranks. By default this is set to '"
        text_03 = "ALL"  # bold
        text_04 = "' (i.e., all objects/samples are compared to each other) however you may reduce this by " \
                  "specifying an "
        text_05 = "integer value"  # bold
        text_06 = " via the slider or text box.\n\n"

        text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # insert text
        text_area.insert(tk.END, intro_text)
        text_area.insert(tk.END, mdm_heading, "heading")
        text_area.insert(tk.END, text_01)
        text_area.insert(tk.END, memory_heading, "heading")
        text_area.insert(tk.END, text_02)
        text_area.insert(tk.END, text_03, "bold")
        text_area.insert(tk.END, text_04)
        text_area.insert(tk.END, text_05, "bold")
        text_area.insert(tk.END, text_06)

        # make headings actual headings
        text_area.tag_config("heading", font=self.ebrima_h1)
        text_area.tag_config("heading2", font=self.ebrima_h2)
        text_area.tag_config("bold", font=self.ebrima_bold)

        # once text is inserted, set widget to readonly
        text_area.config(state='disabled')

# class BeginnerHelpDialog(HelpDialog):
#     """Help dialog window to assist the user in the :class:`pyplt.gui.beginnermenu.BeginnerMenu`.
#
#     Extends :class:`pyplt.gui.util.help.HelpDialog`.
#     """
#     def __init__(self, parent_window):
#         """Initializes the window widget with the help text for the :class:`pyplt.gui.beginnermenu.BeginnerMenu`.
#
#         :param parent_window: the window which this window widget will be stacked on top of.
#         :type parent_window: `tkinter.Toplevel`
#         """
#         # call super init method
#         HelpDialog.__init__(self, parent_window)
#
#         text_area = tk.Text(self.main_frame, bd=0, bg=colours.BACKGROUND, selectbackground=colours.BACKGROUND,
#                             cursor='arrow', wrap=tk.WORD, height=14, padx=20, pady=20)
#
#         # add scrollbar
#         self._v_scroll.config(command=text_area.yview)
#         self._v_scroll.pack(side='right', fill='y')
#         text_area.configure(yscrollcommand=self._v_scroll.set)
#
#         tk.Label(self.main_frame, text="Beginner Mode - Help", font=self.ebrima_h1,
#                  bg=colours.HELP_BACKGROUND, fg='white').pack(pady=(20, 0))
#
#         beginner_text = "This beginner mode simplifies the experiment setup process into 4 easy and quick steps: " \
#                         "loading the data set, choosing whether to apply feature selection, choosing a preference " \
#                         "learning algorithm, and finally running the experiment. Initially, the interface areas " \
#                         "for steps 2-4 are locked. Once a dataset is correctly loaded, they are enabled and " \
#                         "accessible. The interface allows you to set up the parameters for steps 2-3 in any order. " \
#                         "Once all the options are set, you can run the experiment via the button at the bottom " \
#                         "of the window.\n\n"
#
#         step1_heading = "Step 1: Load Data\n"
#
#         step1_tab_text_01 = "In this step you are asked to load your data. " \
#                             "A dataset needs to contain two elements: a set of objects (input) and the relation " \
#                             "or order among them (output). In PLT, the dataset may be loaded in one of two formats: a "
#         step1_tab_text_02 = "single file format"  # bold
#         step1_tab_text_03 = " for problems where a total order of objects exists and a "
#         step1_tab_text_04 = "dual file format"  # bold
#         step1_tab_text_05 = " for problems where a partial order of objects exists. A total order of objects is a "
#         step1_tab_text_06 = "rating"  # bold
#         step1_tab_text_07 = " given for each object. A partial order is a set of "
#         step1_tab_text_08 = "pairwise preferences"  # bold
#         step1_tab_text_09 = " given for a number of objects in the dataset. In the single file format, a single " \
#                             "Comma-Separated-Value (.csv) file must be uploaded. This file should contain the " \
#                             "objects together with their individual ratings. On the other hand, in the dual file " \
#                             "format, two Comma-Separated-Value (.csv) files must be uploaded: a file containing the " \
#                             "objects and a file containing the pairwise preferences.\n\n"
#
#         objects_heading = "Objects\n"
#
#         objects_text_01 = "In PLT, all objects in the "
#         objects_text_02 = "objects file or single dataset file"  # bold
#         objects_text_03 = " have to be represented by the same list of features or attributes. Each line/row of the " \
#                           "file contains the feature values* of one object separated by a single character (comma by " \
#                           "default).\n(Optional: the first line/row of the file can contain the name of the " \
#                           "features.)\n(Optional: the first feature (column) of each object (line/row) can be used " \
#                           "as object ID. ID values must be unique integers.)\n*Please note that at the time being, " \
#                           "the Python implementation of PLT does not yet support nominal data and therefore only " \
#                           "numeric feature values in integer (e.g. 1), floating point (e.g. 0.01) or scientific " \
#                           "(e.g 1e-10) format are permitted.\n\n"
#
#         total_order_heading = "Ratings: Total Order\n"
#
#         total_order_text_01 = "When the available order among objects is total (i.e., the relation between any pair " \
#                               "of objects is known) and given as a numeric value** assigned to each object, this " \
#                               "value can be included as the last feature (column) in the "
#         total_order_text_02 = "objects file"  # bold
#         total_order_text_03 = " but uploaded as a "
#         total_order_text_04 = "single data file"  # bold
#         total_order_text_05 = ".\n**Only numeric values in integer (e.g. 1), floating point (e.g. 0.01) or " \
#                               "scientific (e.g 1e-10) format are permitted.\n\n"
#
#         partial_order_heading = "Pairwise Preferences: Partial Order\n"
#
#         partial_order_text_01 = "When the available order among objects is partial (i.e., only the relation between " \
#                                 "some pairs of objects is known), this information should be included in a "
#         partial_order_text_02 = "separate order (ranks) file"  # bold
#         partial_order_text_03 = ".\nEach line/row of the order file contains a pair of object IDs, the first being " \
#                                 "that of the preferred object in the pair and the second being that of the other " \
#                                 "(non-preferred) object in the pair. Note that when the "
#         partial_order_text_04 = "objects file"  # bold
#         partial_order_text_05 = " does not contain object IDs, the line/row number is used as ID (starting at 0 and " \
#                                 "excluding the optional labels line/row).\n(Optional: the first line/row of the file " \
#                                 "can contain the name of columns; e.g., ‘PreferredObject’, and " \
#                                 "‘NonPreferredObject’.)\n(Optional: the first column of each object (line/row) can " \
#                                 "be used as a rank ID. ID values must be unique integers.)\n\n"
#
#         norm_heading = "Feature Normalization\n"
#         norm_text_01 = "By default, all features in the dataset are normalized using the "
#         norm_text_02 = "Min-Max"  # bold
#         norm_text_03 = " method which transposes the values of the given feature to fit the range of 0 to 1.\n\n"
#
#         step2_heading = "Step 2: Feature Selection\n"
#
#         step2_tab_text = "This step allows you to apply a feature selection method (optional) which automatically " \
#                          "finds the most relevant subset of input features for preference models derived from " \
#                          "your data. PLT uses the Sequential Forward Selection method.\n\n"
#
#         sfs_heading = "Sequential Forward Selection (SFS)\n"  # heading
#         sfs_text_01 = "SFS is a bottom-up hill-climbing algorithm where one feature is added at a time to the " \
#                       "current feature set. The feature to be added is selected from the subset of the remaining " \
#                       "features such that the new feature set generates the maximum value of the performance " \
#                       "function over all candidate features for addition. The selection procedure begins with an " \
#                       "empty feature set and terminates when an added feature yields equal or lower " \
#                       "performance to the performance obtained without it. The performance of each subset of " \
#                       "features considered is computed as the prediction accuracy of a model trained using " \
#                       "that subset of features as input. In the Beginner Mode, the "
#         sfs_text_02 = "RankSVM"  # bold
#         sfs_text_03 = " preference learning algorithm is used to train this model using a linear kernel function " \
#                       "(see Preference Learning section for more detail on the algorithm). Furthermore, the model " \
#                       "is tested for the generality of its results by using "
#         sfs_text_04 = "Holdout"  # bold
#         sfs_text_05 = " validation. This means that the model is trained on a proportion of the " \
#                       "dataset (70% in the Beginner Mode) and then tested on the remaining proportion " \
#                       "of the dataset (30%).\n\n"
#
#         step3_heading = "Step 3: Preference Learning\n"
#
#         step3_tab_text = "This step allows you to choose which preference learning algorithm will be used to infer " \
#                          "the final preference model from your data. PLT offers the following preference learning " \
#                          "algorithms:\n\n"
#
#         rsvm_heading2 = "> RankSVM\n"  # heading2
#         rsvm_text_01 = "The RankSVM algorithm is a rank-based version of the traditional Support Vector " \
#                        "Machine (SVM) algorithm. Provided with data examples with annotated classes or categories " \
#                        "as a form of output, an SVM maps the data instances onto geometric points in a high-" \
#                        "dimensional space according to the input features that define them. This mapping process " \
#                        "is carried out via a pre-defined kernel function (a linear function is used in " \
#                        "the Beginner Mode). The SVM then attempts to split the " \
#                        "instances according to their annotated category as accurately as possible via a " \
#                        "hyperplane. A hyperplane is a geometric element of n − 1 dimensions which divides a " \
#                        "space of n dimensions into two just as a 3-dimensional space is divided into two by a " \
#                        "plane (2-dimensional) which, in turn, is divided into two by a line (1-dimensional). " \
#                        "The goal of an SVM is therefore to find the optimal dimension in which the " \
#                        "corresponding hyperplane best separates the data into different categories. Unseen " \
#                        "instances may then be mapped to the space according to their features and an output " \
#                        "(i.e., a category) is produced based on which sub-space they correspond to according " \
#                        "to the hyperplane. In PLT, the algorithm was implemented using the "
#         rsvm_text_02 = "scikit-learn"  # bold
#         rsvm_text_03 = " library. For more information refer to [1-3].\n\n"
#
#         bp_heading2 = "> Backpropagation\n"  # heading2
#         bp_text_01 = "This is a gradient-descent algorithm that iteratively (over 10 epochs in the Beginner Mode) " \
#                      "optimizes an error function by adjusting the weights of an artificial neural network (ANN) " \
#                      "model proportionally to the gradient of the error with respect to the current value of the " \
#                      "weights and current data samples. The proportion and therefore the strength of each update " \
#                      "is regulated by the learning rate (0.1 in the Beginner Mode). The error function used " \
#                      "is the Rank Margin function which for a given pair of data samples (A and B, with A preferred " \
#                      "over B) yields 0 if the network output for A (fA) is more than one unit larger than the " \
#                      "network output for B (fB) and 1.0-((fA)-(fB)) otherwise. The total error is averaged over " \
#                      "the complete set of pairs in the training set. If the error is below a given threshold " \
#                      "(0.1 in the Beginner Mode), training stops before reaching the specified number of epochs, " \
#                      "and the current weight values are returned as the final model. In the Beginner Mode, the " \
#                      "network topology is set up such that it contains one hidden layer of 5 neurons and uses " \
#                      "the Sigmoid activation function for each neuron. In PLT, the algorithm was implemented using the "
#         bp_text_02 = "tensorflow"  # bold
#         bp_text_03 = " library. For more information refer to [4-5].\n\n"
#
#         eval_heading = "Evaluation\n"  # heading
#         eval_text_01 = "The generality of the results is tested by using "
#         eval_text_02 = "Holdout"
#         eval_text_03 = " validation. This means that the model is trained on a proportion of the " \
#                        "dataset (70% in the Beginner Mode) and then tested on the remaining proportion " \
#                        "of the dataset (30%).\n\n"
#
#         step4_heading = "Step 4: Execution and Results\n"
#
#         run_text_01 = "With the algorithm configured, pressing the "
#         run_text_02 = "Run Experiment"  # bold
#         run_text_03 = " button will start the feature selection (if applicable) and preference learning " \
#                       "(model training) algorithms and show an approximated progress report. Once the whole " \
#                       "process is completed, a summary screen with the configuration of each step as well as " \
#                       "the training and validation accuracies of the final model are displayed. You will be able " \
#                       "to save the experiment report and the final model to file.\n\n"
#
#         refs_heading = "References\n"
#         refs_text = "[1] T. Joachims, \"Optimizing search engines using clickthrough data,\" in Proceedings of the " \
#                     "Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. " \
#                     "New York, NY, USA: ACM, 2002, pp. 133-142. [Online]. Available: http://doi.acm.org/" \
#                     "10.1145/775047.775067.\n" \
#                     "[2] R. Herbrich, T. Graepel and K. Obermayer, \"Support vector learning for ordinal " \
#                     "regression,\" in Artificial Neural Networks, 1999. ICANN 99. Ninth International Conference " \
#                     "on (Conf. Publ. no. 470), 1999.\n" \
#                     "[3] scikit-learn Machine Learning in Python. Available: http://scikit-learn.org/.\n" \
#                     "[4] H. P. Martinez, Y. Bengio and G. N. Yannakakis, \"Learning deep physiological models " \
#                     "of affect,\" IEEE Computational Intelligence Magazine, vol. 8, (2), pp. 20-33, 2013.\n" \
#                     "[5] TensorFlow. Available: https://www.tensorflow.org/."
#
#         text_area.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
#
#         # insert text
#         text_area.insert(tk.END, beginner_text)
#         text_area.insert(tk.END, step1_heading, "heading")
#         text_area.insert(tk.END, step1_tab_text_01)
#         text_area.insert(tk.END, step1_tab_text_02, "bold")
#         text_area.insert(tk.END, step1_tab_text_03)
#         text_area.insert(tk.END, step1_tab_text_04, "bold")
#         text_area.insert(tk.END, step1_tab_text_05)
#         text_area.insert(tk.END, step1_tab_text_06, "bold")
#         text_area.insert(tk.END, step1_tab_text_07)
#         text_area.insert(tk.END, step1_tab_text_08, "bold")
#         text_area.insert(tk.END, step1_tab_text_09)
#
#         text_area.insert(tk.END, objects_heading, "heading2")
#         text_area.insert(tk.END, objects_text_01)
#         text_area.insert(tk.END, objects_text_02, "bold")
#         text_area.insert(tk.END, objects_text_03)
#
#         text_area.insert(tk.END, total_order_heading, "heading2")
#         text_area.insert(tk.END, total_order_text_01)
#         text_area.insert(tk.END, total_order_text_02, "bold")
#         text_area.insert(tk.END, total_order_text_03)
#         text_area.insert(tk.END, total_order_text_04, "bold")
#         text_area.insert(tk.END, total_order_text_05)
#
#         text_area.insert(tk.END, partial_order_heading, "heading2")
#         text_area.insert(tk.END, partial_order_text_01)
#         text_area.insert(tk.END, partial_order_text_02, "bold")
#         text_area.insert(tk.END, partial_order_text_03)
#         text_area.insert(tk.END, partial_order_text_04, "bold")
#         text_area.insert(tk.END, partial_order_text_05)
#
#         text_area.insert(tk.END, norm_heading, "heading2")
#         text_area.insert(tk.END, norm_text_01)
#         text_area.insert(tk.END, norm_text_02, "bold")
#         text_area.insert(tk.END, norm_text_03)
#
#         text_area.insert(tk.END, step2_heading, "heading")
#         text_area.insert(tk.END, step2_tab_text)
#         text_area.insert(tk.END, sfs_heading, "heading2")
#         text_area.insert(tk.END, sfs_text_01)
#         text_area.insert(tk.END, sfs_text_02, "bold")
#         text_area.insert(tk.END, sfs_text_03)
#         text_area.insert(tk.END, sfs_text_04, "bold")
#         text_area.insert(tk.END, sfs_text_05)
#
#         text_area.insert(tk.END, step3_heading, "heading")
#         text_area.insert(tk.END, step3_tab_text)
#         text_area.insert(tk.END, rsvm_heading2, "heading2")
#         text_area.insert(tk.END, rsvm_text_01)
#         text_area.insert(tk.END, rsvm_text_02, "bold")
#         text_area.insert(tk.END, rsvm_text_03)
#         text_area.insert(tk.END, bp_heading2, "heading2")
#         text_area.insert(tk.END, bp_text_01)
#         text_area.insert(tk.END, bp_text_02, "bold")
#         text_area.insert(tk.END, bp_text_03)
#         text_area.insert(tk.END, eval_heading, "heading")
#         text_area.insert(tk.END, eval_text_01)
#         text_area.insert(tk.END, eval_text_02, "bold")
#         text_area.insert(tk.END, eval_text_03)
#
#         text_area.insert(tk.END, step4_heading, "heading")
#         text_area.insert(tk.END, run_text_01)
#         text_area.insert(tk.END, run_text_02, "bold")
#         text_area.insert(tk.END, run_text_03)
#
#         text_area.insert(tk.END, refs_heading, "heading")
#         text_area.insert(tk.END, refs_text)
#
#         # make headings actual headings
#         text_area.tag_config("heading", font=self.ebrima_h1)
#         text_area.tag_config("heading2", font=self.ebrima_h2)
#         text_area.tag_config("bold", font=self.ebrima_bold)
#
#         # once text is inserted, set widget to readonly
#         text_area.config(state='disabled')
