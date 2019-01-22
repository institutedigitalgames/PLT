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

import platform
import tkinter as tk
import warnings
from tkinter import ttk
import numpy as np

from pyplt.gui.util import colours
from pyplt.exceptions import DataSetValueWarning


class DataSetPreviewFrame:
    """GUI frame for previewing datasets loaded into PLT."""

    _master = None
    _preview_canvas = None
    _preview_frame = None

    def _on_canvas_config(self, event):
        """Update the canvas scrollregion to account for its entire bounding box.

        This method is bound to all <Configure> events with respect to :attr:`self._preview_frame`.

        :param event: the <Configure> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("__ on_config called __")
        self._preview_canvas.configure(scrollregion=self._preview_canvas.bbox("all"))

    def __init__(self, master, df, heads_bg_colour=colours.PREVIEW_DEFAULT, heads_text_colour='white', style_prefix=''):
        """Instantiates a `tkinter.Canvas` object containing the frame previewing the data.

        :param master: the parent widget of the data preview canvas widget.
        :type master: `tkinter widget`
        :param df: the data to be previewed.
        :type df: `pandas.DataFrame`
        :param heads_bg_colour: the background colour to be used for the column headers
            (representing feature names) in the preview (default :attr:`pyplt.gui.util.colours.PREVIEW_DEFAULT`).
        :type heads_bg_colour: str, optional
        :param heads_text_colour: the text colour to be used for the column headers
            (representing feature names) in the preview (default 'white').
        :type heads_text_colour: str, optional
        :param style_prefix: specifies an additional prefix to add to the name of the
            style used for themed `ttk widgets` (should include a dot character at the end) (default '').
        :type style_prefix: str, optional
        """
        self._master = master
        self._heads_bg_colour = heads_bg_colour
        self._heads_text_colour = heads_text_colour

        self._preview_canvas = tk.Canvas(master, bg='#E6E6E6')

        # set scrollbars
        v_scroll = ttk.Scrollbar(master, orient="vertical", command=self._preview_canvas.yview,
                                 style=style_prefix+"PLT.Vertical.TScrollbar")
        v_scroll.pack(side='right', fill='y')
        self._preview_canvas.configure(yscrollcommand=v_scroll.set)
        h_scroll = ttk.Scrollbar(master, orient="horizontal", command=self._preview_canvas.xview,
                                 style=style_prefix+"PLT.Horizontal.TScrollbar")
        h_scroll.pack(side='bottom', fill='x')
        self._preview_canvas.configure(xscrollcommand=h_scroll.set)

        self._preview_canvas.bind('<Enter>', self._bind_mousewheel)
        self._preview_canvas.bind('<Leave>', self._unbind_mousewheel)

        self.col_limit = 50
        self.row_limit = 50
        self.column_frames = None

        self.update(df)

        self._preview_canvas.pack(side='left', expand=True, fill=tk.BOTH)

        self._OS = platform.system()

    def _bind_mousewheel(self, event):
        """Bind all mouse wheel events with respect to the canvas to a canvas-scrolling function.

        This method is called whenever an <Enter> event occurs with respect to :attr:`self._preview_canvas`.

        :param event: the <Enter> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("Binding mousewheel...")
        # for Windows OS and MacOS
        self._preview_canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        # for Linux OS
        self._preview_canvas.bind_all("<Button-4>", self._on_mouse_scroll)
        self._preview_canvas.bind_all("<Button-5>", self._on_mouse_scroll)

    def _unbind_mousewheel(self, event):
        """Unbind all mouse wheel events with respect to the canvas from any function.

        This method is called whenever a <Leave> event occurs with respect to :attr:`self._preview_canvas`.

        :param event: the <Leave> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("Unbinding mousewheel.")
        # for Windows OS and MacOS
        self._preview_canvas.unbind_all("<MouseWheel>")
        # for Linux OS
        self._preview_canvas.unbind_all("<Button-4>")
        self._preview_canvas.unbind_all("<Button-5>")

    def _on_mouse_scroll(self, event):
        """Vertically scroll through the canvas by an amount derived from the given <MouseWheel> event.

        :param event: the <MouseWheel> event that triggered the method call.
        :type event: `tkinter Event`
        """
        # print("Scrolling DATA SET PREVIEW........................")
        if self._OS == 'Linux':
            if event.num == 4:
                self._preview_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._preview_canvas.yview_scroll(1, "units")
        else:
            self._preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update(self, df):
        """Update the preview frame with the given data.

        :param df: the data to be previewed.
        :type df: `pandas.DataFrame`
        """
        # Destroy old stuff first
        if self._preview_frame is not None:
            # print("Destroying frame and canvas...")
            self._preview_frame.destroy()
            self._preview_canvas.delete("all")

        shape = list(df.shape)
        shape[0] += 1  # to include column names
        if shape[0] > self.col_limit:
            shape[0] = 52
        if shape[1] > self.col_limit:
            shape[1] = 52
        print("shape: " + str(shape))
        self.column_frames = np.empty(shape=shape[1], dtype=object)

        # Now redraw
        self._preview_frame = tk.Frame(self._preview_canvas, relief='groove', bg='#E6E6E6')

        self._preview_frame.bind('<Configure>', self._on_canvas_config)

        self._preview_canvas.create_window((0, 0), window=self._preview_frame, anchor='nw')

        r = 0
        cn = 0
        for col_name in df.columns:
            col_frame = tk.Frame(self._preview_frame)
            col_frame.grid(row=0, column=cn)
            item = tk.Label(col_frame, text=col_name, background=self._heads_bg_colour, borderwidth=2, relief='raised',
                            fg=self._heads_text_colour)
            self.column_frames[cn] = col_frame
            if len(str(col_name)) < 10:
                item.configure(width=10)
            item.grid(row=r, column=0, sticky='nsew')
            cn += 1
            if cn > self.col_limit:  # only display a maximum of (col_limit) columns!
                # add final '...' column to indicate more data
                col_frame = tk.Frame(self._preview_frame)
                col_frame.grid(row=0, column=cn)
                item = tk.Label(col_frame, text='...', background=self._heads_bg_colour, borderwidth=2,
                                relief='raised',
                                fg=self._heads_text_colour)
                self.column_frames[cn] = col_frame
                if len(str(col_name)) < 10:
                    item.configure(width=10)
                item.grid(row=r, column=0, sticky='nsew')
                break
        r += 1
        for idx, row in df.iterrows():
            c = 0
            for col in df.columns:
                # display ints as ints and floats as floats (to 2 d.p.)
                value = row[col]
                try:
                    if float(value) - int(value) == float(0.0):
                        value = str(int(value))
                    else:
                        value = str("%.2f" % float(value))
                except ValueError:
                    # can't convert value to float or int - keeping value as is (probably string)!
                    warnings.warn("The data set you are loading contains values which are not entirely numeric " +
                                  "(i.e., cannot be converted to float or int).", DataSetValueWarning)
                # display label
                col_frame = self.column_frames[c]
                item = tk.Label(col_frame, text=value, borderwidth=1, relief='groove')
                # alternate colour
                if r % 2 == 0:
                    item.configure(background='#f2f2f2')
                else:
                    item.configure(background='white')
                item.grid(row=r, column=0, sticky='nsew')
                c += 1
                if c > self.col_limit:  # only display a maximum of (col_limit) columns!
                    # insert '...' labels to indicate more data
                    # display label
                    col_frame = self.column_frames[c]
                    item = tk.Label(col_frame, text='...', borderwidth=1, relief='groove')
                    # alternate colour
                    if r % 2 == 0:
                        item.configure(background='#f2f2f2')
                    else:
                        item.configure(background='white')
                    item.grid(row=r, column=0, sticky='nsew')
                    break
            r += 1
            if r > self.row_limit:  # only display a maximum of 50 rows!
                # insert '...' labels to indicate more data
                for c in range(len(df.columns)):
                    col_frame = self.column_frames[c]
                    item = tk.Label(col_frame, text="...", borderwidth=1, relief='groove')
                    if r % 2 == 0:
                        item.configure(background='#f2f2f2')
                    else:
                        item.configure(background='white')
                    item.grid(row=r, column=0, sticky='nsew')
                    if c > self.col_limit:  # only display a maximum of (col_limit) columns! (plus one last '...')
                        break
                break

        for c in range(len(df.columns)):
            self._preview_frame.grid_columnconfigure(c, weight=1, uniform=df.columns[c])
            if c > self.col_limit:  # only display a maximum of (col_limit) columns! (plus one last '...')
                break

    def update_column(self, col_id, new_values):
        """Update the values of a single column in the preview (e.g., for normalization).

        :param col_id: the index of the column to be updated.
        :type col_id: int
        :param new_values: the new values to be displayed in the given column.
        :type new_values: array-like
        """
        col_frame = self.column_frames[col_id]
        labels = col_frame.winfo_children()
        r = -1
        for label in labels:
            if r > self.row_limit:  # only up to row limit
                break
            if r > -1:  # skip column name
                # display ints as ints and floats as floats (to 2 d.p.)
                value = new_values[r]
                if float(value) - int(value) == float(0.0):
                    value = str(int(value))
                else:
                    value = str("%.2f" % float(value))
                # print(str(label.cget("text")) + " now " + str(new_values[r]))
                label.configure(text=str(value))
            r += 1

    def destroy_preview(self):
        """Destroy the preview frame widget and the canvas widget containing it."""
        # print("prev -> preview frame Before: " + str(len(self._preview_frame.winfo_children())))
        self._preview_frame.destroy()
        # print("prev -> preview canvas Before: " + str(len(self._preview_canvas.winfo_children())))
        self._preview_canvas.destroy()
        # print("prev -> preview canvas After: " + str(len(self._preview_canvas.winfo_children())))
