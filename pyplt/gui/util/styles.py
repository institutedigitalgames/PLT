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

"""This module defines the various styles used for `ttk` widgets throughout PLT."""

from pyplt.gui.util import colours


def configure_styles(themed_style):
    """Define the styles used for ttk widgets throughout the GUI."""
    # Menubutton's
    themed_style.configure('PLT.TMenubutton', foreground='black', background='#e6e6e6')
    themed_style.map('PLT.TMenubutton', focuscolor=[('pressed', 'd3d8e2'), ('!pressed', 'white'),
                                                    ('selected', 'd3d8e2'), ('!selected', 'white')])
    themed_style.configure('White.PLT.TMenubutton', foreground='black', background='white')
    themed_style.configure('Gray.PLT.TMenubutton', foreground='black', background='#f2f2f2')
    themed_style.configure('Blue.PLT.TMenubutton', foreground='black', background=colours.PREPROC_FRONT)

    themed_style.configure('FS.PLT.TMenubutton', foreground='black', background=colours.FS_OUTER)
    themed_style.configure('PL.PLT.TMenubutton', foreground='black', background=colours.PL_OUTER)
    themed_style.configure('Sub.PL.PLT.TMenubutton', foreground='black', background=colours.PL_INNER)
    themed_style.configure('Eval.PLT.TMenubutton', foreground='black', background=colours.EVAL_OUTER)

    # Checkbutton's
    themed_style.map('PLT.TCheckbutton', background=[('active', colours.BACKGROUND),
                                                     ('!active', colours.BACKGROUND)])
    themed_style.map("White.PLT.TCheckbutton", background=[('active', 'white'),
                                                           ('!active', 'white')])
    themed_style.map('Gray.PLT.TCheckbutton', background=[('active', '#f2f2f2'),
                                                          ('!active', '#f2f2f2')])
    themed_style.map('Blue.PLT.TCheckbutton', background=[('active', colours.PREPROC_FRONT),
                                                          ('!active', colours.PREPROC_FRONT)])
    # Scrollbars
    themed_style.configure("PLT.Horizontal.TScrollbar", background=colours.BACKGROUND)
    themed_style.configure("PLT.Vertical.TScrollbar", background=colours.BACKGROUND)
    themed_style.configure("White.PLT.Vertical.TScrollbar", background='white')
    themed_style.configure("White.PLT.Horizontal.TScrollbar", background='white')
    themed_style.configure("Yellow.PLT.Horizontal.TScrollbar", background=colours.PREPROC_BACK)
    themed_style.configure("Yellow.PLT.Vertical.TScrollbar", background=colours.PREPROC_BACK)
    themed_style.configure("Help.PLT.Horizontal.TScrollbar", background=colours.HELP_BACKGROUND)
    themed_style.configure("Help.PLT.Vertical.TScrollbar", background=colours.HELP_BACKGROUND)

    # remove dashed line in tab headers
    themed_style.configure("Tab", focuscolor=themed_style.configure(".")["background"])

    # Entry's
    themed_style.configure('PLT.TEntry', foreground='black', background=colours.BACKGROUND)
    themed_style.configure('PL.PLT.TEntry', foreground='black', background=colours.PL_INNER)
    themed_style.configure('Eval.PLT.TEntry', foreground='black', background=colours.EVAL_INNER)

    # Combobox's
    themed_style.configure('PLT.TCombobox', foreground='black', background=colours.BACKGROUND)

    # Radiobutton's
    themed_style.configure('PLT.TRadiobutton', foreground='black', focuscolor=themed_style.configure(".")["background"])
    themed_style.map('PLT.TRadiobutton', background=[('active', colours.BACKGROUND), ('!active', colours.BACKGROUND)])

    # Progressbar
    themed_style.configure("PLT.Horizontal.TProgressbar", background=colours.PROGRESS_BACK)

    # Scale
    themed_style.configure("PLT.Horizontal.TScale", background=colours.BACKGROUND)

