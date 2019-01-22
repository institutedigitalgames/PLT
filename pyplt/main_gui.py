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

"""This module runs the graphical user interface (GUI) of PLT.

The root widget of the GUI is a :class:`pyplt.gui.mainmenu.MainMenu` widget. The rest of the GUI is managed by the
:mod:`pyplt.gui` subpackage.
"""

from pyplt.gui.util import colours

if __name__ == '__main__':
    import os
    from pyplt import ROOT_PATH
    import tkinter as tk
    from ttkthemes import ThemedStyle
    from tkinter import font
    from pyplt.gui.mainmenu import MainMenu
    import pyplt.gui.util.styles as styles

    # Graphical User Interface (GUI) version
    root = tk.Tk()
    root.style = ThemedStyle(root)  # Like other Tkinter classes, a Style can take a master argument
    root.style.set_theme("arc")

    root.title("Preference Learning Toolbox")
    img = tk.PhotoImage(file=os.path.join(ROOT_PATH, 'assets/plt_logo.png'))
    root.tk.call('wm', 'iconphoto', root._w, img)

    print("PLT  Copyright (C) 2018  Institute of Digital Games, University of Malta \n" +
          "This program comes with ABSOLUTELY NO WARRANTY. \n" +
          "This is free software, and you are welcome to redistribute it \n" +
          "under certain conditions. \n" +
          "For more details see the GNU General Public License v3 which may be \n" +
          "downloaded at http://plt.institutedigitalgames.com/download.php.")

    # set default font to Ebrima
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family='Ebrima', size=10, weight=font.NORMAL)  # i.e., ebrima_small
    root.option_add("*Font", default_font)

    # set default widget background to colours.BACKGROUND
    root.tk_setPalette(background=colours.BACKGROUND)

    # configure ttk styles
    styles.configure_styles(root.style)

    gui = MainMenu(root)

    # place window in the center of the top left quadrant of the screen
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    # print("screen_width: " + str(screen_width))
    screen_height = root.winfo_screenheight()
    win_width = root.winfo_width()
    # print("win_width: " + str(win_width))
    win_height = root.winfo_height()
    top_left_center_x = screen_width/4
    # print("top_left_center_x: " + str(top_left_center_x))
    top_left_center_y = screen_height/4
    # print("win_width/2: " + str(win_width/2))
    top_left_x = top_left_center_x - win_width/2
    # print("top_left_x: " + str(top_left_x))
    top_left_y = top_left_center_y - win_height/2
    geom = "+%d+%d" % (top_left_x, top_left_y)
    root.wm_geometry(geom)

    root.mainloop()
