import tkinter as tk
from threading import Thread

from BasicFunctions import *


class TestDisplayWindow(tk.Toplevel):
    """
    The display window for the prediction results
    """
    def __init__(self, gui, prev_window, run_func):
        super().__init__()

        self.gui = gui
        self.prev_window = prev_window
        self.run_func = run_func
        self.latest_img = None

        self.image_lbl = None

        self.setup_window()

    def setup_window(self):
        self.geometry("800x800")  # define the window's size
        self.title("Image test on model")

        self.image_lbl = tk.Label(self, text='Loading image...')
        self.image_lbl.pack()

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        menubar = tk.Menu(self)
        menubar.add_cascade(label='Back',
                            command=combine_functions(self.destroy, self.prev_window.enable, self.prev_window.focus_force))
        self.config(menu=menubar)
        self.focus_force()

        t = Thread(target=self.do_test)
        t.start()

    def do_test(self):
        self.latest_img = self.run_func(self.image_lbl)
        if self.latest_img is not None:
            self.geometry(f'{self.latest_img.width()}x{self.latest_img.height()}')
        elif self.winfo_exists():
            self.destroy()
            self.prev_window.enable()
            self.prev_window.focus_force()
