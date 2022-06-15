import functools
import tkinter as tk
from threading import Thread

from TestWindow import TestWindow
from TrainTestExportWindow import TrainTestExportWindow
from BasicFunctions import *


class LoadTestWindow(tk.Toplevel):
    """
    The choose load data or test model screen
    """
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.stop_thread = False

        self.data_btn = None
        self.test_btn = None

        self.setup_window()

    def setup_window(self):
        self.geometry("200x120")  # define the window's size
        self.title("Choose load or test")

        self.data_btn = tk.Button(self, bg="Yellow", text="Click here to load the data",
                                  command=self.on_load_data)
        self.test_btn = tk.Button(self, bg="Yellow", text="Click here to test a model",
                                  command=self.on_test)

        self.data_btn.pack(pady=20)
        self.test_btn.pack()

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        menubar = tk.Menu(self)
        menubar.add_cascade(label='Exit', command=self.gui.root.destroy)
        self.config(menu=menubar)

    def enable(self):
        self.data_btn.configure(state='normal')
        self.test_btn.configure(state='normal')

    def disable(self):
        self.data_btn.configure(state='disable')
        self.test_btn.configure(state='disable')

    def on_load_data(self):
        self.disable()
        t = Thread(target=combine_functions(self.gui.data_handler.handle_dataset, self.on_thread_stop))
        t.start()
        self.wait_thread_finish(self.destroy, functools.partial(TrainTestExportWindow, self.gui, self))

    def on_test(self):
        self.disable()

        def load():
            result = self.gui.model_handler.load_model()
            if result:
                TestWindow(self.gui, self)
            else:
                self.enable()
                self.focus_force()

        t = Thread(target=load)
        t.start()

    def on_thread_stop(self):
        self.stop_thread = True

    def wait_thread_finish(self, *funcs):
        if self.stop_thread:
            self.stop_thread = False
            for func in funcs:
                func()
        else:
            self.gui.root.after(100, self.wait_thread_finish, *funcs)
