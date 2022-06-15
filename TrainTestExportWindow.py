import tkinter as tk
from threading import Thread

from BasicFunctions import *
from TestWindow import TestWindow


class TrainTestExportWindow(tk.Toplevel):
    """
    The choose train, test or export model
    """
    def __init__(self, gui, prev_window):
        super().__init__()

        self.gui = gui
        self.prev_window = prev_window
        self.stop_thread = False

        self.new_model_btn = None
        self.test_existing_btn = None
        self.export_model_btn = None

        self.setup_window()

    def setup_window(self):
        self.geometry("230x200")  # define the window's size
        self.title("Choose train test or export")

        self.new_model_btn = tk.Button(self, bg="Yellow", text="Click here to train a new model",
                                       command=self.on_new_model)
        self.test_existing_btn = tk.Button(self, bg="Yellow", text="Click here to test an existing model",
                                           command=self.on_test_existing)
        self.export_model_btn = tk.Button(self, bg="Yellow", text="Click here to export a model",
                                          command=self.on_export)

        self.new_model_btn.pack(pady=20)
        self.test_existing_btn.pack()
        self.export_model_btn.pack(pady=20)

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        menubar = tk.Menu(self)
        menubar.add_cascade(label='Exit', command=self.gui.root.destroy)
        self.config(menu=menubar)
        self.focus_force()

    def enable(self):
        self.new_model_btn.configure(state='normal')
        self.test_existing_btn.configure(state='normal')
        self.export_model_btn.configure(state='normal')

    def disable(self):
        self.new_model_btn.configure(state='disable')
        self.test_existing_btn.configure(state='disable')
        self.export_model_btn.configure(state='disable')

    def on_new_model(self):
        self.disable()
        t = Thread(target=combine_functions(self.gui.model_handler.train_model, self.on_thread_stop))
        t.start()

        self.wait_thread_finish(self.enable)

    def on_test_existing(self):
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

    def on_export(self):
        self.disable()
        t = Thread(target=combine_functions(self.gui.model_handler.export_model, self.on_thread_stop))
        t.start()

        self.wait_thread_finish(self.enable)

    def on_thread_stop(self):
        self.stop_thread = True

    def wait_thread_finish(self, *funcs):
        if self.stop_thread:
            self.stop_thread = False
            for func in funcs:
                func()
        else:
            self.gui.root.after(100, self.wait_thread_finish, *funcs)
