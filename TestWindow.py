import tkinter as tk
from threading import Thread

from BasicFunctions import *
from TestDisplayWindow import TestDisplayWindow


class TestWindow(tk.Toplevel):
    """
    The test model window
    """
    def __init__(self, gui, prev_window):
        super().__init__()

        self.gui = gui
        self.prev_window = prev_window
        self.stop_thread = False

        self.test_image_btn = None
        self.test_video_btn = None
        self.test_camera_btn = None

        self.setup_window()

    def setup_window(self):
        self.geometry("200x200")  # define the window's size
        self.title("Test Model")

        self.test_image_btn = tk.Button(self, bg="Yellow", text="Click here to test on an image",
                                        command=self.on_test_image)
        self.test_video_btn = tk.Button(self, bg="Yellow", text="Click here to test on video",
                                        command=self.on_test_video)
        self.test_camera_btn = tk.Button(self, bg="Yellow", text="Click here to test on live camera",
                                         command=self.on_test_live)

        self.test_image_btn.pack(pady=20)
        self.test_video_btn.pack()
        self.test_camera_btn.pack(pady=20)

        self.protocol("WM_DELETE_WINDOW",
                      self.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.resizable(False, False)
        # create a menu bar with an Exit command
        menubar = tk.Menu(self)
        menubar.add_cascade(label='Back', command=combine_functions(self.destroy, self.prev_window.enable, self.prev_window.focus_force))
        self.config(menu=menubar)
        self.focus_force()

    def enable(self):
        self.test_image_btn.configure(state='normal')
        self.test_video_btn.configure(state='normal')
        self.test_camera_btn.configure(state='normal')

    def disable(self):
        self.test_image_btn.configure(state='disable')
        self.test_video_btn.configure(state='disable')
        self.test_camera_btn.configure(state='disable')

    def on_test_image(self):
        self.disable()
        TestDisplayWindow(self.gui, self, self.gui.model_handler.test_on_image)

    def on_test_video(self):
        self.disable()
        t = Thread(target=combine_functions(self.gui.model_handler.test_on_video, self.on_thread_stop))
        t.start()
        self.wait_thread_finish(self.enable)

    def on_test_live(self):
        self.disable()
        TestDisplayWindow(self.gui, self, self.gui.model_handler.test_on_live)

    def on_thread_stop(self):
        self.stop_thread = True

    def wait_thread_finish(self, *funcs):
        if self.stop_thread:
            self.stop_thread = False
            for func in funcs:
                func()
        else:
            self.gui.root.after(100, self.wait_thread_finish, *funcs)
