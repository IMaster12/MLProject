import tkinter as tk

from LoadTestWindow import LoadTestWindow
from data_handler import DataHandler
from model_handler import ModelHandler
from PIL import Image, ImageOps, ImageTk


class ProjectGui:
    """
    The main project gui window
    """
    def __init__(self):
        self.root = tk.Tk()
        self.data_handler = DataHandler(self)
        self.model_handler = ModelHandler(self, self.data_handler)

        self.__btn = None

    def main_window(self):
        self.root.geometry('400x450')
        self.root.title('Main Menu')
        tk.Frame(master=self.root, width=340, height=20, bg="White").pack(fill=tk.X)
        tk.Message(self.root, width=310, bg="Light blue",
                   text='\n'.join(['Hello!',
                                   'Once you hit start windows will pop up, you can choose which actions to take',
                                   'Here are some general insturctions:',
                                   'First, You either load the data or test an exported model',
                                   'From there you will be guided through the process of running the project',
                                   'In some windows you can go back to the previous window by using the Back button',
                                   'In some windows you can exit the project by pressing Exit',
                                   'In order to start press the green button below'])).pack()
                   # text='\n'.join(['Hello!',
                   #                 'You are going to run the project.',
                   #                 '\nSoon when you start you will see some windows one after one, please do what they say.',
                   #                 'First, you will load the data.',
                   #                 'Then, you will train the model.',
                   #                 'After that you will test it.',
                   #                 'At the end you will choose the option you want.',
                   #                 '\nIn order to start please press on the button down here.'])).pack()
        try:
            img = Image.open('resources/coverimage.png')
            img = ImageOps.contain(img, (300, 300))
            photo = ImageTk.PhotoImage(img)
            tk.Label(self.root, image=photo).pack()
        except Exception as e:
            pass
        self.__btn = tk.Button(self.root, bg="green", text='Click here to start',
                               command=self.on_btn)  # button to start - opens the data window

        self.__btn.pack()
        self.root.protocol("WM_DELETE_WINDOW",
                           self.root.iconify)  # make the top right close button (X) minimize (iconify) the window/form
        self.root.resizable(False, False)
        # create a menu bar with an Exit command
        menubar = tk.Menu(self.root)
        menubar.add_cascade(label='Exit', command=self.root.destroy)
        self.root.config(menu=menubar)
        self.root.focus_force()
        self.root.mainloop()

    def disable(self):
        self.__btn.configure(state='disable')

    def on_btn(self):
        self.disable()
        LoadTestWindow(self)



