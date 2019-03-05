"""
ICE NUCLEATION DETECTOR WITH GUI
"""

import re
import os

import tkinter as tk
import cv2
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

os.chdir("c:/Users/fc12293/ice_pics")

def tryint(s):#This is for sorting the files into human numerical order
    try: return int(s)
    except: return s

def alphanum_key(s):#This is for sorting the files into human numerical order
    return [tryint(c) for c in re.split("([0-9]+)", s)]

##Sort files##############
file_extension = ".tiff"

os.chdir("c:/Users/fc12293/ice_pics/")
files = [i for i in os.listdir() if i[-len(file_extension):] == file_extension]
files.sort(key = alphanum_key)#Human sort of numbers 1->2->3 instead of 1->11->12...
##########################

LARGE_FONT = ("Verdana", 12)

img = cv2.imread(files[0])

class App:
    def __init__(self, window, window_title, img):
        self.window = window
        self.window.title = window_title

        self.cv_img = img
        cv2.resize(self.cv_img, (500,300))
        photo = ImageTk.PhotoImage(Image.fromarray(img))

        self.height, self.width, _ = self.cv_img.shape

        self.canvas = tk.Canvas(window, width=self.width,
                                height=self.height)
        self.canvas.pack()
        

class Mainframe(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.menu = menu(parent)
        self.menu.grid(column = 0, row = 0)
        global img
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.image = tk.Label(self, image=photo)
        self.image.grid(column = 1, row = 0)
        self.image.grid(columnspan=5)

class menu(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        b1 = tk.Button(self, text = "Button 1")
        b1.pack()

        b2 = tk.Button(self, text = "Button 2")
        b2.pack()

        b3 = tk.Button(self, text = "Button 3")
        b3.pack()

class image(tk.Frame):
    def __init__(self, parent, img):
        tk.Frame.__init__(self, parent)

        self.canvas = tk.Label(self)
        
        self.display_image(img)
        self.canvas.pack()

    def display_image(self, img):
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.image = img
        
        

root = tk.Tk()
root.title("Feet to metres")

app = Mainframe(root)

root.mainloop()
        
        
        
