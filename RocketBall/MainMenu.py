import os
import pygame
from Config import *
from Button import *


class MainMenu:
    def __init__(self):
        self.btn_start = Button(screen_width / 2 - 32, 100, "Start")
        self.btn_exit = Button(screen_width / 2 - 32, 200, "Exit")

    def get_object(self):
        return self.btn_start, self.btn_exit


