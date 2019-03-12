import os
import pygame
from Config import *


class Platform(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(os.path.join(os.path.dirname(__file__), "platform.png")).convert()
        self.rect = self.image.get_rect()

        self.radius = 75

        self.rect.x = screen_width / 2 - self.image.get_width() / 2 + 100
        self.rect.y = screen_height - self.image.get_height()/4

    def draw(self, screen):
        return screen.blit(self.image, self.rect)

    def move(self, step):
        self.rect.left += step * 2