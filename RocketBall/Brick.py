import pygame
import random
from Config import *


class Brick(pygame.sprite.Sprite):
    def __init__(self, color=(255, 255, 255), width: object = 40, height: object = 20):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        self.rect = self.image.get_rect()

        self.rect.x = random.randrange(screen_width - 40)
        self.rect.y = random.randrange(screen_height / 2)

        print(self.rect, "брик")

