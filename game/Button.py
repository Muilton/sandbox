import os
import pygame


class Button(pygame.sprite.Sprite):
    def __init__(self, x, y, text, text_color=(255,255,255), background_color=(101, 105, 79)):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([100, 60])
        self.image.fill((101, 105, 79))
        self.image = pygame.Surface([100, 60])
        self.image.fill(background_color)
        self.font = pygame.font.SysFont("Arial", 30, bold=True)
        self.text = self.font.render(text, 1, text_color)
        self.image.blit(self.text, [100 / 2 - self.text.get_width() / 2, 60 / 2 - self.text.get_height() / 2])
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, screen):
        return screen.blit(self.image, self.rect)