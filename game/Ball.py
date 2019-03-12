import os
import pygame


class Ball(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.dx = 1
        self.dy = 1
        self.x = 0
        self.y = 0
        self.speed = 2
        self.radius = 15

        self.image = pygame.image.load(os.path.join(os.path.dirname(__file__), "ball.png")).convert()

        self.rect = self.image.get_rect()

    def draw(self, screen):
        return screen.blit(self.image, self.rect)

    def move(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed

        if self.x >= 0:
            if self.x // 1 != 0: self.rect.centerx += self.x // 1
            self.x = self.x % 1
            # print("Вектор dX: ", self.dx,"Счетчик координат X: ", self.x, " Текущая координата Х: ", self.rect.centerx)
        else:
            if abs(self.x) // 1 != 0: self.rect.centerx += (abs(self.x) // 1) * -1
            self.x = (abs(self.x) % 1) * -1
            # print("Вектор dX: ", self.dx, "Счетчик координат X: ", self.x, " Текущая координата Х: ", self.rect.centerx)

        if self.y >= 0:
            if self.y // 1 != 0: self.rect.centery += self.y // 1
            self.y = self.y % 1
            # print("Вектор dY: ", self.dy, "Счетчик координат Y: ", self.y, " Текущая координата Y: ", self.rect.centery)
        else:
            if abs(self.y) // 1 != 0: self.rect.centery += (abs(self.y) // 1) * -1
            self.y = (abs(self.y) % 1) * -1
            # print("Вектор dY: ", self.dy, "Счетчик координат Y: ", self.y, " Текущая координата Y: ", self.rect.centery)


        # self.rect.centerx += self.dx * self.speed
        # self.rect.centery += (self.dy/self.dx)*self.dx