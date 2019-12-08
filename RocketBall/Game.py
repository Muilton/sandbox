import sys
import pygame
from MainMenu import *
from Level import *
from BackMenu import *


class Game:

    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(screen_size)
        self.level = Level()
        self.main_menu = MainMenu()
        self.back_menu = BackMenu()

    def menu(self):
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.main_menu.btn_start.rect.collidepoint(pygame.mouse.get_pos()):
                        self.start()
                    elif event.button == 1 and self.main_menu.btn_exit.rect.collidepoint(pygame.mouse.get_pos()):
                        pygame.quit()
                        sys.exit()

            self.main_menu.get_object()  # получаем объекты меню
            self.screen.fill((0, 0, 0))  # фон окна

            # отрисовываем элементы
            self.main_menu.btn_start.draw(self.screen)
            self.main_menu.btn_exit.draw(self.screen)

            pygame.display.flip()  # обновляем окно

    def start(self):
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

            # self.level.ball.rect = self.level.ball.rect.move([self.level.ball.dx, self.level.ball.dy])
            self.level.ball.move()
            self.level.collision()

            self.level.step()

            self.clock.tick(fps)

            self.level.get_object()  # получаем объекты
            self.screen.fill((0, 0, 0))  # фон окна

            # отрисовываем элементы
            self.level.ball.draw(self.screen)
            self.level.platform.draw(self.screen)
            self.level.bricks_list.draw(self.screen)

            pygame.display.flip()  # обновляем окно


if __name__ == "__main__":
    player = Game()
    player.menu()