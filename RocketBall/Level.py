import pygame
import sys
from Config import *
from Brick import Brick
from Ball import Ball
from Platform import Platform
import math


class Level:
    def __init__(self):
        self.bricks_list = pygame.sprite.Group()
        self.ball = Ball()
        self.platform = Platform()
        self.collision_list = None
        self.collision_brick = None
        self.make_bricks(10)

    def make_bricks(self, count):
        for _ in range(count):
            br = Brick()
            self.bricks_list.add(br)

    def reset(self):
        pass

    def step(self):
        if pygame.key.get_pressed()[pygame.K_LEFT] and self.platform.rect.left > 0:
            self.platform.move(-2)
        if pygame.key.get_pressed()[pygame.K_RIGHT] and self.platform.rect.right < screen_width:
            self.platform.move(2)

    def collision(self):
            # отсткок от стенок
        if self.ball.rect.left <= 0 and self.ball.dx < 0:
            self.ball.dx = -self.ball.dx
        if self.ball.rect.right > screen_width and self.ball.dx > 0:
            self.ball.dx = -self.ball.dx
        if self.ball.rect.top <= 0 and self.ball.dy < 0:
            self.ball.dy = -self.ball.dy

        if self.ball.rect.centery >= screen_height:
            sys.exit("GAME OVER")


            # отскок от платформы
        # if self.ball.rect.bottom == self.platform.rect.top:
        #     if ((self.ball.rect.left < self.platform.rect.left and self.ball.rect.right < self.platform.rect.left) or
        #        (self.ball.rect.left > self.platform.rect.right and self.ball.rect.right > self.platform.rect.right)):
        #         sys.exit("ПОТРАЧЕНО")
        #     else:
        #         self.ball.dy = - self.ball.dy

            # кирпичи закончились
        if len(self.bricks_list) == 0:
            sys.exit("Вы выиграли!!!")

        if math.sqrt((self.platform.rect.centerx - self.ball.rect.centerx) ** 2 + (self.ball.rect.centery - self.platform.rect.centery) ** 2)  < (self.ball.radius + self.platform.radius):
                # and math.sqrt((self.platform.rect.centerx - (self.ball.rect.centerx - self.ball.dx)) ** 2 + (self.platform.rect.centery - (self.ball.rect.centery - self.ball.dy)) ** 2) >= (self.ball.radius + self.platform.radius):

            vector_x = self.platform.rect.centerx - self.ball.rect.centerx  # нормаль по Х
            vector_y = self.platform.rect.centery - self.ball.rect.centery  # нормаль по Y
            piece_of_radius = self.ball.radius/(self.ball.radius + self.platform.radius) # скаляр вектора-нормали
            point = [(self.ball.rect.centerx + vector_x * piece_of_radius), (self.ball.rect.centery + vector_y * piece_of_radius)]   # точка касания окружностей


                # коэффицинты уравнения касательной
            # A = self.ball.rect.centery - point[1]
            # B = point[0] - self.ball.rect.centerx
            # C = A * self.ball.rect.centerx + B * self.ball.rect.centery + self.ball.radius**2
            # # print(f"Уравнение касательной: {A}*x + {B}*y - {C}")# коэффицинты уравнения касательной

            A = self.ball.rect.centery - self.platform.rect.centery
            B = self.platform.rect.centerx - self.ball.rect.centerx
            C = A * self.ball.rect.centerx + B * self.ball.rect.centery + self.ball.radius**2
            # print(f"Уравнение касательной: {A}*x + {B}*y - {C}")

                # расчет прямой по вектору направления
            point_second_x = self.ball.rect.centerx + self.ball.dx
            point_second_y = self.ball.rect.centery + self.ball.dy

            A_v = self.ball.rect.centery - point_second_y
            B_v = point_second_x - self.ball.rect.centerx
            C_v = self.ball.rect.centerx * point_second_y - self.ball.rect.centery * point_second_x



            # print(f"Уравнение вектора : {A_v}*x + {B_v}*y - {C_v}")

            cos_angle = (A*A_v + B*B_v)/ ((A**2 + B**2)**0.5 * (A_v**2 + B_v**2)**0.5)

            print("collision", self.ball.rect.center, self.platform.rect.center, [A, B, C], [A_v, B_v, C_v], cos_angle)
            # print("Косинус угла: ", cos_angle, math.degrees(math.acos(cos_angle)))
            radius_by_vector = (self.ball.dx**2 + self.ball.dy**2)**0.5
            print("Старая скорость: ", self.ball.dx, self.ball.dy, self.ball.rect.center)
            self.ball.dx = radius_by_vector * math.sin(math.radians(90+math.degrees(math.acos(cos_angle))))
            self.ball.dy = radius_by_vector * math.cos(math.radians(90+math.degrees(math.acos(cos_angle))))
            print("Новая скорость: ", self.ball.dx, self.ball.dy)

            self.ball.x = self.ball.x - 5
            self.ball.y = self.ball.y - 5

        self.collision_list = pygame.sprite.spritecollide(self.ball, self.bricks_list, True)

        if len(self.collision_list) != 0:
            for self.collision_brick in self.collision_list:
                if self.ball.dx > 0 and self.ball.dy > 0:
                    if self.collision_brick.rect.topleft[0] - self.ball.rect.centerx < self.collision_brick.rect.topleft[1] - self.ball.rect.centery:
                        print("Сверху")
                        self.ball.dy = -self.ball.dy
                    else:
                        print("Слева")
                        self.ball.dx = -self.ball.dx
                if self.ball.dx < 0 and self.ball.dy < 0:
                    if self.collision_brick.rect.bottomright[0] - self.ball.rect.centerx < self.collision_brick.rect.bottomright[1] - self.ball.rect.centery:
                        print("Снизу")
                        self.ball.dy = -self.ball.dy
                    else:
                        print("Справа")
                        self.ball.dx = -self.ball.dx
                if self.ball.dx > 0 and self.ball.dy < 0:
                    if self.collision_brick.rect.bottomleft[0] - self.ball.rect.centerx < self.collision_brick.rect.bottomleft[1] - self.ball.rect.centery:
                        print("Снизу")
                        self.ball.dy = -self.ball.dy
                    else:
                        print("Слева")
                        self.ball.dx = -self.ball.dx
                if self.ball.dx < 0 and self.ball.dy > 0:
                    if self.collision_brick.rect.topright[0] - self.ball.rect.centerx < self.collision_brick.rect.topright[1] - self.ball.rect.centery:
                        print("Сверху")
                        self.ball.dy = -self.ball.dy
                    else:
                        print("Справа")
                        self.ball.dx = -self.ball.dx




                # if self.ball.rect.bottomright[1] >= self.collision_brick.rect.topleft[1]+1 and self.ball.rect.bottomright[0] > self.collision_brick.rect.topleft[0] and self.ball.rect.bottomleft[0] < self.collision_brick.rect.topright[0]:
                #     print("Сверху")
                #     self.ball.dy = -self.ball.dy
                # if self.ball.rect.topright[1] <= self.collision_brick.rect.bottomleft[1]+1 and self.ball.rect.topright[0] > self.collision_brick.rect.bottomleft[0] and self.ball.rect.topleft[0] < self.collision_brick.rect.bottomright[0]:
                #     print("Снизу")
                #     self.ball.dy = -self.ball.dy
                # if self.ball.rect.topright[0] >= self.collision_brick.rect.bottomleft[0]-1 and self.ball.rect.topright[1] < self.collision_brick.rect.bottomleft[1] and self.ball.rect.bottomright[1] > self.collision_brick.rect.topleft[1]:
                #     print("Слева")
                #     self.ball.dx = -self.ball.dx
                # if self.ball.rect.topleft[0] <= self.collision_brick.rect.topright[0]-1 and self.ball.rect.topleft[1] < self.collision_brick.rect.bottomright[1] and self.ball.rect.bottomleft[1] > self.collision_brick.rect.topright[1]:
                #     print("Справа")
                #     self.ball.dx = -self.ball.dx

    def get_object(self):
        return self.ball, self.platform, self.bricks_list