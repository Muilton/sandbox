import pygame
import random
import math

SCREEN_DIM = (800, 600)


class Help:
    def draw_help(self):
        gameDisplay.fill((50, 50, 50))
        self.font1 = pygame.font.SysFont("courier", 24)
        self.font2 = pygame.font.SysFont("serif", 24)
        self.data = []
        self.data.append(["F1", "Show Help"])
        self.data.append(["R", "Restart"])
        self.data.append(["P", "Pause/Play"])
        self.data.append(["Y", "More points"])
        self.data.append(["T", "Less points"])
        self.data.append(["D", "Delete last point"])
        self.data.append(["", ""])
        self.data.append([str(steps), "Current points"])
        pygame.draw.lines(gameDisplay, (255, 50, 50, 255), True, [
            (0, 0), (800, 0), (800, 600), (0, 600)], 5)
        for i, text in enumerate(self.data):
            gameDisplay.blit(self.font1.render(
                text[0], True, (128, 128, 255)), (100, 100 + 30 * i))
            gameDisplay.blit(self.font2.render(
                text[1], True, (128, 128, 255)), (200, 100 + 30 * i))


class Vec2d:
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.dx = random.random() * 2
        self.dy = random.random() * 2

    def __add__(self, vector):
        return (self.x + vector.x, self.y + vector.y)

    def __sub__(self, x):
        return (self.x - x.x, self.y - x.y)

    def __mul__(self, k):
        self.x = self.x * k
        self.y = self.y * k
        return self

    def len(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def int_par(self):
        return (int(self.x), int(self.y))


class Polyline:

    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(point)

    def set_points(self):
        for p in range(len(self.points)):
            self.points[p].x += self.points[p].dx
            self.points[p].y += self.points[p].dy

            if self.points[p].x > SCREEN_DIM[0] or self.points[p].x < 0:
                self.points[p].dx = - self.points[p].dx
            if self.points[p].y > SCREEN_DIM[1] or self.points[p].y < 0:
                self.points[p].dy = - self.points[p].dy

    def draw_points(self, count, style="points", width=3, color=(255, 255, 255)):

        if style == "line":
            points = self.get_knot(count)
            for i in range(-1, len(points) - 1):
                pygame.draw.line(gameDisplay, color, (int(points[i].x), int(points[i].y)),
                                 (int(points[i + 1].x), int(points[i + 1].y)), width)

        elif style == "points":
            for i in self.points:
                pygame.draw.circle(gameDisplay, color, (int(i.x), int(i.y)), width)


class Knot(Polyline):

    def get_knot(self, count=35):
        if len(self.points) < 3:
            return self.points
        result = []
        for i in range(-2, len(self.points) - 2):
            ptn = []
            ptn.append(Vec2d(self.points[i] + self.points[i + 1]) * 0.5)
            ptn.append(self.points[i + 1])
            ptn.append(Vec2d(self.points[i + 1] + self.points[i + 2]) * 0.5)

            result.extend(self.__get_points(ptn, count))
        return result

    def __get_points(self, ptn, count):
        alpha = 1 / count
        result = []
        for i in range(count):
            result.append(self.__get_point(ptn, i * alpha))
        return result

    def __get_point(self, ptn, alpha, deg=None):
        if deg is None:
            deg = len(ptn) - 1
        if deg == 0:
            return ptn[0]
        return Vec2d((ptn[deg].x * alpha + self.__get_point(ptn, alpha, deg - 1).x * (1 - alpha),
                      ptn[deg].y * alpha + self.__get_point(ptn, alpha, deg - 1).y * (1 - alpha)))

    def delete_point(self):
        self.points = self.points[:-1]

    def speed_of_point(self, key):
        for i in self.points:
            if key == "up":
                i.dx = i.dx * 1.1
                i.dy = i.dy * 1.1
            elif key == "down":
                i.dx = i.dx * 0.9
                i.dy = i.dy * 0.9


# Основная программа
if __name__ == "__main__":
    pygame.init()
    gameDisplay = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("MyScreenSaver")

    steps = 35
    working = True
    points = []
    speeds = []
    show_help = False
    pause = True
    line = Polyline()
    knot = Knot()
    help = Help()


    hue = 0
    color = pygame.Color(0)

    while working:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                working = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    working = False
                if event.key == pygame.K_r:
                    knot.points = []
                if event.key == pygame.K_p:
                    pause = not pause
                if event.key == pygame.K_y:
                    steps += 1
                if event.key == pygame.K_F1:
                    show_help = not show_help
                if event.key == pygame.K_t:
                    steps -= 1 if steps > 1 else 0
                if event.key == pygame.K_d:
                    knot.delete_point()
                if event.key == pygame.K_g:
                    knot.speed_of_point("up")
                if event.key == pygame.K_h:
                    knot.speed_of_point("down")

            if event.type == pygame.MOUSEBUTTONDOWN:
                knot.add_point(Vec2d(event.pos))

        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        color.hsla = (hue, 100, 50, 100)
        knot.draw_points(steps)
        knot.draw_points(steps, "line", 3, color)
        if not pause:
            knot.set_points()
        if show_help:
            help.draw_help()

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
