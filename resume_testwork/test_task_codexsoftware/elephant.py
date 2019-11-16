import sys
import os


class Picture:
    def __init__(self, file, weight, height):
        self.file = file
        self.weight = weight # params of canvas
        self.height = height

    def draw_canvas(self): # make canvas
        self.file.write(('-' * (self.weight + 2) + '\n'))
        self.file.write(('|' + ' ' * self.weight + '|\n') * self.height)
        self.file.write('-' * (self.weight + 2) + '\n')

    def draw_line(self, x1, y1, x2, y2):
        if x1 == x2:
            self.draw_points(self.get_vertical_line(x1, y1, y2 - y1))

        elif y1 == y2:
            self.draw_points(self.get_horizontal_line(x1, y1, x2 - x1))

    def draw_rectangle(self, x1, y1, x2, y2):
        self.draw_points(self.get_horizontal_line(x1, y1, x2 - x1))
        self.draw_points(self.get_horizontal_line(x1, y2, x2 - x1))
        self.draw_points(self.get_vertical_line(x1, y1, y2 - y1))
        self.draw_points(self.get_vertical_line(x2, y1, y2 - y1))

    def bucket_fill(self, x, y, old_color, color='c'):   # autofilling realized by recursion
        self.file.seek(self.point_counter(x, y))
        if old_color == self.file.read(1):
            self.file.seek(self.point_counter(x, y))
            self.draw_points(self.get_vertical_line(x, y, 0), color)
        else:
            return

        if x > 1:
            self.bucket_fill(x - 1, y, old_color, color)  # to left

        if x < self.weight:
            self.bucket_fill(x + 1, y, old_color, color)  # to right

        if y > 1:
            self.bucket_fill(x, y - 1, old_color, color)  # to up

        if y < self.height:
            self.bucket_fill(x, y + 1, old_color, color)  # to down

    def get_horizontal_line(self, x, y, lenth): # return list of points of line
        points = []
        for i in range(lenth + 1):
            points.append(self.point_counter(x + i, y))

        return points

    def get_vertical_line(self, x, y, lenth): # return list of points of line
        points = []
        for i in range(lenth + 1):
            points.append(self.point_counter(x, y + i))

        return points

    def point_counter(self, x, y): # counting seek position by coordinates
        return (self.weight + 4) * y + x

    def draw_points(self, points, color='x'):  # this method drawind points from list
        for i in points:
            self.file.seek(i)
            self.file.write(color)


def drawing():

    # check, file is empty?
    if open('data/input.txt', 'r').read() == '':
        print("File with commands is empty!")
        return
    else:
        commands = open('data/input.txt', 'r').read().split('\n')

    # create output.txt if it doesn't exist
    if not os.path.exists('output.txt'):
        open('output.txt', 'w').close()

    output = open('output.txt', 'r+') # open output.txt

    for command in commands: # read commands from file in cycle by one string

        params = command.split(' ')
        '''
            paramns[0] - command key (C, L, R, B)
            paramns[1]/paramns[2]/paramns[3]/paramns[4] - coordinates (or color with key B)
        '''
        try:
            if params[0] == 'C':
                picture = Picture(output, int(params[1]), int(params[2])) # make class instance
                picture.draw_canvas() # draw borders of the canvas
                continue
        except:
            print("Command params aren't correct: ", command, "- Canvas didn't create.")
            continue

        try:
            if not picture: # check class instance and canvas. IF it isn't, skip command
                continue

            elif params[0] == 'L':  # check key of command
                try:
                    picture.draw_line(int(params[1]), int(params[2]), int(params[3]), int(params[4])) # call draw_line()
                except:
                    print("Command params aren't correct: ", command, '\n Please, check and repeat!')

            elif params[0] == 'R': # check key of command
                try:
                    picture.draw_rectangle(int(params[1]), int(params[2]), int(params[3]), int(params[4])) # call draw_rectangle()
                except:
                    print("Command params aren't correct: ", command, '\n Please, check and repeat!')

            elif params[0] == 'B': # check key of command
                try:
                    picture.file.seek(picture.point_counter(int(params[1]), int(params[2]))) # set market in need position
                    old_color = picture.file.read(1)
                    picture.bucket_fill(int(params[1]), int(params[2]), old_color, params[3])
                except:
                    print("Command params aren't correct: ", command, '\n Please, check and repeat!')
            else:
                print("Key of command isn't correct: ", params[0], '\nPlease, use only C, L, R or B')
        except:
            print("Canvas didn't create. Skip this command!")

    output.close()

    print("Programm finished!")


if __name__ == '__main__':
    drawing()
