class Light:
    def __init__(self, dim=(2, 2)):
        self.dim = dim
        self.grid = [[0 for i in range(dim[0])] for _ in range(dim[1])]
        self.lights = []
        self.obstacles = []

    def set_dim(self, dim):
        self.dim = dim
        self.grid = [[0 for i in range(dim[0])] for _ in range(dim[1])]

    def set_lights(self, lights):
        self.lights = lights
        self.generate_lights()

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        self.generate_lights()

    def generate_lights(self):
        return self.grid.copy()


class System:
    def __init__(self):
        self.map = self.grid = [[0 for i in range(22)] for _ in range(18)]
        self.map[5][7] = 1  # Источники света
        self.map[3][8] = 1  # Источники света
        self.map[5][2] = -1  # Стены
        self.map[1][9] = -1  # Стены

    def get_lightening(self, light_mapper):
        self.lightmap = light_mapper.lighten(self.map)


class MappingAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def lighten(self, grid):
        shape = self.find_shape_of_map(grid)  # shape of map
        lights = self.find_light(grid)  # list of source of lights
        obstacles = self.find_obstacle(grid)  # list of obstacles
        self.adaptee.set_dim(shape)
        self.adaptee.set_lights(lights)
        self.adaptee.set_obstacles(obstacles)

        return self.adaptee.generate_lights()

    def find_light(self, grid):
        source_of_light = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    source_of_light.append((i, j))
        return source_of_light

    def find_obstacle(self, grid):
        source_of_obstacle = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == -1:
                    source_of_obstacle.append((i, j))
        return source_of_obstacle

    def find_shape_of_map(self, grid):
        return (len(grid[0]), len(grid))


# для проверки работоспособности нашего класса-адаптера

# light = Light()
# system = System()
# adapter = MappingAdapter(light)
# system.get_lightening(adapter)
# system.lightmap, system.map