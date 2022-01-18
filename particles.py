import os
import sys
from ctypes import *
import matplotlib.pyplot as plt
from random import random
from operator import attrgetter


class Vector(Structure):
    _fields_ = [("x", c_float),
                ("z", c_float),
                ("y", c_float)]

    def __init__(self, x: float, y: float, z: float) -> None:
        super(Vector, self).__init__(x, y, z)


class Particle(Structure):
    _fields_ = [("position", Vector),
                ("speed", Vector),
                ("mass", c_float)]

    def __init__(self, x: float, y: float, z: float, x_speed: float, y_speed: float, z_speed: float, mass: float) -> None:
        super(Particle, self).__init__(position=Vector(x, y, z),
                                       speed=Vector(x_speed, y_speed, z_speed), mass=mass)


def generate(min: float, max: float) -> float:
    return min + random() * (max - min)


class Simulation:

    def __init__(self,
                 particles_number: int,
                 min: float,
                 max: float,
                 mass_min: float,
                 mass_max: float,
                 min_speed: float,
                 max_speed: float,
                 view_min: float,
                 view_max: float,
                 G: float,
                 dt: float) -> None:
                 
        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111, projection='3d')
        self.min = min
        self.max = max
        self.view_min = view_min
        self.view_max = view_max
        self.particles_number = c_int(particles_number)

        particles = []

        for i in range(particles_number):
            particles.append(Particle(x=generate(min, max),
                                      y=generate(min, max),
                                      z=generate(min, max),
                                      x_speed=generate(min_speed, max_speed),
                                      y_speed=generate(min_speed, max_speed),
                                      z_speed=generate(min_speed, max_speed),
                                      mass=generate(mass_min, mass_max)))

        self.particles = (Particle * particles_number)(*particles)

        self.dt = c_float(dt)
        self.G = c_float(G)

    def run(self) -> None:
        if GPU:
            move_particles(self.particles)
        else:
            move_particles(self.particles, self.particles_number)

    def draw(self) -> None:
        positions = list(map(attrgetter('position'), self.particles))

        self.subplot.scatter(list(map(attrgetter('x'), positions)),
                             list(map(attrgetter('y'), positions)),
                             list(map(attrgetter('z'), positions)),
                             s=5,
                             c='r',
                             marker='o')

        self.subplot.set_xlim(self.view_min, self.view_max)
        self.subplot.set_ylim(self.view_min, self.view_max)
        self.subplot.set_zlim(self.view_min, self.view_max)

        plt.ion()
        plt.pause(0.0001)
        # plt.pause(1)
        self.subplot.clear()

    def cpu_initialize(self) -> None:
        cpu_initialize(self.G, self.dt)

    def cuda_initialize(self) -> None:
        cuda_initialize(self.particles, self.particles_number, self.dt, self.G)
        print("gpu initialized")

    def cuda_clean(self) -> None:
        cuda_clean()
        print("variables freed")


if __name__ == '__main__':

    if len(sys.argv) == 2:
        mode = sys.argv[1]

        if mode == '--cpu':
            GPU = False
            library = CDLL(os.path.join(
                os.path.abspath('.'), "move_particles_cpu.so"))
            move_particles = library.move_particles
            cpu_initialize = library.cpu_initalize
        elif mode == '--gpu':
            GPU = True
            library = CDLL(os.path.join(
                os.path.abspath('.'), "move_particles_gpu.so"))
            move_particles = library.move_particles
            cuda_initialize = library.cuda_initialize
            cuda_clean = library.cuda_clean
        else:
            print('wrong argument')
            exit()

    else:
        print('wrong argument')
        exit()

    simulation = Simulation(particles_number=128,
                            G=1,
                            min=-50,
                            max=50,
                            mass_min=1,
                            mass_max=2,
                            min_speed=1,
                            max_speed=3,
                            view_min=-200,
                            view_max=200,
                            dt=0.1)

    if GPU:
        simulation.cuda_initialize()
    else:
        simulation.cpu_initialize()

    simulation.draw()
    # print('cpu: ' + simulation.particles[0].position.x)
    for i in range(4000):

        simulation.run()
        simulation.draw()
        # print(f' python: {simulation.particles[0].speed.x}')

    if GPU:
        simulation.cuda_clean()