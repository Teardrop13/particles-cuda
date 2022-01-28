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
                 position_limits: tuple[float, float],
                 mass_limits: tuple[float, float],
                 speed_limits: tuple[float, float],
                 G: float,
                 dt: float) -> None:
                 
        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111, projection='3d')

        min = position_limits[0]
        max = position_limits[1]
        min_speed = speed_limits[0]
        max_speed = speed_limits[1]
        mass_min = mass_limits[0]
        mass_max = mass_limits[1]
        
        self.view_limits_x = position_limits
        self.view_limits_y = position_limits
        self.view_limits_z = position_limits
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

        positions = list(map(attrgetter('position'), self.particles))

        self.subplot.scatter(list(map(attrgetter('x'), positions)),
                             list(map(attrgetter('y'), positions)),
                             list(map(attrgetter('z'), positions)),
                             s=5,
                             c='r',
                             marker='o')

        self.subplot.set_xlim(*self.view_limits_x)
        self.subplot.set_ylim(*self.view_limits_y)
        self.subplot.set_zlim(*self.view_limits_z)

        plt.ion()
        plt.pause(0.00001)
        self.view_limits_x = self.subplot.get_xlim()
        self.view_limits_y = self.subplot.get_ylim()
        self.view_limits_z = self.subplot.get_zlim()
        self.subplot.clear()

    def initialize(self) -> None:
        if GPU:
            cuda_initialize(self.particles, self.particles_number, self.dt, self.G)
            print("gpu initialized")
        else:
            cpu_initialize(self.G, self.dt)

    def clean(self) -> None:
        if GPU:
            cuda_clean()
            print("variables freed")


if __name__ == '__main__':

    if len(sys.argv) == 2:
        mode = sys.argv[1]

        global GPU

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

    finished = False
    first_run = True

    particles_number=256
    position_limits=(-5, 5)
    mass_limits=(0.1, 1)
    speed_limits=(1, 10)
    G=10
    dt=0.001

    while first_run or input("exit? [N|y] ") != 'y':

        particles_number_str = input(f"particles number [{particles_number}]: ")
        if particles_number_str != "":
            particles_number = int(particles_number_str)

        position_limits_str = input(f"position limits [{position_limits[0]}, {position_limits[1]}]: ")
        if position_limits_str != "":
            position_limits = tuple(float(x) for x in position_limits_str.split(","))

        mass_limits_str = input(f"mass limits [{mass_limits[0]}, {mass_limits[1]}]: ")
        if mass_limits_str != "":
            mass_limits = tuple(float(x) for x in mass_limits_str.split(","))

        speed_limits_str = input(f"speed limits [{speed_limits[0]}, {speed_limits[1]}]: ")
        if speed_limits_str != "":
            speed_limits = tuple(float(x) for x in speed_limits_str.split(","))

        G_str = input(f"G [{G}]: ")
        if G_str != "":
            G = float(G_str)

        dt_str = input(f"dt [{dt}]: ")
        if dt_str != "":
            dt = float(dt_str)

        first_run = False

        simulation = Simulation(particles_number=particles_number,
                                position_limits=position_limits,
                                mass_limits=mass_limits,
                                speed_limits=speed_limits,
                                G=G,
                                dt=dt)

        simulation.initialize()

        while True:

            # if plot is still open
            if plt.get_fignums():
                simulation.run()
            else:
                break

        print("\nsimulation finished")

        simulation.clean()