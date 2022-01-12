import os
from ctypes import *
import matplotlib.pyplot as plt
from random import random

GPU = False

if GPU:
    library = CDLL(os.path.join(os.path.abspath('.'), "move_particles_gpu.so"))
    move_particles = library.move_particles
    cuda_initialize = library.cuda_initialize
    cuda_clean = library.cuda_clean
else:
    library = CDLL(os.path.join(os.path.abspath('.'), "move_particles.so"))
    move_particles = library.move_particles


class Simulation:

    def __init__(self, particles: int, min: float, max: float, mass_min: float, mass_max: float) -> None:
        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111, projection='3d')
        self.min = min
        self.max = max
        self.scaling = 5
        self.particles = particles

        position_x = []
        position_y = []
        position_z = []

        acceleration_x = []
        acceleration_y = []
        acceleration_z = []

        mass = []

        for i in range(particles):
            position_x.append(self._generate(min, max))
            position_y.append(self._generate(min, max))
            position_z.append(self._generate(min, max))

            acceleration_x.append(0)
            acceleration_y.append(0)
            acceleration_z.append(0)

            mass.append(self._generate(mass_min, mass_max))

        self.position_x = (c_float * particles)(*position_x)
        self.position_y = (c_float * particles)(*position_y)
        self.position_z = (c_float * particles)(*position_z)

        self.acceleration_x = (c_float * particles)(*acceleration_x)
        self.acceleration_y = (c_float * particles)(*acceleration_y)
        self.acceleration_z = (c_float * particles)(*acceleration_z)
        
        self.mass = (c_float * particles)(*mass)

    def _generate(self, min: float, max: float) -> float:
        return min + random() * (max - min)

    def run(self) -> None:
        move_particles(self.position_x, self.position_y, self.position_z, self.acceleration_x, self.acceleration_y, self.acceleration_z, self.mass, self.particles)

    def draw(self) -> None:
        self.subplot.scatter(simulation.position_x, simulation.position_y, simulation.position_z, s=20, c='r', marker='o')
        self.subplot.set_xlim(self.min, self.max*self.scaling)
        self.subplot.set_ylim(self.min, self.max*self.scaling)
        self.subplot.set_zlim(self.min, self.max*self.scaling)
        plt.ion()
        plt.pause(0.01)
        self.subplot.clear()

    def cuda_initialize(self) -> None:
        cuda_initialize(self.position_x, self.position_y, self.position_z, self.acceleration_x, self.acceleration_y, self.acceleration_z, self.mass, self.particles)

    def cuda_clean(self) -> None:
        cuda_clean()

if __name__ == '__main__':
    simulation = Simulation(200, 0, 10, 0.1, 10)

    if GPU:
        simulation.cuda_initialize()

    for i in range(4000):

        simulation.run()
        simulation.draw()

    if GPU:
        simulation.cuda_clean()