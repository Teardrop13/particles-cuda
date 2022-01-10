import os
from ctypes import *
import matplotlib.pyplot as plt
from random import random


change_list = CDLL(os.path.join(os.path.abspath('.'), "move_particles.so")).change_list


class Simulation:

    def __init__(self, particles: int, min: float, max: float) -> None:
        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111, projection='3d')

        self.particles = particles
        self.min = min
        self.max = max

        position_x = []
        position_y = []
        position_z = []

        acceleration_x = []
        acceleration_y = []
        acceleration_z = []

        for i in range(particles):
            position_x.append(self._generate())
            position_y.append(self._generate())
            position_z.append(self._generate())
            acceleration_x.append(0)
            acceleration_y.append(0)
            acceleration_z.append(0)

        self.position_x = (c_float * particles)(*position_x)
        self.position_y = (c_float * particles)(*position_y)
        self.position_z = (c_float * particles)(*position_z)
        self.acceleration_x = (c_float * particles)(*acceleration_x)
        self.acceleration_y = (c_float * particles)(*acceleration_y)
        self.acceleration_z = (c_float * particles)(*acceleration_z)

    def _generate(self) -> float:
        return self.min + random() * (self.max - self.min)

    def run(self) -> None:
        change_list(self.position_x, self.position_y, self.position_z, self.acceleration_x, self.acceleration_y, self.acceleration_z, self.particles)

    def draw(self) -> None:
        self.subplot.scatter(simulation.position_x, simulation.position_y, simulation.position_z, s=20, c='r', marker='o')
        plt.ion()
        plt.pause(0.01)
        self.subplot.clear()


if __name__ == '__main__':
    simulation = Simulation(500, 0, 10)

    for i in range(4000):

        simulation.run()
        simulation.draw()
