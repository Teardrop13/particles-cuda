import os
from ctypes import *
import matplotlib.pyplot as plt
from random import random
from operator import attrgetter

GPU = True

if GPU:
    library = CDLL(os.path.join(os.path.abspath('.'), "move_particles_gpu.so"))
    move_particles = library.move_particles
    cuda_initialize = library.cuda_initialize
    cuda_clean = library.cuda_clean
else:
    library = CDLL(os.path.join(os.path.abspath('.'), "move_particles.so"))
    move_particles = library.move_particles


class Vector(Structure):
    _fields_ = [("x", c_float),
                ("z", c_float),
                ("y", c_float)]
    def __init__(self, x: float, y: float, z: float) -> None:
        super(Vector, self).__init__(x, y, z)


class Particle(Structure):
    _fields_ = [("x", c_float),
                 ("y", c_float),
                 ("z", c_float),
                 ("mass", c_float),
                 ("acceleration", Vector)]

    def __init__(self, x: float, y: float, z: float, mass: float) -> None:
        super(Particle, self).__init__(x, y, z, mass, Vector(0,0,0))


class Simulation:

    def __init__(self, particles_number: int, min: float, max: float, mass_min: float, mass_max: float) -> None:
        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111, projection='3d')
        self.min = min
        self.max = max
        self.scaling = 5
        self.particles_number = particles_number

        step = 0.01
        G = 2

        particles = []

        for i in range(particles_number):
            particles.append(Particle(self._generate(min, max),
                                        self._generate(min, max),
                                        self._generate(min, max),
                                        self._generate(mass_min, mass_max)))

        self.particles = (Particle * particles_number)(*particles)
        
        self.step = c_float(step)
        self.G = c_float(G)

    def _generate(self, min: float, max: float) -> float:
        return min + random() * (max - min)

    def run(self) -> None:
        if GPU:
            move_particles(self.particles)
        else:
            move_particles(self.position_x, self.position_y, self.position_z, self.acceleration_x, self.acceleration_y, self.acceleration_z, self.mass, self.particles)

    def draw(self) -> None:
        self.subplot.scatter(map(attrgetter('x'), self.particles),
                            map(attrgetter('y'), self.particles),
                            map(attrgetter('z'), self.particles), 
                            s=20, 
                            c='r',
                            marker='o')
        # self.subplot.set_xlim(self.min, self.max*self.scaling)
        # self.subplot.set_ylim(self.min, self.max*self.scaling)
        # self.subplot.set_zlim(self.min, self.max*self.scaling)
        plt.ion()
        plt.pause(0.0001)
        self.subplot.clear()

    def cuda_initialize(self) -> None:
        cuda_initialize(self.particles, self.step, self.particles, self.G)

    def cuda_clean(self) -> None:
        cuda_clean()

if __name__ == '__main__':
    simulation = Simulation(128, 0, 10, 0.1, 2)

    if GPU:
        simulation.cuda_initialize()
        print("gpu initialized")

    simulation.draw()
    for i in range(4000):

        simulation.run()
        simulation.draw()

    if GPU:
        simulation.cuda_clean()
        print("variables freed")