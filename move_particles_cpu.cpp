#include "move_particles_cpu.hpp"

#include <math.h>

#include <stdio.h>

#include <cstdlib>
#include <iostream>
#include <chrono>


using namespace std::chrono;

float G; // nie prawdziwe G
float dt;

float get_distance(Vector a, Vector b) {
    return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2) + pow(a.z-b.z, 2));
}

void cpu_initalize(float _G, float _dt) {
    G = _G;
    dt = _dt;
}

void move_particles(Particle *particles, int length) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < length; i++) {
        Vector acceleration = Vector(0,0,0);

        for (int k = 0; k < length; k++) {
            
            if (i == k) {
                continue;
            }
            
            float distance = get_distance(particles[i].position, particles[k].position);

            if (distance < 0.01) {
                continue;
            }

            acceleration += (particles[k].position-particles[i].position) * G * particles[k].mass / pow(distance, 3);
        }

        particles[i].speed += (acceleration) * dt;
    }

    for (int i = 0; i < length; i++) {
        particles[i].position += particles[i].speed * dt;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "\r" << duration.count() / 1000000. << " ms   " << std::flush;
}