#include "move_particles.hpp"

#include <math.h>

#include <stdio.h>

#include <cstdlib>
#include <iostream>

const float G = 2; // nie prawdziwe G

void change_list(float *position_x,
                 float *position_y,
                 float *position_z,
                 float *acceleration_x,
                 float *acceleration_y,
                 float *acceleration_z,
                 int length) {

    float time = 0.1;
    time = pow(time, 2);

    float mass = 2;

    for (int i = 0; i < length; i++) {
        for (int k = 0; k < length; k++) {
            
                float distance = pow(position_x[i]-position_x[k], 2) + pow(position_y[i]-position_y[k], 2) + pow(position_z[i]-position_z[k], 2);
                if (distance == 0){
                    distance = 0.000000001;
                }

                float f = G * mass / distance;
                acceleration_x[i] += (position_x[k]-position_x[i]) * f;
                acceleration_y[i] += (position_y[k]-position_y[i]) * f;
                acceleration_z[i] += (position_z[k]-position_z[i]) * f;
        }
    }

    for (int i = 0; i < length; i++) {
        position_x[i] += acceleration_x[i] * time / 2;
        position_y[i] += acceleration_y[i] * time / 2;
        position_z[i] += acceleration_z[i] * time / 2;
    }
}