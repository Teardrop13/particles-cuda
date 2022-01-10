#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "move_particles.hpp"

const float G = 2;  // nie prawdziwe G

long maxGridSize;
long maxThreadsPerBlock;

float *position_x;
float *position_y;
float *position_z;
float *acceleration_x;
float *acceleration_y;
float *acceleration_z;
int length;
float *mass;

float *d_position_x;
float *d_position_y;
float *d_position_z;
float *d_acceleration_x;
float *d_acceleration_y;
float *d_acceleration_z;
int *d_length;
float *d_mass;

__global__ void calculate_move(float *position_x,
                               float *position_y,
                               float *position_z,
                               float *acceleration_x,
                               float *acceleration_y,
                               float *acceleration_z,
                               float *mass,
                               int length) {
    for (int i = 0; i < length; i++) {
        for (int k = 0; k < length; k++) {
            float distance = pow(position_x[i] - position_x[k], 2) + pow(position_y[i] - position_y[k], 2) + pow(position_z[i] - position_z[k], 2);
            if (distance == 0) {
                distance = 0.000000001;
            }

            float f = G * mass / distance;
            acceleration_x[i] += (position_x[k] - position_x[i]) * f;
            acceleration_y[i] += (position_y[k] - position_y[i]) * f;
            acceleration_z[i] += (position_z[k] - position_z[i]) * f;
        }
    }

    for (int i = 0; i < length; i++) {
        position_x[i] += acceleration_x[i] * time / 2;
        position_y[i] += acceleration_y[i] * time / 2;
        position_z[i] += acceleration_z[i] * time / 2;
    }
}

void cuda_initialize(float *position_x,
                     float *position_y,
                     float *position_z,
                     float *acceleration_x,
                     float *acceleration_y,
                     float *acceleration_z,
                     float *mass,
                     int length) {
    int device = 0;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    maxGridSize = deviceProp.maxGridSize[0];
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    std::cout << "==============================================" << std::endl;
    std::cout << "Max dimension size of a grid size (x): " << maxGridSize << std::endl;
    std::cout << "Maximum number of threads per block: " << maxThreadsPerBlock << std::endl;
    std::cout << std::endl;

    cudaMalloc((void **)&d_position_x, sizeof(float) * length);
    cudaMemcpy(d_position_x, &position_x, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_position_y, sizeof(float) * length);
    cudaMemcpy(d_position_y, &position_y, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_position_z, sizeof(float) * length);
    cudaMemcpy(d_position_z, &position_z, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_acceleration_x, sizeof(float) * length);
    cudaMemcpy(d_acceleration_x, &acceleration_x, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_acceleration_y, sizeof(float) * length);
    cudaMemcpy(d_acceleration_y, &acceleration_y, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_acceleration_z, sizeof(float) * length);
    cudaMemcpy(d_acceleration_z, &acceleration_z, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_mass, sizeof(float) * length);
    cudaMemcpy(d_mass, &mass, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_length, sizeof(int));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);
}

void cuda_clean() {
    cudaFree(d_position_x);
    cudaFree(d_position_y);
    cudaFree(d_position_z);
    cudaFree(d_acceleration_x);
    cudaFree(d_acceleration_y);
    cudaFree(d_acceleration_z);
    cudaFree(d_mass);
    cudaFree(d_length);
}

void move_particles(float *position_x,
                    float *position_y,
                    float *position_z) {
    float time = 0.1;
    time = pow(time, 2);

    float mass = 2;

    <<<>>> calculate_move(d_position_x,
                          d_position_y,
                          d_position_z,
                          d_acceleration_x,
                          d_acceleration_y,
                          d_acceleration_z,
                          d_mass,
                          d_length);

    cudaMemcpy(position_x, &d_position_x, sizeof(float) * length, cudaMemcpyDeviceToHost);
    cudaMemcpy(position_y, &d_position_y, sizeof(float) * length, cudaMemcpyDeviceToHost);
    cudaMemcpy(position_z, &d_position_z, sizeof(float) * length, cudaMemcpyDeviceToHost);
}