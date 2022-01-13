#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "move_particles.hpp"

long maxGridSize;
long maxThreadsPerBlock;

int number_of_particles;

float *d_position_x;
float *d_position_y;
float *d_position_z;
float *d_acceleration_x;
float *d_acceleration_y;
float *d_acceleration_z;
float *d_mass;
float *d_step;  // step nie jest do podnoszony do kwadratu ani dzielony przez 2
float *d_number_of_particles;
float *d_G;

__device__ calculate_acceleration_one_particle(float *d_current_position_x,
                                               float *d_current_position_y,
                                               float *d_current_position_z,
                                               float *d_current_acceleration_x,
                                               float *d_current_acceleration_y,
                                               float *d_current_acceleration_z,
                                               float *d_other_position_x,
                                               float *d_other_position_y,
                                               float *d_other_position_z,
                                               float *d_other_mass,
                                               float *d_step,
                                               float *d_G) {

    float distance = pow((*d_current_position_x) - (*d_other_position_x), 2) + pow((*d_current_position_y) - (*d_other_position_y), 2) + pow((*d_current_position_z) - (*d_other_position_z), 2);
    float f = (*d_G) * (*d_other_mass) / distance;

    *d_current_acceleration_x += ((*d_other_position_x) - (*d_current_position_x)) * f;
    *d_current_acceleration_y += ((*d_other_position_y) - (*d_current_position_y)) * f;
    *d_current_acceleration_z += ((*d_other_position_z) - (*d_current_position_z)) * f;
}

__device__ calculate_move_one_particle(float *d_current_position_x,
                                       float *d_current_position_y,
                                       float *d_current_position_z,
                                       float *d_current_acceleration_x,
                                       float *d_current_acceleration_y,
                                       float *d_current_acceleration_z,
                                       float *d_position_x,
                                       float *d_position_y,
                                       float *d_position_z,
                                       float *d_mass,
                                       float *d_step,
                                       int *d_number_of_particles,
                                       float d_G) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    <<<1, 1024>>> calculate_acceleration_one_particle(d_current_position_x,
                                               d_current_position_y,
                                               d_current_position_z,
                                               d_current_acceleration_x,
                                               d_current_acceleration_y,
                                               d_current_acceleration_z,
                                               d_position_x[i],
                                               d_position_y[i],
                                               d_position_z[i],
                                               d_mass,
                                               d_time,
                                               d_number_of_particles,
                                               d_G);

    *d_current_position_x += (*d_current_acceleration_x) * (*d_step);
    *d_current_position_y += (*d_current_acceleration_y) * (*d_step);
    *d_current_position_z += (*d_current_acceleration_z) * (*d_step);
}

__global__ void calculate_move_all_particles(float *d_position_x,
                                             float *d_position_y,
                                             float *d_position_z,
                                             float *d_acceleration_x,
                                             float *d_acceleration_y,
                                             float *d_acceleration_z,
                                             float *d_mass,
                                             float *d_step,
                                             int *d_number_of_particles,
                                             float d_G) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    <<< 1, 1024>>> calculate_move_one_particle(d_position_x[i],
                                               d_position_y[i],
                                               d_position_z[i],
                                               d_acceleration_x[i],
                                               d_acceleration_y[i],
                                               d_acceleration_z[i],
                                               d_position_x,
                                               d_position_y,
                                               d_position_z,
                                               d_mass,
                                               d_time,
                                               d_number_of_particles,
                                               d_G);
}

void cuda_initialize(float *position_x,
                     float *position_y,
                     float *position_z,
                     float *acceleration_x,
                     float *acceleration_y,
                     float *acceleration_z,
                     float *mass,
                     float *step,
                     int *number_of_particles,
                     float G) {
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

    cudaMalloc((void **)&d_position_x, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_position_x, &position_x, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_position_y, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_position_y, &position_y, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_position_z, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_position_z, &position_z, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_acceleration_x, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_acceleration_x, &acceleration_x, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaFree(d_mass);
    cudaMalloc((void **)&d_acceleration_y, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_acceleration_y, &acceleration_y, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_acceleration_z, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_acceleration_z, &acceleration_z, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_mass, sizeof(float) * (*number_of_particles));
    cudaMemcpy(d_mass, &mass, sizeof(float) * (*number_of_particles), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_step, sizeof(float));
    cudaMemcpy(d_step, &step, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_number_of_particles, sizeof(int));
    cudaMemcpy(d_number_of_particles, &number_of_particles, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_G, sizeof(float));
    cudaMemcpy(d_G, &G, sizeof(float), cudaMemcpyHostToDevice);
}

void cuda_clean() {
    cudaFree(d_position_x);
    cudaFree(d_position_y);
    cudaFree(d_position_z);
    cudaFree(d_acceleration_x);
    cudaFree(d_acceleration_y);
    cudaFree(d_acceleration_z);
    cudaFree(d_mass);
    cudaFree(d_step);
    cudaFree(d_number_of_particles);
    cudaFree(d_G);
}

void move_particles(float *position_x,
                    float *position_y,
                    float *position_z) {
    <<< 1, 1024>>> calculate_move_all_particles(d_position_x,
                                                d_position_y,
                                                d_position_z,
                                                d_acceleration_x,
                                                d_acceleration_y,
                                                d_acceleration_z,
                                                d_mass,
                                                d_time,
                                                d_number_of_particles,
                                                d_G);

    cudaMemcpy(position_x, &d_position_x, sizeof(float) * number_of_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(position_y, &d_position_y, sizeof(float) * number_of_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(position_z, &d_position_z, sizeof(float) * number_of_particles, cudaMemcpyDeviceToHost);
}