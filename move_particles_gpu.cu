#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "move_particles_gpu.hpp"

long maxGridSize;
long maxThreadsPerBlock;

int number_of_particles;

int threads;
int *d_threads;
int blocks;
int *d_blocks;

float *d_current_particles;
float *d_previous_particles;
float *d_dt;  // dt nie jest do podnoszony do kwadratu ani dzielony przez 2
int *d_number_of_particles;
float *d_G;

#define cuda_check(ans) \
    { _check((ans), __LINE__); }
inline void _check(cudaError_t code, char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error:\n%s\n%d\n", cudaGetErrorString(code), line);
        exit(code);
    }
}

float get_distance(Vector a, Vector b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

__global__ void calculate_acceleration_one_to_one_particle(Particle *current_particle,
                                                           Particle *other_particle,
                                                           float *d_G) {
    float distance = get_distance((*current_particle).position, (*other_particle).position);

    if (distance == 0) {
        return;
    }
    float a = (*d_G) * (*other_particle).mass / pow(distance, 3);

    atomicAdd((*current_particle).speed, ((*other_particle).position - (*current_particle).position) * a * dt);
}

__global__ void calculate_speed_all_to_one_particle(Particle *current_particle,
                                                    Particle *d_previous_particles,
                                                    int *d_number_of_particles,
                                                    float *d_dt,
                                                    float *d_G,
                                                    int *d_blocks,
                                                    int *d_threads) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    calculate_speed_one_to_one_particle<<<*d_blocks, *d_threads>>>(*current_particle,
                                                                   &d_previous_particles[i],
                                                                   *d_G);
}

__global__ void calculate_speed_all_to_all_particles(Particle *d_current_particles,
                                                     Particle *d_previous_particles,
                                                     int *d_number_of_particles,
                                                     float *d_dt,
                                                     float *d_G,
                                                     int *d_blocks,
                                                     int *d_threads) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    calculate_speed_all_to_one_particle<<<*d_blocks, *d_threads>>>(&d_current_particles[i],
                                                                   d_previous_particles,
                                                                   d_number_of_particles,
                                                                   d_dt,
                                                                   d_G,
                                                                   d_blocks,
                                                                   d_threads);
}

__global__ void calculate_position_all_particles(Particle *d_current_particles,
                                                 float *d_dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    (*d_current_particles)[i].position += (*d_current_particles)[i].speed * (*d_dt);
}

void cuda_initialize(Particle *particles,
                     int _number_of_particles,
                     float dt,
                     float G) {
    int device = 0;
    number_of_particles = _number_of_particles;

    threads = 512;
    blocks = number_of_particles / thread;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    maxGridSize = deviceProp.maxGridSize[0];
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    std::cout << "==============================================" << std::endl;
    std::cout << "Max dimension size of a grid size (x): " << maxGridSize << std::endl;
    std::cout << "Maximum number of threads per block: " << maxThreadsPerBlock << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;

    cudaMalloc((void **)d_current_particles, sizeof(Particle) * number_of_particles);
    cudaMemcpy(d_current_particles, particles, sizeof(Particle) * number_of_particles, cudaMemcpyHostToDevice);

    cudaMalloc((void **)d_previous_particles, sizeof(Particle) * number_of_particles);
    cudaMemcpy(d_previous_particles, particles, sizeof(Particle) * number_of_particles, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_dt, sizeof(float));
    cudaMemcpy(d_dt, &dt, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_number_of_particles, sizeof(int));
    cudaMemcpy(d_number_of_particles, &number_of_particles, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_G, sizeof(float));
    cudaMemcpy(d_G, &G, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_blocks, sizeof(int));
    cudaMemcpy(d_blocks, &blocks, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_threads, sizeof(int));
    cudaMemcpy(d_threads, &threads, sizeof(int), cudaMemcpyHostToDevice);
}

void cuda_clean() {
    cudaFree(d_current_particles);
    cudaFree(d_previous_particles);
    cudaFree(d_dt);
    cudaFree(d_number_of_particles);
    cudaFree(d_G);
    cudaFree(d_blocks);
    cudaFree(d_threads);
}

void move_particles(float *particles) {
    std::cout << "przed: " << position_x[0] << ", " << position_x[number_of_particles - 1] << std::endl;

    calculate_speed_all_particles<<<blocks, threads>>>(d_current_particles,
                                                       d_previous_particles,
                                                       d_dt,
                                                       d_G,
                                                       d_number_of_particles,
                                                       d_blocks,
                                                       d_threads);

    cudaDeviceSynchronize();

    calculate_position_all_particles<<<blocks, threads>>>(d_current_particles, d_dt);

    cudaDeviceSynchronize();

    cuda_check(cudaMemcpy(d_previous_particles, d_current_particles, sizeof(Particle) * number_of_particles, cudaMemcpyHostToHost));

    cuda_check(cudaMemcpy(particles, d_current_particles, sizeof(Particle) * number_of_particles, cudaMemcpyDeviceToHost));
    // std::cout << "po skopiowaniu: " << position_x[0] << ", " << position_x[number_of_particles-1] << std::endl;
}