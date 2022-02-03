#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "move_particles_gpu.hpp"

long maxGridSize;
long maxThreadsPerBlock;

int number_of_particles;

int threads;
int blocks;

Particle *d_particles;
float dt;
float G;

#define cuda_check(ans) \
    { _check((ans), __LINE__); }
inline void _check(cudaError_t code, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error:\n%s\n%d\n", cudaGetErrorString(code), line);
        exit(code);
    }
}

__device__ float get_distance(Vector a, Vector b) {
    return sqrtf(powf(a.x - b.x, 2) + powf(a.y - b.y, 2) + powf(a.z - b.z, 2));
}

__device__ void calculate_speed_one_to_one_particle(Particle *current_particle,
                                                           Particle *other_particle,
                                                           float G,
                                                           float dt) {
    float distance = get_distance((*current_particle).position, (*other_particle).position);

    if (distance < 0.00001) {
        return;
    }
    float a = G * (*other_particle).mass / pow(distance, 3);

    atomicAdd(&(*current_particle).speed.x, ((*other_particle).position.x - (*current_particle).position.x) * a * dt);
    atomicAdd(&(*current_particle).speed.y, ((*other_particle).position.y - (*current_particle).position.y) * a * dt);
    atomicAdd(&(*current_particle).speed.z, ((*other_particle).position.z - (*current_particle).position.z) * a * dt);
}

__global__ void calculate_speed_all_to_one_particle(Particle *current_particle,
                                                    Particle *d_particles,
                                                    int number_of_particles,
                                                    float dt,
                                                    float G) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < number_of_particles) {
    calculate_speed_one_to_one_particle(current_particle,
                                        (&d_particles)[i],
                                        G,
                                        dt);
    }
}

__global__ void calculate_speed_all_to_all_particles(Particle *d_particles,
                                                    int number_of_particles,
                                                     float dt,
                                                     float G,
                                                     int blocks,
                                                     int threads) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < number_of_particles) {
    calculate_speed_all_to_one_particle<<<blocks, threads>>>(&d_particles[i],
                                                                   d_particles,
    number_of_particles,
                                                                   dt,
                                                                   G);
    }
}

__global__ void calculate_position_all_particles(Particle *d_particles, int number_of_particles, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < number_of_particles) {
        d_particles[i].position += (d_particles[i].speed * Vector(dt, dt, dt));
    }
}

void cuda_initialize(Particle *particles,
                     int _number_of_particles,
                     float _dt,
                     float _G) {
    int device = 0;
    number_of_particles = _number_of_particles;
    dt = _dt;
    G = _G;

    threads = 64;
    blocks = number_of_particles / threads;

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

    cudaMalloc((void **)&d_particles, sizeof(Particle) * number_of_particles);
    cudaMemcpy(d_particles, particles, sizeof(Particle) * number_of_particles, cudaMemcpyHostToDevice);
}

void cuda_clean() {
    cudaFree(d_particles);
}

void move_particles(Particle *particles) {
    // std::cout << "gpu przed: " << particles[0].speed.x << std::endl;

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    calculate_speed_all_to_all_particles<<<blocks, threads>>>(d_particles,
                                                        number_of_particles,
                                                       dt,
                                                       G,
                                                       blocks,
                                                       threads);

    cudaDeviceSynchronize();

    calculate_position_all_particles<<<blocks, threads>>>(d_particles, number_of_particles, dt);

    cudaDeviceSynchronize();

    cuda_check(cudaMemcpy(particles, d_particles, sizeof(Particle) * number_of_particles, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "\r" << time << " ms   " << std::flush;
}