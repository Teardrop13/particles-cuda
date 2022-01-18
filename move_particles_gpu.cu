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

Particle *d_particles;
float *d_dt;  // dt nie jest do podnoszony do kwadratu ani dzielony przez 2
float *d_G;

#define cuda_check(ans) \
    { _check((ans), __LINE__); }
inline void _check(cudaError_t code, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error:\n%s\n%d\n", cudaGetErrorString(code), line);
        exit(code);
    }
}

__device__ float get_distance(Vector a, Vector b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

__global__ void calculate_speed_one_to_one_particle(Particle *current_particle,
                                                           Particle *other_particle,
                                                           float *d_G,
                                                           float *d_dt) {
    float distance = get_distance((*current_particle).position, (*other_particle).position);

    if (distance < 0.1) {
        return;
    }
    float a = (*d_G) * (*other_particle).mass / pow(distance, 3);

    // tu powinien być atomicAdd
    // (*current_particle).speed += ((*other_particle).position - (*current_particle).position) * Vector(a* (*d_dt),a* (*d_dt),a* (*d_dt));
    
    atomicAdd(&(*current_particle).speed.x, ((*other_particle).position.x - (*current_particle).position.x) * a * (*d_dt));
    atomicAdd(&(*current_particle).speed.y, ((*other_particle).position.y - (*current_particle).position.y) * a * (*d_dt));
    atomicAdd(&(*current_particle).speed.z, ((*other_particle).position.z - (*current_particle).position.z) * a * (*d_dt));
}

__global__ void calculate_speed_all_to_one_particle(Particle *current_particle,
                                                    Particle *d_particles,
                                                    float *d_dt,
                                                    float *d_G,
                                                    int *d_blocks,
                                                    int *d_threads) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;


    calculate_speed_one_to_one_particle<<<*d_blocks, *d_threads>>>(current_particle,
                                                                   (&d_particles)[i],
                                                                   d_G,
                                                                   d_dt);
}

__global__ void calculate_speed_all_to_all_particles(Particle *d_particles,
                                                     float *d_dt,
                                                     float *d_G,
                                                     int *d_blocks,
                                                     int *d_threads) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // resetowanie prędkości
    // d_particles[i].speed = Vector(0,0,0);
    calculate_speed_all_to_one_particle<<<*d_blocks, *d_threads>>>(&d_particles[i],
                                                                   d_particles,
                                                                   d_dt,
                                                                   d_G,
                                                                   d_blocks,
                                                                   d_threads);
}

__global__ void calculate_position_all_particles(Particle *d_particles, float *d_dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    d_particles[i].position += (d_particles[i].speed * Vector(*d_dt,*d_dt,*d_dt));
    // d_particles[i].position += d_particles[i].speed;
    // d_particles[i].position += (d_particles[i].speed * (*d_dt));
    // d_particles[i].position.x += 2.;
}

void cuda_initialize(Particle *particles,
                     int _number_of_particles,
                     float dt,
                     float G) {
    int device = 0;
    number_of_particles = _number_of_particles;

    threads = 64;
    blocks = number_of_particles / threads;

    // threads = 32;
    // blocks = 32;

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

    cudaMalloc((void **)&d_dt, sizeof(float));
    cudaMemcpy(d_dt, &dt, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_G, sizeof(float));
    cudaMemcpy(d_G, &G, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_blocks, sizeof(int));
    cudaMemcpy(d_blocks, &blocks, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_threads, sizeof(int));
    cudaMemcpy(d_threads, &threads, sizeof(int), cudaMemcpyHostToDevice);
}

void cuda_clean() {
    cudaFree(d_particles);
    cudaFree(d_dt);
    cudaFree(d_G);
    cudaFree(d_blocks);
    cudaFree(d_threads);
}

void move_particles(Particle *particles) {
    // std::cout << "gpu przed: " << particles[0].speed.x << std::endl;

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    calculate_speed_all_to_all_particles<<<blocks, threads>>>(d_particles,
                                                       d_dt,
                                                       d_G,
                                                       d_blocks,
                                                       d_threads);

    cudaDeviceSynchronize();

    calculate_position_all_particles<<<blocks, threads>>>(d_particles, d_dt);

    cudaDeviceSynchronize();

    cuda_check(cudaMemcpy(particles, d_particles, sizeof(Particle) * number_of_particles, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << time << std::endl;

    // std::cout << "gpu po: " << particles[0].speed.x << std::endl;
}