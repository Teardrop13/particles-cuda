#ifndef MOVE_PARTICLES_GPU_H
#define MOVE_PARTICLES_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

struct Vector
{
    float x;
    float y;
    float z;
};


struct Particle
{
    float x;
    float y;
    float z;
    float mass;
    Vector acceleration;
};

void cuda_initialize(Particle *particles,
                     float step,
                     int _number_of_particles,
                     float G);

void cuda_clean();

void move_particles(Particle *particles);

#ifdef __cplusplus
}
#endif

#endif /* MOVE_PARTICLES_GPU_H */