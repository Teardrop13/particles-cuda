#ifndef MOVE_PARTICLES_H
#define MOVE_PARTICLES_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_initialize(float *position_x,
                     float *position_y,
                     float *position_z,
                     float *acceleration_x,
                     float *acceleration_y,
                     float *acceleration_z,
                     float *mass,
                     int length);

void cuda_clean();

void move_particles(float *position_x,
                    float *position_y,
                    float *position_z);

#ifdef __cplusplus
}
#endif

#endif /* MOVE_PARTICLES_H */