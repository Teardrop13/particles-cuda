#ifndef MOVE_PARTICLES_GPU_H
#define MOVE_PARTICLES_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

__host__ __device__ struct Vector {
    float x;
    float y;
    float z;

    __host__ __device__ Vector(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    __host__ __device__ Vector() {
        x=0;
        y=0;
        z=0;
    }

    __host__ __device__ Vector operator + (const Vector &obj) {
         Vector new_vector;
         new_vector.x = this->x + obj.x;
         new_vector.y = this->y + obj.y;
         new_vector.z = this->z + obj.z;
         return new_vector;
    }

    __host__ __device__ Vector operator - (const Vector &obj) {
         Vector new_vector;
         new_vector.x = this->x - obj.x;
         new_vector.y = this->y - obj.y;
         new_vector.z = this->z - obj.z;
         return new_vector;
    }

    __host__ __device__ Vector operator / (const float &val) {
         Vector new_vector;
         new_vector.x = this->x / val;
         new_vector.y = this->y / val;
         new_vector.z = this->z / val;
         return new_vector;
    }

    __host__ __device__ Vector operator * (const Vector &val) {
         Vector new_vector;
         new_vector.x = this->x * val.x;
         new_vector.y = this->y * val.x;
         new_vector.z = this->z * val.x;
         return new_vector;
    }

    __host__ __device__ Vector operator * (const float &val) {
         Vector new_vector;
         new_vector.x = x * val;
         new_vector.y = y * val;
         new_vector.z = z * val;
         return new_vector;
    }

    __host__ __device__ Vector& operator += (const Vector &vec) {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;
        return *this;
    }
};

__host__ __device__ struct Particle {
    Vector position;
    Vector speed;
    float mass;
    float radius;
};

void cuda_initialize(Particle *particles,
                     int _number_of_particles,
                     float _dt,
                     float _G);

void cuda_clean();

void move_particles(Particle *particles);

#ifdef __cplusplus
}
#endif

#endif /* MOVE_PARTICLES_GPU_H */