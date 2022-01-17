#ifndef MOVE_PARTICLES_H
#define MOVE_PARTICLES_H

#ifdef __cplusplus
extern "C" {
#endif

struct Vector {
    float x;
    float y;
    float z;

    Vector(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    Vector() {
        x=0;
        y=0;
        z=0;
    }

    Vector operator + (const Vector &obj) {
         Vector new_vector;
         new_vector.x = x + obj.x;
         new_vector.y = y + obj.y;
         new_vector.z = z + obj.z;
         return new_vector;
    }

    Vector operator - (const Vector &obj) {
         Vector new_vector;
         new_vector.x = x - obj.x;
         new_vector.y = y - obj.y;
         new_vector.z = z - obj.z;
         return new_vector;
    }

    Vector operator / (const float &val) {
         Vector new_vector;
         new_vector.x = x / val;
         new_vector.y = y / val;
         new_vector.z = z / val;
         return new_vector;
    }

    Vector operator * (const float &val) {
         Vector new_vector;
         new_vector.x = x * val;
         new_vector.y = y * val;
         new_vector.z = z * val;
         return new_vector;
    }

    Vector& operator += (const Vector &vec) {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;
        return *this;
    }
};

struct Particle {
    Vector position;
    Vector speed;
    float mass;
};

void cpu_initalize(float _G, float _dt);

void move_particles(Particle *particles, int length);

#ifdef __cplusplus
}
#endif

#endif /* MOVE_PARTICLES_H */