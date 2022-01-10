#ifndef MOVE_PARTICLES_H
#define MOVE_PARTICLES_H

#ifdef __cplusplus
extern "C" {
#endif

int funkcja(int a, int b);
void set_x(int _x);
void init_list(int _int_list[], int _list_length);
void print_list();
int *get_list();
void read_list(int list[], int length);
void change_list(float *position_x,
                 float *position_y,
                 float *position_z,
                 float *acceleration_x,
                 float *acceleration_y,
                 float *acceleration_z,
                 int length);

#ifdef __cplusplus
}
#endif

#endif /* MOVE_PARTICLES_H */