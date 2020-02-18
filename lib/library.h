#ifndef LIB_LIBRARY_H
#define LIB_LIBRARY_H

typedef struct MlpModel {
    int layers;
    int *npl;
    double ***weigths;
} MlpModel;

int sign(double value);

#endif //LIB_LIBRARY_H