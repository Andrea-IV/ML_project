#ifndef LIB_LIBRARY_H
#define LIB_LIBRARY_H

int sign(double value);

typedef struct LinearModel {
    int inDim;
    int outDim;
    double **weights;
} LinearModel;

#endif //LIB_LIBRARY_H