#ifndef MAIN_H
#define MAIN_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    double *data; // actual data stored
    int *shape; // store the size of each dimension of the tensor
    int dimensions; // dimension of the tensor
    int size; // total components from every dimension
} tensor;

// MEMORY HANDLING =========================================================================================
tensor *createTensor(int dim, const int *shape);
tensor *copyTensor(const tensor *ten);
void destroyTensor(tensor *ten);
// =========================================================================================================

// TENSOR OPERATORS ========================================================================================
void tensorFill(tensor *ten, double A);
void tensorFillZEROS(tensor *ten);
void tensorFillONES(tensor *ten);
void tensorRandomUinformUnity(tensor *ten);
void tensorRandomNormal(tensor *ten, double mean, double sigma);

// ---------------------------------------------------------------------------------------------------------

void tensorScale(tensor *ten, double scalar);
tensor *tensorAdd(const tensor *ten1, const tensor *ten2, bool isAdd);
tensor *tensorHadamardProduct(const tensor *ten1, const tensor *ten2);
tensor *tensorRelu(const tensor *ten);
tensor *tensorSigmoid(const tensor *ten);
tensor *tensorTanh(const tensor *ten);

// ---------------------------------------------------------------------------------------------------------

tensor *tensorTranspose(const tensor *ten, const int *axes);
tensor *tensorInverse(const tensor *ten);
tensor *tensorMultiply(const tensor *ten1, const tensor *ten2);
double matrixDet(const tensor *mat);

// ---------------------------------------------------------------------------------------------------------

double tensorSum(const tensor *ten);
double tensorMean(const tensor *ten);
tensor *tensorSumByAxis(const tensor *ten, int axis);

// ---------------------------------------------------------------------------------------------------------

tensor *tensorReshape(const tensor *ten, int *newShape, int newDim);
tensor *tensorFlatten(const tensor *ten);
void gaussianElimination(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign);
// =========================================================================================================

#endif