#include "../include/main.h"
#include <stdlib.h>

void matrixMultiplyCore(tensor *result, const double *data1, const double *data2, int M, int K, int N, int batch);

tensor *tensorTranspose(const tensor *ten, const int *axes)
{
    int *resultShape = malloc(ten->dimensions * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }

    // reshape the tensor
    for (int i = 0; i < ten->dimensions; ++i)
    {
        resultShape[i] = ten->shape[axes[i]];
    }

    tensor *result = createTensor(ten->dimensions, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to malloc for result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    int *newCoords = malloc(ten->dimensions * sizeof(int));
    if (newCoords == NULL)
    {
        printf("Error: Failed to malloc for newCoords!\n");
        destroyTensor(result);
        return NULL;
    }

    int *oldCoords = malloc(ten->dimensions * sizeof(int));
    if (oldCoords == NULL)
    {
        printf("Error: Failed to malloc for oldCoords!\n");
        free(newCoords);
        destroyTensor(result);
        return NULL;
    }

    for (int i = 0; i < result->size; ++i)
    {
        // locate addresses in the new tensor
        // row-major
        int temp = i;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            newCoords[d] = temp % result->shape[d]; // jump this number of steps in dimension d
            temp /= result->shape[d]; // accumulate jumps at dimension d and bring it onto dimension d - 1
        }

        // maps back to the address in the old tensor
        for (int d = 0; d < result->dimensions; ++d)
        {
            oldCoords[axes[d]] = newCoords[d];
        }

        int oldI = 0;
        int multiplier = 1;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            oldI += oldCoords[d] * multiplier; // accumulate jumps
            multiplier *= ten->shape[d]; // magnitude of a jump at the next dimension, so-called "weight" of each dimension
        }

        // finally assign to the new tensor's data
        result->data[i] = ten->data[oldI];
    }

    free(newCoords);
    free(oldCoords);
    return result;
}

tensor *tensorMultiply(const tensor *ten1, const tensor *ten2)
{
    // ten1->shape: {10, 5, 3, 4}
    // to be multiplicable: ten2->shape: {10, 5, 4, int N}
    // whereas: {10, 5}: batched dimensions

    // validation
    if (ten1->dimensions < 2 || ten2->dimensions < 2)
    {
        printf("Error: Tensors' ranks must be at least 2!\n");
        return NULL;
    }
    else if (ten1->dimensions != ten2->dimensions)
    {
        printf("Error: Tensors' dimensions must be the same!\n");
        return NULL;
    }
    int dims = ten1->dimensions;


    // if inputs are just matrices
    if (dims == 2)
    {
        // perform ordinary matrix multiplication
        int firstCols = ten1->shape[1];
        int secondRows = ten2->shape[0];
        if (secondRows != firstCols)
        {
            printf("Error: matrices not multiplicable!\n");
            return NULL;
        }
        int firstRows = ten1->shape[0];
        int secondCols = ten2->shape[1];

        int *resultShape = malloc(2 * sizeof(int));
        if (resultShape == NULL)
        {
            printf("Error: Failed to malloc for resultShape!\n");
            return NULL;
        }
        resultShape[0] = firstRows;
        resultShape[1] = secondCols;

        tensor *result = createTensor(2, resultShape);
        if (result == NULL)
        {
            printf("Error: Failed to malloc for result matrix!\n");
            free(resultShape);
            return NULL;
        }
        free(resultShape);
        tensorFillZEROS(result);

        matrixMultiplyCore(result, ten1->data, ten2->data, firstRows, firstCols, secondCols, 0);
        return result;
    }

    // if inputs are tensors of rank higher than 2

    // check batch dimensions (0..(dimensions - 3))
    for (int d = 0; d < dims - 2; ++d)
    {
        if (ten1->shape[d] != ten2->shape[d])
        {
            printf("Error: Dissimilar batch dimension!\nten1->shape[%i] != ten2->shape[%i]!\n", d, d);
            return NULL;
        }
    }

    // obtain the actual dimensionality of the will-be-multiplied matrices
    int firstCols = ten1->shape[dims - 1];
    int secondRows = ten2->shape[dims - 2];
    if (secondRows != firstCols)
    {
        printf("Error: matrices not multiplicable!\n");
        return NULL;
    }
    int firstRows = ten1->shape[dims - 2];
    int secondCols = ten2->shape[dims - 1];
    
    // resultShape
    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }
    for (int i = 0; i < dims - 1; ++i)
    {
        resultShape[i] = ten1->shape[i];
    }
    resultShape[dims - 1] = secondCols;
    
    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    tensorFillZEROS(result);

    // total numbers of matrices to be multiplied
    int totBatches = 1;
    for (int i = 0; i < dims - 2; ++i)
    {
        totBatches *= ten1->shape[i];
    }

    // multiply each pair of matrices
    for (int b = 0; b < totBatches; ++b)
    {
        matrixMultiplyCore(result, ten1->data, ten2->data, firstRows, firstCols, secondCols, b);
    }
    return result;
}

double matrixDet(const tensor *mat)
{
    // check if the inputs is an actual matrix
    int dim = mat->dimensions;
    if (dim != 2)
    {
        printf("Error: Input is not a matrix but a rank-%i tensor!\n", dim);
        return NAN;
    }
    
    // check if the input is a square matrix
    if (mat->shape[0] != mat->shape[1])
    {
        printf("Error: Input isn't a square matrix but a %ix%i one!\n", mat->shape[0], mat->shape[1]);
        return NAN;
    }

    tensor *cloneMat = copyTensor(mat);
    if (cloneMat == NULL)
    {
        printf("Error: Failed to clone input tensor!\n");
        return NAN;
    }

    // this is for handling the effect of swapping rows when performing Gaussian elimination
    int swapSign = 1;
    
    // gaussian eliminationt
    gaussianElimination(cloneMat, 0, 0, &swapSign);

    // mutliply pivots to get the determinant
    double determinant = 1;
    int maxRowIdx = cloneMat->shape[0];
    int maxColIdx = cloneMat->shape[1];
    for (int rowIdx = 0; rowIdx < maxRowIdx; ++rowIdx)
    {
        // if the current matrix element is 0, return 0
        double elementValue = *(cloneMat->data + rowIdx * (maxColIdx + 1));
        if (elementValue == 0)
        {
            destroyTensor(cloneMat);
            return 0;
        }

        determinant *= elementValue;
    }
    determinant *= swapSign;

    destroyTensor(cloneMat);
    return  determinant;
}



// helper
void matrixMultiplyCore(tensor *result, const double *data1, const double *data2, int M, int K, int N, int batch)
{
    // offset
    int offsetFirst = batch * M * K; // batch * size of the first matrix
    int offsetSecond = batch * K * N; // batch * size of the second matrix
    int offsetResult = batch * M * N; // batch * size of the result matrix

    // matrix multpiplication
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            for (int j = 0; j < N; ++j)
            {
                int indexFirst = offsetFirst + (i * K + k);
                int indexSecond = offsetSecond + (k * N + j);
                int indexResult = offsetResult + (i * N + j);
                result->data[indexResult] += data1[indexFirst] * data2[indexSecond];
            }
        }
    }
}