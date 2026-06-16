#include "../include/main.h"
#include <string.h>

tensor *tensorReshape(const tensor *ten, int *newShape, int newDim)
{
    if (ten == NULL)
    {
        printf("Error: Cannot reshape NULL!\n");
        return NULL;
    }

    tensor *result = createTensor(newDim, newShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    if (result->size != ten->size)
    {
        printf("Error: Dissimilar size!\nInput tensor has size %i while new tensor has size %i\n", ten->size, result->size);
        destroyTensor(result);
        return NULL;
    }

    // allocate data
    memcpy(result->data, ten->data, ten->size * sizeof(double));

    return result;
}

tensor *tensorFlatten(const tensor *ten)
{
    if (ten == NULL)
    {
        printf("Error: Cannot flatten NULL!\n");
        return NULL;
    }

    int resultShape[1] = {ten->size};

    tensor *result = createTensor(1, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    memcpy(result->data, ten->data, ten->size * sizeof(double));
    return result;
}

// Gaussian elimination for MxN matrix
static void gaussianEliminationCore(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign)
{
    // static const int DOUBLE_SIZE = sizeof(double);
    // 0. base case: entryRow == mat->shape[0] - 1 || entryCol == mat->shape[1] - 1
    int maxRowIdx = mat->shape[0];
    int maxColIdx = mat->shape[1];
    if (entryRowIdx >= maxRowIdx || entryColIdx >= maxColIdx)
    {
        return;
    }
    int matCols = maxColIdx;

    // 1. search down in the entry col, find the row i which will give
    //    maximum entry.
    int rowIdxWithMaxVal = entryRowIdx;
    double maxAbsValue = fabs(*(mat->data + entryRowIdx * matCols + entryColIdx));
    for (int rowIdx = entryRowIdx + 1; rowIdx < maxRowIdx; ++rowIdx)
    {
        double absVal = fabs(*(mat->data + rowIdx * matCols + entryColIdx));
        if (absVal > maxAbsValue)
        {
            maxAbsValue = absVal;
            rowIdxWithMaxVal = rowIdx;
        }
    }

    // 2. swap that mat[entryRow] with mat[i] (if i != entryRow)
    if (rowIdxWithMaxVal != entryRowIdx)
    {
        *swapSign = -*swapSign;
        for (int colIdx = entryColIdx; colIdx < maxColIdx; ++colIdx)
        {
            double tempVal = *(mat->data + entryRowIdx * matCols + colIdx);
            *(mat->data + entryRowIdx * matCols + colIdx) = *(mat->data + rowIdxWithMaxVal * matCols + colIdx);
            *(mat->data + rowIdxWithMaxVal * matCols + colIdx) = tempVal;
        }
    }

    // check if the entire col is nonzero
    double pivotVal = *(mat->data + entryRowIdx * matCols + entryColIdx);
    if (pivotVal != 0)
    {
        // 3. loop from mat[entryRow + 1] through and to mat[mat->space[0] - 1], at each iteration i:
        for (int rowIdx = entryRowIdx + 1; rowIdx < maxRowIdx; ++rowIdx)
        {
            //    3.1. find the value u such that:
            //         mat[i][entryCol] == mat[entryRow][entryCol] * u
            double u = *(mat->data + rowIdx * matCols + entryColIdx) / *(mat->data + entryRowIdx * matCols + entryColIdx);

            //    3.2. loop from mat[i][entryCol] through and to mat[i][mat->shape[1] - 1],
            //         at each iteration j: assign mat[i][j] -= mat[entryRow][j] * u
            for (int colIdx = entryColIdx; colIdx < maxColIdx; ++colIdx)
            {
                *(mat->data + rowIdx * matCols + colIdx) -= *(mat->data + entryRowIdx * matCols + colIdx) * u;
            }
        }
    }

    // 4. move to the next pivot and recursive
    gaussianEliminationCore(mat, entryRowIdx + 1, entryColIdx + 1, swapSign);
    return;
}

void gaussianElimination(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign)
{
    // dummy address on stack
    int dummy = 1;

    // check if the user wants to store the sign of the elimination
    swapSign = (swapSign == NULL)? &dummy : swapSign;

    // call the actual process
    gaussianEliminationCore(mat, entryRowIdx, entryColIdx, swapSign);
}