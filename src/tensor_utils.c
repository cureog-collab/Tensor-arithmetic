#include "../include/main.h"

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
