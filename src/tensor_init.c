#include "../include/main.h"

double genNormalRandom(double mean, double sigma);

tensor *createTensor(int dim, const int *shape)
{
    // malloc for tensor
    tensor *outputTensor = malloc(sizeof(tensor));
    if (outputTensor == NULL)
    {
        printf("Error: Failed to malloc a tensor!\n");
        return NULL;
    }

    // malloc for outputTensor->shape
    outputTensor->shape = malloc(dim * sizeof(int));
    if (outputTensor->shape == NULL)
    {
        free(outputTensor);
        printf("Error: Failed to malloc for shape of the tensor!\n");
        return NULL;
    }

    // copy shape info to outputTensor->shape
    for (int dimIdx = 0; dimIdx < dim; ++dimIdx)
    {
        outputTensor->shape[dimIdx] = shape[dimIdx];
    }

    // calculate outputTensor->size
    int size = 1;
    for (int i = 0; i < dim; ++i)
    {
        size *= shape[i];
    }
    outputTensor->size = size;

    // malloc for outputTensor->data
    outputTensor->data = malloc(size * sizeof(double));
    if (outputTensor->data == NULL)
    {
        free(outputTensor->shape);
        free(outputTensor);
        printf("Error: Failed to malloc for data of the tensor!\n");
        return NULL;
    }

    // assign outputTensor->dimension
    outputTensor->dimensions = dim;

    return outputTensor;
}

tensor *copyTensor(const tensor *ten)
{
    tensor *result = malloc(sizeof(tensor));
    if (result == NULL)
    {
        printf("Error: Failed to malloc for clone tensor!\n");
        return NULL;
    }

    int dims = ten->dimensions;
    result->shape = malloc(dims * sizeof(int));
    if (result->shape == NULL)
    {
        printf("Error: Failed to malloc for clone tensor's shape!\n");
        return NULL;
    }

    result->data = malloc(ten->size * sizeof(double));
    if (result->data == NULL)
    {
        printf("Error: Failed to malloc for clone tensor's data!\n");
        free(result->shape);
        return NULL;
    }

    memcpy(result->shape, ten->shape, dims * sizeof(int));
    memcpy(result->data, ten->data, ten->size * sizeof(double));
    result->size = ten->size;
    result->dimensions = dims;
    return result;    
}

void destroyTensor(tensor *ten)
{
    if (ten == NULL)
    {
        return;
    }

    free(ten->data);
    free(ten->shape);
    free(ten);
}

void tensorFill(tensor *ten, double A)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = A;
    }
}

void tensorFillONES(tensor *ten)
{
    tensorFill(ten, 1);
}

void tensorFillZEROS(tensor *ten)
{
    memset(ten->data, 0, ten->size * sizeof(double));
}

tensor *createIdentityMatrix(int matSize)
{
    if (matSize < 1)
    {
        printf("Error: Cannot create an identity matrix of size %i!\n", matSize);
        return NULL;
    }

    int resultShape[2] = {matSize, matSize};

    tensor *result = createTensor(2, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create identity matrix!\n");
        return NULL;
    }

    // clean garbage values
    tensorFillZEROS(result);

    // assign 1's to the diagonal elements
    for (int i = 0; i < matSize; ++i)
    {
        *(result->data + i * (matSize + 1)) = 1.0;
    }

    return result;
}

// assign random values in [0,1) uniformly distributed to a tensor
void tensorRandomUinformUnity(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = random() / (RAND_MAX + 1.0);
    }
}

// assign random values in [0,1) normally distributed to a tensor
void tensorRandomNormal(tensor *ten, double mean, double sigma)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = genNormalRandom(mean, sigma);
    }
}

// generate random number normally distributed
double genNormalRandom(double mean, double sigma)
{
    // Box-Muller transform
    double u1;
    do
    {
        u1 = random() / (RAND_MAX + 1.0);
    }
    while (u1 == 0);
    double u2 = random() / (RAND_MAX + 1.0);
    double z = sqrt(-2 * log(u1)) * sin(2 * M_PI * u2);

    return (z * sigma + mean);
}