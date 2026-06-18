#include "../include/tenCor.h"

double tensorMSE(const tensor *predictions, const tensor *labels)
{
    if (!checkShapeSim(predictions, labels))
    {
        printf("Error: Dissimilar shapes!\n");
        return NAN;
    }
    int tenSize = predictions->size;
    double invSize = 1.0 / tenSize;

    double sum = 0;
    for (int i = 0; i < tenSize; ++i)
    {
        double diff = predictions->data[i] - labels->data[i]; 
        sum += diff * diff;
    }

    return (sum * invSize);
}

double tensorCrossEntropy(const tensor *predictions, const tensor *labels)
{
    if (!checkShapeSim(predictions, labels))
    {
        printf("Error: Dissimilar shapes!\n");
        return NAN;
    }
    int tenSize = predictions->size;
    double invSize = 1.0 / tenSize;

    double sum = 0;
    for (int i = 0; i < tenSize; ++i)
    {
        sum += labels->data[i] * log(predictions->data[i] + EPSILON);
    }

    return -(sum * invSize);
}