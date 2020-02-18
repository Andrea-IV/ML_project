#include "library.h"
#include <chrono>
#include <random>

#include <iostream>

extern "C" {
    __declspec(dllexport) double * create_model(int inDim, int outDim) {
        int weightsSize = inDim + 1;
        auto *weights = (double *)(malloc(weightsSize * sizeof(double)));

        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{-1, 1};

        for(int i = 0; i < weightsSize; i++) {
            weights[i] = distribution(randomEngine);
        }

        return weights;
    }

    __declspec(dllexport) int predict(double *model, int inDim, double *params) {
        double sum = model[0];
        for(int i = 0; i < inDim; i++) {
            sum += params[i] * model[i + 1];
        }

        int result = sign(sum);
        return result;
    }

    __declspec(dllexport)void train(double *model, int inDim, int epoch, double trainingStep, double *trainingParams, int trainingParamsNumber,
                                    const double *trainingResults) {

        for(int e = 0; e < epoch; e++) {
            for(int i = 0; i < trainingParamsNumber; i++) {
                int trainingParamsPosition = inDim * i;
                double modification = (double)trainingStep * (trainingResults[i] - predict(model, inDim, &trainingParams[trainingParamsPosition]));
                model[0] += modification;

                for(int j = 0; j < inDim; j++) {
                    model[j + 1] += modification * trainingParams[trainingParamsPosition + j];
                }
            }
        }
    }

    __declspec(dllexport) void clear_model(const double *model) {
        delete model;
    }
}

int sign(double value) {
    return value == 0 ? 0 : value < 0 ? -1 : 1;
}

void predictAll(double *model) {
    for(double i = -7; i < 8; i += 1) {
        for(double j = 0; j < 15; j += 1) {
            double params[] = {i, j};
            int predicted = predict(model, 2, params);
            std::cout << (predicted == 0 ? "0" : predicted >= 1 ? "+" : "-");
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    int inDim = 2;
    double *model = create_model(inDim, 0);

    double trainingParams[] = {-3, 9, 6, 13, -7, 2};
    double trainingResults[] = {1, 1, -1};
    train(model, 2, 10, 0.1, trainingParams, 3, trainingResults);

    predictAll(model);

    return 0;
}