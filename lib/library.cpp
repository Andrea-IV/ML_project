#include "library.h"
#include <chrono>
#include <random>

#include <iostream>

//[type_model][class/regression][fonction]

extern "C" {
    __declspec(dllexport) double * linearCreateModel(int inDim) {
        int weightsSize = inDim + 1;
        auto weights = (double *)(malloc(weightsSize * sizeof(double)));

        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{-1, 1};

        for(int i = 0; i < weightsSize; i++) {
            weights[i] = distribution(randomEngine);
        }

        return weights;
    }

    __declspec(dllexport) int linearClassPredict(double *model, int inDim, double *params) {
        double sum = model[0];
        for(int i = 0; i < inDim; i++) {
            sum += params[i] * model[i + 1];
        }

        int result = sign(sum);
        return result;
    }

    __declspec(dllexport)void linearClassTrain(double *model, int inDim, int epoch, double trainingStep,
            double *trainingParams, int trainingParamsNumber, const double *trainingResults) {

        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{0, 1};

        for(int e = 0; e < epoch; e++) {
            int trainingPicked = (int) floor(distribution(randomEngine) * trainingParamsNumber);
            int trainingParamsPosition = inDim * trainingPicked;
            double modification = (double)trainingStep * (trainingResults[trainingPicked] -
                    linearClassPredict(model, inDim, &trainingParams[trainingParamsPosition]));
            model[0] += modification;

            for(int j = 0; j < inDim; j++) {
                model[j + 1] += modification * trainingParams[trainingParamsPosition + j];
            }
        }
    }

    __declspec(dllexport) void linearClearModel(const double *model) {
        delete model;
    }

    __declspec(dllexport) double * mlpCreateModel(int layers, const int npl[]) {
        int totalWeights = 0;
        for(int i = 0; i < layers - 1; i++) {
            totalWeights += (npl[i] + 1) * npl[i + 1];
        }

        totalWeights += npl[layers - 1] + 1;

        int modelSize = 1 + layers + totalWeights;
        auto model = (double *)(malloc(modelSize * sizeof(double)));

        model[0] = layers;
        int modelIndex = 1;
        for(int i = 0; i < layers; i++) {
            model[modelIndex] = npl[i];
            modelIndex++;
        }

        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{-1, 1};

        while(modelIndex < modelSize) {
            model[modelIndex] = distribution(randomEngine);
            modelIndex++;
        }

        return model;
    }

    __declspec(dllexport) double mlpPredict(double *rawModel, double *inParams) {
        auto model = importMlpModel(rawModel);
        auto nodes = (double **) malloc(model->layers * sizeof(double *));

        nodes[0] = (double *) malloc((model->npl[0] + 1) * sizeof(double));

        nodes[0][0] = 1;
        for(int i = 0; i < model->npl[0]; i++) {
            nodes[0][i + 1] = inParams[i];
        }

        for(int l = 1; l < model->layers; l++) {
            nodes[l] = (double *) malloc((model->npl[l] + 1) * sizeof(double));
            nodes[l][0] = 1;
            for(int i = 0; i < model->npl[l]; i++) {
                double sum = 0;
                for(int j = 0; j < model->npl[l - 1] + 1; j++) {
                    sum += nodes[l - 1][j] * model->weigths[l - 1][j][i];
                }
                nodes[l][i + 1] = tanh(sum);
            }
        }

        //Display nodes to verify
        std::cout << "Nodes :" << std::endl << "[ " << std::endl;
        for(int l = 0; l < model->layers; l++) {
            std::cout << "\t[ ";
            for(int i = 0; i < model->npl[l] + 1; i++) {
                std::cout << nodes[l][i] << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;

        double sum = 0;
        for(int i = 0; i < model->npl[model->layers - 1] + 1; i++) {
            sum += nodes[model->layers - 1][i] * model->weigths[model->layers - 1][i][0];
        }
        return sign(sum);
    }
}

int sign(double value) {
    return value == 0 ? 0 : value < 0 ? -1 : 1;
}

void predictAll(double *model) {
    for(double i = -7; i < 8; i += 1) {
        for(double j = 0; j < 15; j += 1) {
            double params[] = {i, j};
            int predicted = linearClassPredict(model, 2, params);
            std::cout << (predicted == 0 ? "0" : predicted >= 1 ? "+" : "-");
        }
        std::cout << std::endl;
    }
}

void displayMlpModel(MlpModel *model) {
    std::cout << "layers : " << model->layers << std::endl;
    std::cout << "npl : [ ";
    for(int i = 0; i < model->layers; i++) {
        std::cout << model->npl[i] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Weights :" << std::endl << "[" << std::endl;
    for(int l = 0; l < model->layers - 1; l++) {
        std::cout << "\t[" << std::endl;
        for(int i = 0; i < model->npl[l] + 1; i++) {
            std::cout << "\t\t[ ";
            for(int j = 0; j < model->npl[l + 1]; j++) {
                std::cout << model->weigths[l][i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "\t]" << std::endl;
    }

    std::cout << "\t[" << std::endl << "\t\t[ ";
    for(int i = 0; i < model->npl[model->layers - 1] + 1; i++) {
        std::cout << model->weigths[model->layers - 1][i][0] << " ";
    }
    std::cout << "]" << std::endl << "\t]" << std::endl << "]" << std::endl;
}

MlpModel * importMlpModel(const double *rawModel) {
    auto *model = (MlpModel *) malloc(sizeof(MlpModel));
    model->layers = (int) rawModel[0];
    model->npl = (int *) malloc(model->layers * sizeof(int));

    int modelIndex = 1;
    // Fill npl array
    for(int i = 0; i < model->layers; i++) {
        model->npl[i] = (int) rawModel[modelIndex];
        modelIndex++;
    }

    // Malloc and fill weights array
    model->weigths = (double ***) malloc((model->layers) * sizeof(double **));
    for(int l = 1; l < model->layers; l++) {
        model->weigths[l - 1] = (double **) malloc((model->npl[l - 1] + 1) * sizeof(double *));
        for(int i = 0; i < model->npl[l - 1] + 1; i++) {
            model->weigths[l - 1][i] = (double *) malloc(model->npl[l] * sizeof(double));
            for(int j = 0; j < model->npl[l]; j++) {
                model->weigths[l - 1][i][j] = rawModel[modelIndex];
                modelIndex++;
            }
        }
    }

    model->weigths[model->layers - 1] = (double **) malloc(model->npl[model->layers - 1] * sizeof(double *));
    for(int i = 0; i < model->npl[model->layers - 1] + 1; i++) {
        model->weigths[model->layers - 1][i] = (double *) malloc(sizeof(double));
        model->weigths[model->layers - 1][i][0] = rawModel[modelIndex];
        modelIndex++;
    }

    return model;
}

double * exportMlpModel(MlpModel *model) {
    int totalNodes = 0;
    for(int i = 0; i < model->layers - 1; i++) {
        totalNodes += (model->npl[i] + 1) * model->npl[i + 1];
    }

    int modelSize = 1 + model->layers + totalNodes;
    auto rawModel = (double *)(malloc(modelSize * sizeof(double)));

    rawModel[0] = model->layers;
    int modelIndex = 1;
    for(int i = 0; i < model->layers; i++) {
        rawModel[modelIndex] = model->npl[i];
        modelIndex++;
    }

    for(int l = 0; l < model->layers - 1; l++) {
        for(int i = 0; i < model->npl[l] + 1; i++) {
            for(int j = 0; j < model->npl[l + 1]; j++) {
                rawModel[modelIndex] = model->weigths[l][i][j];
                modelIndex++;
            }
        }
    }

    return rawModel;
}

int main(int argc, char **argv) {
//    int inDim = 2;
//    double *model = linearCreateModel(inDim);
//
//    double trainingParams[] = {-3, 9, 6, 13, -7, 2};
//    double trainingResults[] = {1, 1, -1};
//    linearClassTrain(model, 2, 1000, 0.1, trainingParams, 3, trainingResults);
//
//    predictAll(model);

//    int layers = 5;
//    int npl[] = {3, 4, 4, 2, 3};
    int layers = 3;
    int npl[] = {2, 3, 1};
    double *rawModel = mlpCreateModel(layers, npl);
    MlpModel *model = importMlpModel(rawModel);
    displayMlpModel(model);

//    displayMlpModel(importMlpModel(exportMlpModel(model)));

    double inParams[] = {3.14, 1.12};
    std::cout << mlpPredict(rawModel, inParams);

    return 0;
}