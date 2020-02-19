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
        auto nodes = calculateNodes(model, inParams);

        //Display nodes to verify
        displayNodes(model, nodes);

        double sum = 0;
        for(int i = 0; i < model->npl[model->layers - 1]; i++) {
            sum += nodes[model->layers - 1][i] * model->weigths[model->layers - 1][i][0];
        }

        return sign(sum);
    }

    __declspec(dllexport) void mlpClassTrain(double *rawModel, double trainingStep,
            int trainingNumber, double *trainingParams, double *trainingExpected) {
        MlpModel *model = importMlpModel(rawModel);

        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{0, 1};

        // Loop here for epoch
        auto trainingPosition = (int) floor(distribution(randomEngine) * trainingNumber) * (model->npl[0] + 1);
        auto nodes = calculateNodes(model, &trainingParams[trainingPosition]);

        std::cout << "First delta calculated : " << std::endl;

        auto delta = (double **) malloc(model->layers * sizeof(double *));
        delta[model->layers - 1] = (double *) malloc(model->npl[model->layers - 1] * sizeof(double));
        for(int i = 0; i < model->npl[model->layers - 1]; i++) {
            delta[model->layers - 1][i] = (1 - pow(nodes[model->layers - 1][i], 2) * (nodes[model->layers - 1][i] - trainingExpected[i]));
            std::cout << "delta[" << model->layers - 1 << "][" << i << "] : " << delta[model->layers - 1][i] << std::endl;
        }
        std::cout << std::endl;

        for(int l = model->layers - 2; l >= 0; l--) {
            delta[l] = (double *) malloc((model->npl[l] + 1) * sizeof(double));
            for(int i = 0; i < model->npl[l] + 1; i++) {
                double sum = 0;
//                for(int j = (l == model->layers - 2 ? 0 : 1); j < (l == model->layers - 2 ? model->npl[l + 1] : model->npl[l + 1] + 1); j++) {
                for(int j = 0; j < model->npl[l + 1]; j++) {
                    sum += model->weigths[l][i][j] * delta[l + 1][j];
                    std::cout << "model->weigths[" << l << "][" << i << "][" << j << "] : " << model->weigths[l][i][j] <<
                        "; delta[" << l + 1 << "][" << j << "] : " << delta[l + 1][j] << std::endl;
                }
                delta[l][i] = (1 - pow(nodes[l][i], 2)) * sum;
                std::cout << "calculated delta[" << l << "][" << i << "] : " << delta[l][i] <<
                "; nodes[" << l << "][" << i << "] : " << nodes[l][i] << std::endl << std::endl;
            }
        }

        for(int l = 0; l < model->layers - 1; l++) {
            for(int i = 0; i < model->npl[l]; i++) {
                for(int j = 0; j < model->npl[l + 1]; j++) {
                    model->weigths[l][i][j] = model->weigths[l][i][j] - trainingStep * nodes[l][i] * delta[l + 1][j];
                }
            }
        }

        //Display delta to verify
        std::cout << "Delta :" << std::endl << "[" << std::endl;
        for(int l = 0; l < model->layers; l++) {
            std::cout << "\t[ ";
            for(int i = 0; i < model->npl[l]; i++) {
                std::cout << delta[l][i] << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;

        //Free delta
        for(int l = 0; l < model->layers; l++) {
            free(delta[l]);
        }
        free(delta);
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

    std::cout << "]" << std::endl;
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
    for(int i = 0; i < model->npl[model->layers - 1]; i++) {
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

double ** calculateNodes(MlpModel *model, double *inParams) {
    auto nodes = (double **) malloc(model->layers * sizeof(double *));

    nodes[0] = (double *) malloc((model->npl[0] + 1) * sizeof(double));

    nodes[0][0] = 1;
    for(int i = 0; i < model->npl[0]; i++) {
        nodes[0][i + 1] = inParams[i];
    }

    for(int l = 1; l < model->layers - 1; l++) {
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

    nodes[model->layers - 1] = (double *) malloc(model->npl[model->layers -1] * sizeof(double));
    for(int i = 0; i < model->npl[model->layers - 1]; i++) {
        double sum = 0;
        for(int j = 0; j < model->npl[model->layers - 2] + 1; j++) {
            sum += nodes[model->layers - 2][j] * model->weigths[model->layers - 2][j][i];
        }
        nodes[model->layers - 1][i] = tanh(sum);
    }

    return nodes;
}

void displayNodes(MlpModel *model, double **nodes) {
    std::cout << "Nodes :" << std::endl << "[ " << std::endl;
    for(int l = 0; l < model->layers; l++) {
        std::cout << "\t[ ";
        auto nodesOnThisLayer = l == model->layers - 1 ? model->npl[l] : model->npl[l] + 1;
        for(int i = 0; i < nodesOnThisLayer; i++) {
            std::cout << nodes[l][i] << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv) {
//    int layers = 5;
//    int npl[] = {3, 4, 4, 2, 3};
    int layers = 3;
    int npl[] = {2, 3, 1};
    double *rawModel = mlpCreateModel(layers, npl);
    MlpModel *model = importMlpModel(rawModel);
    displayMlpModel(model);

//    displayMlpModel(importMlpModel(exportMlpModel(model)));

    double inParams[] = {3.14, 1.12};
    std::cout << mlpPredict(rawModel, inParams) << std::endl;

    double expected[] = {5.67};
    mlpClassTrain(rawModel, 0.1, 1, inParams, expected);

    return 0;
}