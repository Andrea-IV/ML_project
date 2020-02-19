#include "library.h"
#include <chrono>
#include <random>

#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

//[type_model][class/regression][fonction]

extern "C" {
__declspec(dllexport) double * linearCreateModel(int inDim, int outDim) {
    int weightsSize = (inDim + 1) * outDim + 2;
    auto rawModel = (double *)(malloc(weightsSize * sizeof(double)));
    rawModel[0] = inDim;
    rawModel[1] = outDim;
    int rawModelIndex = 2;

    std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution{-1, 1};

    for(int i = 0; i < weightsSize; i++) {
        rawModel[rawModelIndex++] = i; //distribution(randomEngine);
    }

    return rawModel;
}

__declspec(dllexport) int* linearClassPredict(double *model, int inDim, int outDim, double *params) {
    auto result = (int *) malloc(sizeof(int) * (outDim + 1));
    result[0] = outDim;

    for(int d = 0; d < outDim; d++) {
        double sum = model[d];
        for(int i = 0; i < inDim; i++) {
            sum += params[i] * model[i * outDim + outDim + d];
        }
        result[d + 1] = sign(sum);
    }

    return result;
}

__declspec(dllexport) double linearRegPredict(double *model, int inDim, double *params) {
    double sum = model[0];
    for(int i = 0; i < inDim; i++) {
        sum += params[i] * model[i + 1];
    }

    return sum;
}

__declspec(dllexport)void linearRegTrain(double *model, int inDim, double trainingStep,
                                         double *trainingParams, int trainingParamsNumber, const double *trainingResults) {

    std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution{0, 1};
    MatrixXd XMatrix(trainingParamsNumber, inDim + 1);
    MatrixXd YMatrix(trainingParamsNumber,1);
    for(int line = 0; line < trainingParamsNumber; line++) {
        YMatrix(line, 0) = trainingResults[line];
        XMatrix(line, 0) = 1;
        for(int column = 1; column < (inDim + 1); column++) {
            XMatrix(line, column) = trainingParams[(line * inDim) + (column - 1)];
        }
    }

    std::cout << "Matrice X :" << std::endl;
    std::cout << XMatrix << std::endl;
    std::cout << "Matrice Y :" << std::endl;
    std::cout << YMatrix << std::endl;
    MatrixXd resultMatrix = ((XMatrix.transpose() * XMatrix).inverse() * XMatrix.transpose()) * YMatrix;
    std::cout << "Resultat :" << std::endl;
    std::cout << resultMatrix << std::endl;

    for (int i = 0; i <= inDim; i++) {
        model[i] = resultMatrix(i, 0);
    }
}

__declspec(dllexport)void linearClassTrain(double *model, int inDim, int outDim, int epoch, double trainingStep,
                                           double *trainingParams, int trainingParamsNumber, const double *trainingResults) {

    std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution{0, 1};

    for(int e = 0; e < epoch; e++) {
        int trainingPicked = floor(distribution(randomEngine) * trainingParamsNumber);
        int trainingParamsPosition = inDim * trainingPicked;
        int *prediction = linearClassPredict(model, inDim, outDim, &trainingParams[trainingParamsPosition]);
        for(int i = 0; i < outDim; i++) {
            double modification = (double)trainingStep * (trainingResults[trainingPicked * outDim + i] - prediction[i + 1]);

            model[i] += modification;

            for(int j = 0; j < inDim; j++) {
                model[i + outDim * j] += modification * trainingParams[trainingParamsPosition + j];
            }
        }
    }
}

__declspec(dllexport) void linearClearModel(const double *model) {
    delete model;
}
}

int sign(double value) {
    return value == 0 ? 0 : value < 0 ? -1 : 1;
}

void linearClassPredictAll(double *model) {
    for(double i = -7; i < 8; i += 1) {
        for(double j = 0; j < 15; j += 1) {
            double params[] = {i, j};
            int predicted = linearClassPredict(model, 2, 1, params)[1];
            std::cout << (predicted == 0 ? "0" : predicted >= 1 ? "+" : "-");
        }
        std::cout << std::endl;
    }
}

LinearModel * importLinearModel(double *rawModel) {
    auto *model = (LinearModel *) malloc(sizeof(LinearModel));
    model->inDim = rawModel[0];
    model->outDim = rawModel[1];

    int rawModelIndex = 2;
    model->weights = (double **) malloc(sizeof(double *) * model->inDim + 1);
    for(int i = 0; i < model->inDim + 1; i++) {
        model->weights[i] = (double *) malloc(sizeof(double) * model->outDim);
        for(int j = 0; j < model->outDim; j++) {
            model->weights[i][j] = rawModel[rawModelIndex++];
        }
    }

    return model;
}

double * exportLinearModel(LinearModel * model) {
    auto rawModel = (double *) malloc(sizeof(double) * (model->inDim + 1) * model->outDim + 2);
    rawModel[0] = model->inDim;
    rawModel[1] = model->outDim;

    int rawModelIndex = 2;
    for(int i = 0; i < model->inDim + 1; i++) {
        for(int j = 0; j < model->outDim; j++) {
            rawModel[rawModelIndex++] = model->weights[i][j];
        }
    }

    return rawModel;
}

void displayLinearModel(LinearModel *model) {
    std::cout << "Linear model :" << std::endl <<
        "inDim : " << model->inDim << std::endl <<
        "outDim : " << model->outDim << std::endl <<
        "Weights :" << std::endl << "[" << std::endl;
    for(int i = 0; i < model->inDim + 1; i++) {
        std::cout << "\t[ ";
        for(int j = 0; j < model->outDim; j++) {
            std::cout << model->weights[i][j] << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv) {
    int inDim = 3;
    int outDim = 2;
//    double *model = linearCreateModel(inDim, outDim);
//    double trainingParams[] = {-4.2, 0.41, 3.77, 12.79, -5.45, 8.55};
//    double trainingExpects[] = {1, 1, -1};
//
//    linearClassTrain(model, inDim, outDim, 100000, 0.1, trainingParams, 3, trainingExpects);
//    linearClassPredictAll(model);

    displayLinearModel(importLinearModel(linearCreateModel(inDim, outDim)));

    std::cout << "test";
    return 0;
}