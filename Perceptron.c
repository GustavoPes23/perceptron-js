#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define PROPABILITY 0.9
#define BIAS 0.1

struct Neuron {
    double *weights;
    double bias;
};

void constructNeuron(struct Neuron *neuron, int inputSinapse) {
    neuron->weights = (double *)malloc(inputSinapse * sizeof(double));
    neuron->bias = BIAS;
}

void initializeWeights(struct Neuron *neuron, int inputSinapse) {
    for (int i = 0; i < inputSinapse + 1; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

int activation(double input) {
    return input >= 0 ? 1 : 0;
}

int sigmoid(double input) {
    return 1 / (1 + exp(-input));
}

int output(struct Neuron *neuron, const double *inputs, bool isTraining) {
    double sum = neuron->weights[0];

    for (int i = 0; i < 5; i++) {
        sum += inputs[i] * neuron->weights[i + 1];
    }

    if (isTraining) {
        return sigmoid(sum);
    }

    return activation(sum);
}

double calculateWeightDeltas(double error, double output) {
    return error * output * (1 - output);
}

double *calculateHiddenErrors(struct Neuron *neuron, const double *inputs, double delta) {
    double *hiddenLayerGradient = (double *)malloc(5 * sizeof(double));

    for (int i = 0; i < 5; i++) {
        hiddenLayerGradient[i] = delta * neuron->weights[i + 1];
    }

    return hiddenLayerGradient;
}

void updateWeights(struct Neuron *neuron, double delta, const double *inputs) {
    double learningRate = PROPABILITY;

    neuron->weights[0] += delta * neuron->bias * learningRate;
    for (int i = 0; i < 5; i++) {
        neuron->weights[i + 1] += delta * inputs[i] * neuron->bias * learningRate;
    }
}

void updateWeightsHiddenLayer(struct Neuron *neuron, const double *inputs, double *hiddenLayerGradient) {
    for (int i = 0; i < 5; i++) {
        neuron->weights[i + 1] += inputs[i] * (1 - inputs[i]) * neuron->bias;
    }
}

void backPropagation(struct Neuron *neuron, const double *inputs, int target) {
    double outputValue = output(neuron, inputs, true);
    double error = target - outputValue;
    double delta = calculateWeightDeltas(error, outputValue);

    updateWeights(neuron, delta, inputs);

    double *hiddenLayerGradient = calculateHiddenErrors(neuron, inputs, delta);

    updateWeightsHiddenLayer(neuron, inputs, hiddenLayerGradient);
    free(hiddenLayerGradient);
}

int main() {
    srand(time(NULL)); // Seed de inicialização para valores aleatórios

    struct Neuron perceptron;
    constructNeuron(&perceptron, 5);

    const int targetValues[] = {0, 0, 0, 0, 1, 1, 1, 1};
    const double trainingData[][5] = {
        {1, 1, 1, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {1, 0, 0, 1, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 1, 1},
        {0, 0, 0, 0, 1}
    };

    for (int i = 0; i < 10000; i++) {
        int randomIndex = rand() % 7;
        backPropagation(&perceptron, trainingData[randomIndex], targetValues[randomIndex]);
    }

    const char *translate[] = {"Cão", "Gato"};

    double pre = output(&perceptron, (double[]){1, 1, 0, 0, 0}, false);
    double pre2 = output(&perceptron, (double[]){1, 1, 1, 1, 1}, false);

    printf("Previsão: %s\n", translate[(int)pre]);
    printf("Previsão: %s\n", translate[(int)pre2]);

    free(perceptron.weights);
    return 0;
}