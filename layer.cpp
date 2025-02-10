#include <bits/stdc++.h>
#include <random>
using namespace std;

class Layer
{
private:
    // add more activation functions later on(this is sigmoid)
    double ActivationFunction(double val)
    {
        return max(0.01 * val, val);
        // return 1.0 / (1.0 + exp(-val));
    }

public:
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> weightGradients;
    vector<double> biasGradients;
    int nodesIn;
    int nodesOut;
    string activation;
    // backprop
    vector<double> z_values;
    vector<double> a_values;
    vector<double> dl_dz; // dl/dz

    Layer(int _nodesIn, int _nodesOut, string _activation) : weights(_nodesIn, vector<double>(_nodesOut)),
                                                             biases(_nodesOut),
                                                             weightGradients(_nodesIn, vector<double>(_nodesOut)),
                                                             biasGradients(_nodesOut),
                                                             nodesIn(_nodesIn), nodesOut(_nodesOut),
                                                             activation(_activation),
                                                             z_values(_nodesOut),
                                                             a_values(_nodesOut),
                                                             dl_dz(_nodesOut, 0) {}

    void GenerateRandomWeightsAndBiases()
    {
        random_device rd{};
        mt19937 gen{rd()};

        normal_distribution<double> d{0, sqrt(2.0 / nodesIn)};
        for (int i = 0; i < nodesIn; i++)
        {
            for (int j = 0; j < nodesOut; j++)
            {
                weights[i][j] = d(gen);
            }
        }

        for (int i = 0; i < nodesOut; i++)
        {
            biases[i] = d(gen);
        }
    }

    vector<double> CalculateLayerOutput(vector<double>& inputs)
    {
        for (int nodeOut = 0; nodeOut < nodesOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < nodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
            }

            z_values[nodeOut] = weightedInput;
            a_values[nodeOut] = ActivationFunction(weightedInput);
        }

        return a_values;
    }

    void UpdateWeightsAndBiases(double learnRate)
    {
        for (int i = 0; i < nodesOut; i++)
        {
            biases[i] -= biasGradients[i] * learnRate;
            for (int j = 0; j < nodesIn; j++)
            {
                weights[j][i] -= weightGradients[j][i] * learnRate;
            }
        }
    }

    void NormalizeGradients(int inputSize)
    {
        for (int i = 0; i < nodesIn; i++)
        {
            for (int j = 0; j < nodesOut; j++)
            {
                weightGradients[i][j] /= inputSize;
            }
        }

        for (int i = 0; i < nodesOut; i++)
        {
            biasGradients[i] /= inputSize;
        }
    }

    // currently for relu
    double ActivationDerivative(double val)
    {
        if (val < 0)
            return 0.01;
        else if (val == 0)
            return 0;
        else
            return 1;
    }
};
