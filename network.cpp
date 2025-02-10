#include <bits/stdc++.h>
#include <C:\Users\advai\Documents\Coding\C++\NN2\layer.cpp>

void DebugPrintVector(vector<double> &v)
{
    for (auto i : v)
    {
        cout << i << " ";
    }
    cout << endl;
}

class NeuralNetwork
{
private:
    double NodeLoss(double output, double expectedOutput)
    {
        double error = output - expectedOutput;
        return error * error;
    }

    double CalculateLossPerData(vector<double>& inputs, vector<double>& expectedOutputs)
    {
        double loss = 0;
        vector<double> outputs = CalculateOutputs(inputs);

        for (int i = 0; i < outputs.size(); i++)
        {
            loss += NodeLoss(outputs[i], expectedOutputs[i]);
        }

        return loss;
    }

    void CalculateGradientsBruteForce(vector<vector<double>>& xTrain, vector<vector<double>>& yTrain)
    {
        double h = 1e-6;
        double loss = CalculateTotalLoss(xTrain, yTrain);

        for (int l = 0; l < layers.size(); l++)
        {
            // for weights
            for (int nodeOut = 0; nodeOut < layers[l].nodesOut; nodeOut++)
            {
                for (int nodeIn = 0; nodeIn < layers[l].nodesIn; nodeIn++)
                {
                    layers[l].weights[nodeIn][nodeOut] += h;
                    double weightGrad = (CalculateTotalLoss(xTrain, yTrain) - loss) / h;
                    layers[l].weights[nodeIn][nodeOut] -= h;
                    layers[l].weightGradients[nodeIn][nodeOut] = weightGrad;
                }
            }

            // for biases
            for (int nodeOut = 0; nodeOut < layers[l].nodesOut; nodeOut++)
            {
                layers[l].biases[nodeOut] += h;
                double biasGrad = (CalculateTotalLoss(xTrain, yTrain) - loss) / h;
                layers[l].biases[nodeOut] -= h;
                layers[l].biasGradients[nodeOut] = biasGrad;
            }
        }
    }

    void CalculateLastLayerGradients(vector<double> &outputs, vector<double> &currSampleY)
    {
        // calculating dl_dz for last layer
        vector<double> dl_dz_last(layers[numLayers - 1].nodesOut);
        for (int nodeOut = 0; nodeOut < layers[numLayers - 1].nodesOut; nodeOut++)
        {
            double dl_da = 2 * (outputs[nodeOut] - currSampleY[nodeOut]);
            double z_val = layers[numLayers - 1].z_values[nodeOut];
            double da_dz = layers[numLayers - 1].ActivationDerivative(z_val);
            dl_dz_last[nodeOut] = dl_da * da_dz;
        }
        layers[numLayers - 1].dl_dz = dl_dz_last;
    }

    void CalculateBiasGradients(int l)
    {
        // calculating bias gradients
        for (int nodeOut = 0; nodeOut < layers[l].nodesOut; nodeOut++)
        {
            double dz_db = 1;
            layers[l].biasGradients[nodeOut] += layers[l].dl_dz[nodeOut] * dz_db;
        }
    }

    void CalculateWeightGradients(int l, vector<double> &currSampleX)
    {
        // calculating weight gradients
        for (int nodeIn = 0; nodeIn < layers[l].nodesIn; nodeIn++)
        {
            double dz_dw;
            if (l)
                dz_dw = layers[l - 1].a_values[nodeIn];
            else
                dz_dw = currSampleX[nodeIn];

            for (int nodeOut = 0; nodeOut < layers[l].nodesOut; nodeOut++)
            {
                layers[l].weightGradients[nodeIn][nodeOut] += layers[l].dl_dz[nodeOut] * dz_dw;
            }
        }
    }

    void CalculatePrevLayer_dl_dz(int l)
    {
        if (l == 0)
            return;

        // for dl_dz of prev layer
        vector<double> dl_dz_prev(layers[l - 1].nodesOut, 0);
        for (int nodeIn = 0; nodeIn < layers[l].nodesIn; nodeIn++)
        {
            for (int nodeOut = 0; nodeOut < layers[l].nodesOut; nodeOut++)
            {
                double dl_dz = layers[l].dl_dz[nodeOut];
                double dz_daPrev = layers[l].weights[nodeIn][nodeOut];
                double zPrev_val = layers[l - 1].z_values[nodeIn];
                double daPrev_dzPrev = layers[l - 1].ActivationDerivative(zPrev_val);
                dl_dz_prev[nodeIn] += dl_dz * dz_daPrev * daPrev_dzPrev;
            }
        }

        layers[l - 1].dl_dz = dl_dz_prev;
    }

    void NormalizeLayerGradients(int numSamples)
    {
        for (int l = numLayers - 1; l >= 0; l--)
        {
            layers[l].NormalizeGradients(numSamples);
        }
    }

    // using backpropogation
    void CalculateGradients(vector<vector<double>>& xTrain, vector<vector<double>>& yTrain)
    {
        int numSamples = xTrain.size();
        // running for all inputs
        for (int sample = 0; sample < numSamples; sample++)
        {
            //  first pass to get activation values for all layers
            vector<double> outputs = CalculateOutputs(xTrain[sample]);

            CalculateLastLayerGradients(outputs, yTrain[sample]);

            for (int l = numLayers - 1; l >= 0; l--)
            {
                CalculateBiasGradients(l);
                CalculateWeightGradients(l, xTrain[sample]);
                CalculatePrevLayer_dl_dz(l);
            }
        }

        NormalizeLayerGradients(numSamples);
    }

    void UpdateAllWeightsAndBiases(double learnRate)
    {
        for (int l = 0; l < layers.size(); l++)
        {
            layers[l].UpdateWeightsAndBiases(learnRate);
        }
    }

public:
    vector<Layer> layers;
    int numLayers;

    /*
    first layer is input layer
    */
    NeuralNetwork(vector<pair<int, string>> LayerProperties, int outputShape) : layers(LayerProperties.size(), {0, 0, ""}), numLayers(LayerProperties.size())
    {
        for (int layerInd = 0; layerInd < LayerProperties.size(); layerInd++)
        {
            pair<int, string> currProperties = LayerProperties[layerInd];
            int nodesOut = (layerInd == LayerProperties.size() - 1) ? outputShape : LayerProperties[layerInd + 1].first;
            Layer layer(currProperties.first, nodesOut, currProperties.second);

            layers[layerInd] = layer;
        }
    }

    void InitNetwork()
    {
        for (int i = 0; i < layers.size() - 1; i++)
        {
            layers[i].nodesOut = layers[i + 1].nodesIn;
        }
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].GenerateRandomWeightsAndBiases();
        }
    }

    //DONT PASS BY REFRENCE HERE
    vector<double> CalculateOutputs(vector<double> inputs)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            inputs = layers[i].CalculateLayerOutput(inputs);
        }

        return inputs;
    }

    double CalculateTotalLoss(vector<vector<double>>& dataInputs, vector<vector<double>>& expectedDataOutputs)
    {
        double loss = 0;
        for (int i = 0; i < dataInputs.size(); i++)
        {
            loss += CalculateLossPerData(dataInputs[i], expectedDataOutputs[i]);
        }
        return loss / dataInputs.size();
    }

    void Fit(vector<vector<double>>& xTrain, vector<vector<double>>& yTrain, double learnRate, int epochs)
    {
        for (int e = 1; e <= epochs; e++)
        {
            cout << "------epoch " << e << "------" << endl;

            CalculateGradients(xTrain, yTrain);
            UpdateAllWeightsAndBiases(learnRate);

            cout << "Loss: " << CalculateTotalLoss(xTrain, yTrain) << endl;
        }
    }

    vector<vector<double>> Predict(vector<vector<double>>& xTest)
    {
        vector<vector<double>> predictions;

        for (auto data : xTest)
        {
            predictions.push_back(CalculateOutputs(data));
        }

        return predictions;
    }

    void PrintWeights()
    {
        for (int l = 0; l < layers.size(); l++)
        {
            Layer layer = layers[l];
            cout << "L" << l << endl;
            for (int i = 0; i < layer.nodesIn; i++)
            {
                for (int j = 0; j < layer.nodesOut; j++)
                {
                    cout << "w" << i << j << "->" << layer.weights[i][j] << endl;
                }
            }
            cout << endl;
        }
    }
    void PrintBiases()
    {
        for (int l = 0; l < layers.size(); l++)
        {
            Layer layer = layers[l];
            cout << "L" << l << endl;
            for (int i = 0; i < layer.nodesOut; i++)
            {
                cout << "b" << i << "->" << layer.biases[i] << endl;
            }
            cout << endl;
        }
    }
    void PrintWeightGradients()
    {
        for (int l = 0; l < layers.size(); l++)
        {
            Layer layer = layers[l];
            cout << "L" << l << endl;
            for (int i = 0; i < layer.nodesIn; i++)
            {
                for (int j = 0; j < layer.nodesOut; j++)
                {
                    cout << "wG" << i << j << "->" << layer.weightGradients[i][j] << endl;
                }
            }
            cout << endl;
        }
    }
    void PrintBiasGradients()
    {
        for (int l = 0; l < layers.size(); l++)
        {
            Layer layer = layers[l];
            cout << "L" << l << endl;
            for (int i = 0; i < layer.nodesOut; i++)
            {
                cout << "bG" << i << "->" << layer.biasGradients[i] << endl;
            }
            cout << endl;
        }
    }
};

int main()
{
    auto start = chrono::high_resolution_clock::now();

    NeuralNetwork model({{2, ""}, {32, ""}, {128, ""}, {32, ""}}, 1);
    model.InitNetwork();

    vector<vector<double>> xTrain = {{1, 2}, {3, 6}, {8, 4}, {2, 5}};
    vector<vector<double>> yTrain = {{3}, {9}, {12}, {7}};
    model.Fit(xTrain, yTrain, 0.001, 100);

    cout << "---- TRAINING DONE ----" << endl;

    vector<vector<double>> toPredict = {{1.1, 3}, {5, 2}, {2.3, 4}};
    vector<vector<double>> predictions = model.Predict(toPredict);
    for (auto i : predictions)
    {
        DebugPrintVector(i);
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Time taken: " << duration.count() << endl;
}