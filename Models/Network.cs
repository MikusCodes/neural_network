using System;
using System.Collections.Generic;
using NeuralNetworkApp.Core.Activation;

namespace NeuralNetworkApp.Core.Models
{
    public class Network
    {
        public List<Layer> Layers { get; private set; }
        private readonly IActivationFunction _activation;
        private readonly double _beta;

        public Network(int inputCount, int[] hiddenLayers, int outputCount, IActivationFunction activation, double beta)
        {
            Layers = new List<Layer>();
            _activation = activation;
            _beta = beta;
            Random rand = new Random();

            int currentInputs = inputCount;
            foreach (int hiddenCount in hiddenLayers)
            {
                Layers.Add(new Layer(hiddenCount, currentInputs, rand));
                currentInputs = hiddenCount;
            }
            Layers.Add(new Layer(outputCount, currentInputs, rand));
        }

        public double[] Compute(double[] inputs)
        {
            double[] currentOutputs = inputs;
            foreach (var layer in Layers)
            {
                currentOutputs = layer.FeedForward(currentOutputs, _activation, _beta);
            }
            return currentOutputs;
        }

        public void Backpropagate(double[] inputs, double[] targets, double learningRate)
        {
            Compute(inputs);

            Layer outputLayer = Layers[Layers.Count - 1];
            for (int i = 0; i < outputLayer.Neurons.Length; i++)
            {
                double error = targets[i] - outputLayer.Neurons[i].Output;
                outputLayer.Neurons[i].Delta = error * _activation.CalculateDerivative(outputLayer.Neurons[i].Output, _beta);
            }

            for (int k = Layers.Count - 2; k >= 0; k--)
            {
                Layer currentLayer = Layers[k];
                Layer nextLayer = Layers[k + 1];

                for (int i = 0; i < currentLayer.Neurons.Length; i++)
                {
                    double error = 0.0;
                    for (int m = 0; m < nextLayer.Neurons.Length; m++)
                    {
                        error += nextLayer.Neurons[m].Delta * nextLayer.Neurons[m].Weights[i + 1];
                    }
                    currentLayer.Neurons[i].Delta = error * _activation.CalculateDerivative(currentLayer.Neurons[i].Output, _beta);
                }
            }

            double[] currentLayerInputs = inputs;
            for (int k = 0; k < Layers.Count; k++)
            {
                Layer layer = Layers[k];
                for (int i = 0; i < layer.Neurons.Length; i++)
                {
                    layer.Neurons[i].Weights[0] += 2.0 * learningRate * layer.Neurons[i].Delta * 1.0;
                    for (int j = 0; j < currentLayerInputs.Length; j++)
                    {
                        layer.Neurons[i].Weights[j + 1] += 2.0 * learningRate * layer.Neurons[i].Delta * currentLayerInputs[j];
                    }
                }
                currentLayerInputs = new double[layer.Neurons.Length];
                for (int i = 0; i < layer.Neurons.Length; i++)
                {
                    currentLayerInputs[i] = layer.Neurons[i].Output;
                }
            }
        }
    }
}