using System;
using NeuralNetworkApp.Core.Activation;

namespace NeuralNetworkApp.Core.Models
{
    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        public Layer(int neuronCount, int inputCount, Random rand)
        {
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(inputCount, rand);
            }
        }

        public double[] FeedForward(double[] inputs, IActivationFunction activation, double beta)
        {
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                double sum = Neurons[i].Weights[0];
                for (int j = 0; j < inputs.Length; j++)
                {
                    sum += Neurons[i].Weights[j + 1] * inputs[j];
                }
                Neurons[i].Sum = sum;
                Neurons[i].Output = activation.Calculate(sum, beta);
                outputs[i] = Neurons[i].Output;
            }
            return outputs;
        }
    }
}