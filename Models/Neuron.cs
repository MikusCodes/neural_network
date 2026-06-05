using System;

namespace NeuralNetworkApp.Core.Models
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Output { get; set; }
        public double Sum { get; set; }
        public double Delta { get; set; }

        public Neuron(int inputCount, Random rand)
        {
            Weights = new double[inputCount + 1];
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = rand.NextDouble() * 1.0 - 0.5;
            }
        }
    }
}