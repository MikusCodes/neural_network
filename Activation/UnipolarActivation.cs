using System;

namespace NeuralNetworkApp.Core.Activation
{
    public class UnipolarActivation : IActivationFunction
    {
        public double Calculate(double x, double beta)
        {
            return 1.0 / (1.0 + Math.Exp(-beta * x));
        }

        public double CalculateDerivative(double fx, double beta)
        {
            return beta * fx * (1.0 - fx);
        }
    }
}