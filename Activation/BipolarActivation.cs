using System;

namespace NeuralNetworkApp.Core.Activation
{
    public class BipolarActivation : IActivationFunction
    {
        public double Calculate(double x, double beta)
        {
            double expVal = Math.Exp(-beta * x);
            return (1.0 - expVal) / (1.0 + expVal);
        }

        public double CalculateDerivative(double fx, double beta)
        {
            return beta * (1.0 - fx * fx);
        }
    }
}