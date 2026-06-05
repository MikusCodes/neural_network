namespace NeuralNetworkApp.Core.Activation
{
    public interface IActivationFunction
    {
        double Calculate(double x, double beta);
        double CalculateDerivative(double fx, double beta);
    }
}