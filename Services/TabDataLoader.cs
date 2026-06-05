using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;

namespace NeuralNetworkApp.Core.Services
{
    public class TabDataLoader
    {
        public static (double[][] inputs, double[][] targets) LoadFromTabFile(string filePath, int inputCount, int outputCount)
        {
            var inputsList = new List<double[]>();
            var targetsList = new List<double[]>();

            foreach (var line in File.ReadLines(filePath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] tokens = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length < inputCount + outputCount) continue;
                double[] input = new double[inputCount];
                double[] target = new double[outputCount];
                for (int i = 0; i < inputCount; i++)
                {
                    input[i] = double.Parse(tokens[i], CultureInfo.InvariantCulture);
                }
                for (int i = 0; i < outputCount; i++)
                {
                    target[i] = double.Parse(tokens[inputCount + i], CultureInfo.InvariantCulture);
                }

                inputsList.Add(input);
                targetsList.Add(target);
            }

            return (inputsList.ToArray(), targetsList.ToArray());
        }
    }
}