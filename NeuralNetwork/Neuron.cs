using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; set; }

        public Neuron(int inputs, Random rand = null)
        {
            var random = rand ?? new Random();
            Weights = new int[inputs].Select(i => random.NextDouble()).ToList();
        }

        public double Think(List<double> inputs)
        {
            return Sigmoid(inputs.Select((n, i) => n * Weights[i]).Sum());
        }

        private double Sigmoid(double weightedSum)
        {
            return 1 / (1 + Math.Exp(-weightedSum));
        }
    }
}
