using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron : INeuron
    {
        public List<double> Weights { get; set; }
        public double P { get; set; }

        public Neuron(int inputs, double p, Random rand = null)
        {
            var random = rand ?? new Random();
            Weights = new int[inputs].Select(i => 2 * random.NextDouble() - 1).ToList();
            P = p;
        }

        public double Think(List<double> inputs)
        {
            return Sigmoid(inputs.Select((n, i) => n * Weights[i]).Sum());
        }

        private double Sigmoid(double weightedSum)
        {
            return 1 / (1 + Math.Exp(-weightedSum/P));
        }
    }
}
