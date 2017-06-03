using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Perceptron
    {
        public List<double> Weights { get; set; }
        public double LearningConstant { get; set; }

        public Perceptron(int inputs, double learningConstant = 0.1, Random rand = null)
        {
            var random = rand ?? new Random();
            Weights = new int[inputs].Select(i => random.NextDouble()).ToList();
            LearningConstant = learningConstant;
        }

        public double Think(List<double> inputs)
        {
            return Binary(inputs.Select((n, i) => n * Weights[i]).Sum());
        }

        public void Learn(List<double> inputs, double error)
        {
            Weights = Weights.Select((w, i) => w + LearningConstant * error * inputs[i]).ToList();
        }

        private double Binary(double weightedSum)
        {
            return weightedSum >= 0 ? 1 : 0;
        }
    }

    public enum ActivationFunctions
    {
        Binary,
        Sigmoid
    }
}
