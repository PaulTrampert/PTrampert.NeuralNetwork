using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public double P { get; set; }
        public double Delta { get; set; }
        public double Output { get; set; }
        public double LearningRate { get; set; }

        public Neuron()
        {
            
        }

        public Neuron(int inputs, double learningRate = .1, double p = 1, Random rand = null)
        {
            var random = rand ?? new Random();
            Weights = new int[inputs].Select(i => 2 * random.NextDouble() - 1).ToList();
            LearningRate = learningRate;
            P = p;
        }

        public double Think(List<double> inputs)
        {
            return Output = Sigmoid(inputs.Select((n, i) => n * Weights[i]).Sum());
        }

        public double CalculateDelta(List<double> inputs, double error)
        {
            return Delta = error * SigmoidPrime(Output);
        }

        public void UpdateWeights(List<double> inputs)
        {
            Weights = Weights.Select((w, i) => w + (LearningRate * Delta * inputs[i])).ToList();
        }

        private double SigmoidPrime(double output)
        {
            return output * (1 - output);
        }

        private double Sigmoid(double weightedSum)
        {
            return 1 / (1 + Math.Exp(-weightedSum/P));
        }

        public Neuron Clone(Random rand = null)
        {
            return new Neuron
            {
                Weights = Weights,
                LearningRate = LearningRate,
                P = P
            };
        }
    }
}
