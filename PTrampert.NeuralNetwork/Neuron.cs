using System;
using System.Collections.Generic;
using System.Linq;

namespace PTrampert.NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public ActivationFunction ActivationFunction { get; set; }
        public double P { get; set; }
        public double Delta { get; set; }
        public double Output { get; set; }
        public double LearningRate { get; set; }

        private static readonly Dictionary<ActivationFunction, FuncAndDeriv> ActivationFunctions = new Dictionary<ActivationFunction, FuncAndDeriv>
        {
            { ActivationFunction.Sigmoid, new FuncAndDeriv { Func = Sigmoid, Deriv = SigmoidPrime } },
            { ActivationFunction.Tanh, new FuncAndDeriv { Func = Tanh, Deriv = TanhPrime } }
        };

        public Neuron()
        {
            
        }

        public Neuron(int inputs, ActivationFunction activationFunction = ActivationFunction.Sigmoid, double learningRate = .1, double p = 1, Random rand = null)
        {
            ActivationFunction = activationFunction;
            var random = rand ?? new Random();
            Weights = new int[inputs].Select(i => 2 * random.NextDouble() - 1).ToList();
            LearningRate = learningRate;
            P = p;
        }

        public double Think(List<double> inputs)
        {
            return Output = ActivationFunctions[ActivationFunction].Func(inputs.Select((n, i) => n * Weights[i]).Sum(), P);
        }

        public double CalculateDelta(List<double> inputs, double error)
        {
            return Delta = error * ActivationFunctions[ActivationFunction].Deriv(Output);
        }

        public void UpdateWeights(List<double> inputs)
        {
            Weights = Weights.Select((w, i) => w + (LearningRate * Delta * inputs[i])).ToList();
        }

        private static double SigmoidPrime(double output)
        {
            return output * (1 - output);
        }

        private static double Sigmoid(double weightedSum, double p)
        {
            return 1 / (1 + Math.Exp(-weightedSum/p));
        }

        private static double Tanh(double weightedSum, double p)
        {
            return (1 - Math.Exp(-2 * weightedSum / p)) / (1 + Math.Exp(-2 * weightedSum / p));
        }

        private static double TanhPrime(double output)
        {
            return 1 - Math.Pow(output, 2);
        }

        public Neuron Clone(Random rand = null)
        {
            return new Neuron
            {
                Weights = Weights,
                LearningRate = LearningRate,
                ActivationFunction = ActivationFunction,
                P = P
            };
        }

        private class FuncAndDeriv
        {
            public Func<double, double, double> Func { get; set; }
            public Func<double, double> Deriv { get; set; }
        }
    }

    public enum ActivationFunction
    {
        Sigmoid,
        Tanh
    }
}
