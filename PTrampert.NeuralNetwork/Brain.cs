using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace PTrampert.NeuralNetwork
{
    public class Brain
    {
        public List<NeuronLayer> Layers { get; set; }

        public int NumberOfInputs => Layers.First().NumberOfInputs;

        public double FitnessScore { get; set; }

        public List<double> Genes
        {
            get => Layers.SelectMany(l => l.Weights).ToList();
            set
            {
                for (var i = 0; i < Layers.Count; i++)
                {
                    var layer = Layers[i];
                    for (var j = 0; j < layer.Neurons.Count; j++)
                    {
                        var neuron = layer.Neurons[j];
                        for (var k = 0; k < neuron.Weights.Count; k++)
                        {
                            neuron.Weights[k] = value[i + j + k];
                        }
                    }
                }
            }
        }

        public Brain() { }

        public Brain(int numInputs, int numHiddenLayers, int layerWidth, int numOutputs, double learningRate = .1, double p = 1, Random rand = null)
        {
            if (numHiddenLayers < 1)
            {
                throw new ArgumentException("Must have at least 1 hidden layer.", nameof(numHiddenLayers));
            }
            var random = rand ?? new Random();
            Layers = new List<NeuronLayer>();
            Layers.Add(new NeuronLayer(numInputs, layerWidth, learningRate, p, random));
            Layers.AddRange(new int[numHiddenLayers - 1].Select(i => new NeuronLayer(layerWidth, layerWidth, learningRate, p, random)).ToList());
            Layers.Add(new NeuronLayer(layerWidth, numOutputs, learningRate, p, random));
        }

        public Brain Clone()
        {
            return new Brain
            {
                Layers = Layers.Select(l => l.Clone()).ToList()
            };
        }

        public List<List<double>> Think(List<double> inputs)
        {
            var outputs = new List<List<double>> { inputs };
            for (var i = 0; i < Layers.Count; i++)
            {
                outputs.Add(Layers[i].Think(outputs[i]));
            }
            return outputs;
        }

        public async Task<List<List<double>>> ThinkAsync(List<double> inputs)
        {
            var outputs = new List<List<double>> {inputs};
            for (var i = 0; i < Layers.Count; i++)
            {
                outputs.Add(await Layers[i].ThinkAsync(outputs[i]));
            }
            return outputs;
        }

        public void Learn(List<double> inputs, List<double> correct)
        {
            BackpropErrors(inputs, correct);
            UpdateWeights(inputs);
        }

        private void BackpropErrors(List<double> inputs, List<double> correct)
        {
            for (var i = Layers.Count - 1; i >= 0; i--)
            {
                var layer = Layers[i];
                var errors = i != Layers.Count - 1
                    ? layer.Neurons.Select((n, j) => Layers[i + 1].Neurons.Sum(n1 => n1.Weights[j] * n1.Delta)).ToList()
                    : layer.Neurons.Select((n, j) => correct[j] - n.Output).ToList();
                for (var j = 0; j < layer.Neurons.Count; j++)
                {
                    layer.Neurons[j].CalculateDelta(inputs, errors[j]);
                }
            }
        }

        private void UpdateWeights(List<double> inputs)
        {
            var prevOutputs = inputs;
            foreach (var layer in Layers)
            {
                layer.UpdateWeights(prevOutputs);
                prevOutputs = layer.Neurons.Select(n =>
                {
                    var output = n.Output;
                    n.Output = 0;
                    n.Delta = 0;
                    return output;
                }).ToList();
            }
        }
    }
}
