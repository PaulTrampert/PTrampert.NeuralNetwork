using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Brain
    {
        public List<NeuronLayer> Layers { get; set; }

        public int NumberOfInputs => Layers.First().NumberOfInputs;

        public List<double> Genome => Layers.SelectMany(l => l.Weights).ToList();

        public Brain(int numInputs, int numHiddenLayers, int layerWidth, int numOutputs, Random rand = null)
        {
            if (numHiddenLayers < 1)
            {
                throw new ArgumentException("Must have at least 1 hidden layer.", nameof(numHiddenLayers));
            }
            var random = rand ?? new Random();
            Layers = new List<NeuronLayer>();
            Layers.Add(new NeuronLayer(numInputs, layerWidth, random));
            Layers = new int[numHiddenLayers - 1].Select(i => new NeuronLayer(layerWidth, layerWidth, random)).ToList();
            Layers.Add(new NeuronLayer(layerWidth, numOutputs, random));
        }

        public List<List<double>> Think(List<double> inputs)
        {
            var results = new List<List<double>>{inputs};
            foreach (var layer in Layers)
            {
                results.Add(layer.Think(results.Last()));
            }
            return results;
        }

        public async Task<List<double>> ThinkAsync(List<double> inputs)
        {
            var previousOutput = inputs;
            foreach (var layer in Layers)
            {
                previousOutput = await layer.ThinkAsync(previousOutput);
            }
            return previousOutput;
        }
    }
}
