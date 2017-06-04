using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuronLayer
    {
        public List<INeuron> Neurons { get; set; }

        public int NumberOfInputs => Neurons.Select(n => n.Weights.Count).Distinct().Single();

        public List<double> Weights
        {
            get { return Neurons.SelectMany(n => n.Weights).ToList(); }
            set
            {
                for (var i = 0; i < Neurons.Count; i++)
                {
                    Neurons[i].Weights = value.Skip(i * NumberOfInputs).Take(NumberOfInputs).ToList();
                }
            }
        }

        public NeuronLayer(int numInputs, int numNeurons, double p, Random rand = null)
        {
            var random = rand ?? new Random();
            Neurons = new int[numNeurons].Select(i => new Neuron(numInputs, p, random)).Cast<INeuron>().ToList();
        }

        public List<double> Think(List<double> inputs)
        {
            return Neurons.Select(n => n.Think(inputs)).ToList();
        }

        public async Task<List<double>> ThinkAsync(List<double> inputs)
        {
            var result = await Task.WhenAll(Neurons.Select(n => Task.Run(() => n.Think(inputs))));
            return result.ToList();
        }
    }
}
