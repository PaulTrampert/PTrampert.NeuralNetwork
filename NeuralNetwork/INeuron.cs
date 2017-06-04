using System.Collections.Generic;

namespace NeuralNetwork
{
    public interface INeuron
    {
        List<double> Weights { get; set; }
        double Think(List<double> inputs);
    }
}