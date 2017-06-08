using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace PTrampert.NeuralNetwork.Test
{
    public class NeuronLayerTests
    {
        private NeuronLayer subject;
        private Random random;
        private int inputCount = 50;
        private int neuronCount = 50;

        public NeuronLayerTests()
        {
            random = new Random(0);
            subject = new NeuronLayer(inputCount, neuronCount, rand:random);
        }

        [Fact]
        public void ClonedLayerReturnsSameOutputAsOriginal()
        {
            var clone = subject.Clone();
            for (var i = 0; i < 100; i++)
            {
                var inputs = GetInputs();
                var origResult = subject.Think(inputs);
                var cloneResult = clone.Think(inputs);
                Assert.Equal(origResult, cloneResult);
            }
        }

        private List<double> GetInputs()
        {
            return new double[inputCount].Select(i => Convert.ToDouble(random.Next(-1000, 1000))).ToList();
        }
    }
}
