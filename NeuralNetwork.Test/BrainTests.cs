using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Xunit;

namespace NeuralNetwork.Test
{
    public class BrainTests
    {
        private Brain subject;
        private Random random;
        private int inputCount = 50;
        private int neuronCount = 50;
        private int layers = 5;

        public BrainTests()
        {
            random = new Random(0);
            subject = new Brain(inputCount, layers, neuronCount, 5, rand:random);
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
