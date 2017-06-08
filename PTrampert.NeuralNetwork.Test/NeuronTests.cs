using System;
using System.Collections.Generic;
using Xunit;

namespace PTrampert.NeuralNetwork.Test
{
    public class NeuronTests
    {
        private Neuron subject;
        private Random random;

        public NeuronTests()
        {
            random = new Random(0);
            subject = new Neuron(3, rand: random);
        }

        [Fact]
        public void ItReturnsValueBetweenZeroAndOne()
        {
            for (var i = 0; i < 100; i++)
            {
                var inputs = new List<double>
                {
                    random.Next(-1000, 1000),
                    random.Next(-1000, 1000),
                    random.Next(-1000, 1000)
                };
                var result = subject.Think(inputs);
                Assert.True(result <= 1);
                Assert.True(result >= 0);
            }
        }

        [Fact]
        public void ClonedNeuronReturnsSameOutputAsOld()
        {
            var clone = subject.Clone();
            for (var i = 0; i < 100; i++)
            {
                var inputs = new List<double>
                {
                    random.Next(-1000, 1000),
                    random.Next(-1000, 1000),
                    random.Next(-1000, 1000)
                };
                var oldResult = subject.Think(inputs);
                var cloneResult = clone.Think(inputs);
                Assert.Equal(oldResult, cloneResult);
            }
        }
    }
}
