using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork.Sample
{
    class Program
    {
        static void Main(string[] args)
        {
            var random = new Random();
            var brain = new Brain(3, 1, 3, 1, learningRate: 1, rand: random);
            var correct = 0;
            Console.WriteLine("Beginning training...");
            for (var i = 0; i < 10000; i++)
            {
                var inputs = new List<double> {random.Next(-10, 10), random.Next(-10, 10), 1};
                var correctOutput = inputs[0] >= 2 * Math.Pow(inputs[1], 2) ? 1 : 0;
                brain.Think(inputs);
                brain.Learn(inputs, new List<double> {correctOutput});
            }
            Console.WriteLine("Baseline training complete.");
            while (true)
            {
                for (var i = 0; i < 10000; i++)
                {
                    var inputs = new List<double> {random.Next(-10, 10), random.Next(-10, 10), 1};
                    var correctOutput = inputs[0] >= 0 ? 1 : 0;
                    var output = brain.Think(inputs).Last().First();
                    brain.Learn(inputs, new List<double> {correctOutput});
                    correct += Math.Abs(correctOutput - output) < .1 ? 1 : 0;
                }

                Console.WriteLine($"After training, Brain got {correct} / 10000 inputs correct.");
                correct = 0;
            }
            Console.ReadLine();
        }
    }
}