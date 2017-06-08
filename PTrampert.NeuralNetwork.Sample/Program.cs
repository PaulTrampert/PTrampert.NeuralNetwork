using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace PTrampert.NeuralNetwork.Sample
{
    class Program
    {
        static void Main(string[] args)
        {
            var random = new Random();
            var brain = new Brain(0xFFFF, 1, 500, 6, rand: random);
            var bytes = new byte[0xffff];
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var result = brain.ThinkAsync(bytes.Select(Convert.ToDouble).ToList()).Result.Last();
            stopwatch.Stop();
            Console.WriteLine($"Brain took {stopwatch.Elapsed} to think.");
            Console.ReadLine();
        }

        static void EvoMain(string[] args)
        {
            var random = new Random();
            var evoEng = new EvolutionEngine(random)
            {
                CrossoverRate = 1,
                EliteCopyRate = 10,
                EliteThreshold = 4,
                MaxPerturbation = 1,
                MutationRate = .1,
                Population = new int[100].Select(i => new Brain(3, 1, 3, 1, p: .001, rand: random)).ToList()
            };

            var trainingSet = new int[1000].Select(i => new List<double> {random.Next(-10, 10), random.Next(-10, 10), 1}).ToList();
            var trainingAnswers = new int[1000].Select((i, idx) => Convert.ToDouble(trainingSet[idx][0] >= 2 * trainingSet[idx][1] ? 1 : 0)).ToList();
            while (true)
            {
                foreach (var brain in evoEng.Population)
                {
                    var error = 0.0;
                    for (var i = 0; i < trainingSet.Count; i++)
                    {
                        var output = brain.Think(trainingSet[i]).Last().First();
                        error += Math.Abs(trainingAnswers[i] - output);
                    }
                    brain.FitnessScore = 1 / Math.Pow(error, 10);
                }

                Console.WriteLine($"Generation: {evoEng.Generation}, AvgFitness: {evoEng.AvgFitness}, MaxFitness: {evoEng.MaxFitness}, MinFitness: {evoEng.MinFitness}");
                evoEng.Evolve();
            }
        }

        static void SavedBrainMain(string[] args)
        {
            var random = new Random();
            var json = File.ReadAllText("brain.json");
            var brain = JsonConvert.DeserializeObject<Brain>(json);
            var correct = 0;
            while (true)
            {
                for (var i = 0; i < 10000; i++)
                {
                    var inputs = new List<double> { random.Next(-1000, 1000), random.Next(-1000, 1000), -1 };
                    var correctOutput = inputs[0] >= 2 * inputs[1] ? 1 : 0;
                    var output = brain.Think(inputs).Last().First();
                    correct += Math.Abs(correctOutput - output) < .1 ? 1 : 0;
                }

                Console.WriteLine($"Brain got {correct} / 10000 inputs correct.");
                correct = 0;
            }
        }

        static void NewBrainMain(string[] args)
        {
            var random = new Random();
            var brain = new Brain(3, 5, 5, 1, learningRate: 1, rand: random);
            var correct = 0;
            while (true)
            {
                for (var i = 0; i < 10000; i++)
                {
                    var inputs = new List<double> {random.Next(-1000, 1000), random.Next(-1000, 1000), -1};
                    var correctOutput = inputs[0] >= 2*inputs[1] ? 1 : 0;
                    var output = brain.Think(inputs).Last().First();
                    brain.Learn(inputs, new List<double> {correctOutput});
                    correct += Math.Abs(correctOutput - output) < .1 ? 1 : 0;
                }

                Console.WriteLine($"After training, Brain got {correct} / 10000 inputs correct.");
                if (correct == 10000)
                {
                    File.WriteAllText("brain.json", JsonConvert.SerializeObject(brain, Formatting.Indented));
                    break;
                }
                correct = 0;
            }
            Console.ReadLine();
        }
    }
}