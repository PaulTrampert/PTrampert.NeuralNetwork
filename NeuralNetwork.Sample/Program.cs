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
            var random = new Random(0);
            var populationSize = 10;
            var brains = new int[populationSize]
                .Select(i => new Brain(3, 1, 1, 1, 1, random))
                .ToList();
            var alg = new GeneticAlgorithm
            {
                Genomes = brains.Select(b => new Genome {Genes = b.Genes}).ToList(),
                Seed = 0,
                CrossoverRate = .7,
                MutationRate = .2
            };

            var trainingSet = new List<List<double>>
            {
                new List<double>{1,1,1},
                new List<double>{1,0,1},
                new List<double>{0,1,1},
                new List<double>{0,0,1}
            };

            var resultSet = new List<double> {0, 1, 1, 0};

            Console.WriteLine("Generation,Average Fitness,Max Fitness,Min Fitness");

            while (true)
            {
                for (var i = 0; i < brains.Count; i++)
                {
                    var brain = brains[i];
                    var genome = alg.Genomes[i];
                    var ti = trainingSet[i % 4];
                    var ta = resultSet[i % 4];
                    var result = brain.ThinkAsync(ti).Result.Single();
                    var error = ta - result;
                    genome.Fitness = 1 / Math.Abs(error);
                }
                Console.WriteLine($"{alg.Generation},{alg.AverageFitness},{alg.MaxFitness},{alg.MinFitness}");

                foreach (var brain in brains)
                {
                    var mom = alg.GenomeRoulette();
                    var dad = alg.GenomeRoulette();
                    var child = alg.Crossover(mom, dad);
                    child = alg.Mutate(child);
                    brain.Genes = child.Genes;
                }
                alg.Genomes = brains.Select(b => new Genome {Genes = b.Genes}).ToList();
                alg.Generation++;
            }
        }
    }
}