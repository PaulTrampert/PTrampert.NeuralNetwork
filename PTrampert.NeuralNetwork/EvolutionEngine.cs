using System;
using System.Collections.Generic;
using System.Linq;

namespace PTrampert.NeuralNetwork
{
    public class EvolutionEngine
    {
        public List<Brain> Population { get; set; }

        private Random random;

        public double MaxFitness => Population.Max(b => b.FitnessScore);

        public double MinFitness => Population.Min(b => b.FitnessScore);

        public double AvgFitness => TotalFitness / Population.Count;

        public double TotalFitness => Population.Sum(b => b.FitnessScore);

        public int EliteThreshold { get; set; }

        public int EliteCopyRate { get; set; }

        public double CrossoverRate { get; set; }

        public double MutationRate { get; set; }

        public double MaxPerturbation { get; set; }

        public int Generation { get; set; }

        public EvolutionEngine(Random rand)
        {
            random = rand;
        }

        public void Evolve()
        {
            var targetPopulation = Population.Count;
            var nextGen = GetElites();
            while (nextGen.Count < targetPopulation)
            {
                var mom = Roulette().Clone();
                var dad = Roulette().Clone();
                if (mom == dad)
                {
                    nextGen.Add(mom);
                    nextGen.Add(dad);
                    continue;
                }
                var children = Crossover(mom.Genes, dad.Genes);
                mom.Genes = Mutate(children.Item1);
                dad.Genes = Mutate(children.Item2);
                nextGen.Add(mom);
                nextGen.Add(dad);
            }
            if (nextGen.Count > targetPopulation)
            {
                nextGen.RemoveAt(nextGen.Count - 1);
            }
            Population = nextGen;
            Generation++;
        }

        private List<double> Mutate(List<double> genes)
        {
            return genes.Select(gene => random.NextDouble() > MutationRate
                    ? gene
                    : ((2 * random.NextDouble() - 1) * MaxPerturbation))
                .ToList();
        }

        private Brain Roulette()
        {
            var slice = random.NextDouble() * TotalFitness;
            foreach (var brain in Population)
            {
                slice -= brain.FitnessScore;
                if (slice <= 0)
                {
                    return brain;
                }
            }
            return Population.Last();
        }

        private Tuple<List<double>, List<double>> Crossover(List<double> mom, List<double> dad)
        {
            if (random.NextDouble() > CrossoverRate)
            {
                return new Tuple<List<double>, List<double>>(mom, dad);
            }

            var child1 = mom;
            var child2 = dad;
            var swapIndex = random.Next(child1.Count - 1);
            for (var i = swapIndex; i < child1.Count; i++)
            {
                var tmp = child1[i];
                child1[i] = child2[i];
                child2[i] = tmp;
            }
            return new Tuple<List<double>, List<double>>(child1, child2);
        }

        private List<Brain> GetElites()
        {
            var result = new List<Brain>();
            var sortedPop = Population.OrderByDescending(b => b.FitnessScore).ToList();
            for (var i = 0; i < EliteThreshold; i++)
            {
                for (var j = 0; j < EliteCopyRate; j++)
                {
                    result.Add(sortedPop[i].Clone());
                }
            }
            return result;
        }
    }
}
