using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NeuralNetwork
{
    public class GeneticAlgorithm
    {
        public List<Genome> Genomes { get; set; }

        public int PopulationSize => Genomes.Count;

        public double TotalFitness => Genomes.Sum(g => g.Fitness);

        public double AverageFitness => TotalFitness / PopulationSize;

        public double MaxFitness => Genomes.Max(g => g.Fitness);

        public double MinFitness => Genomes.Min(g => g.Fitness);

        public double MutationRate { get; set; }

        public double CrossoverRate { get; set; }

        public int Generation { get; set; } = 0;

        public int Seed { get; set; } = 0;

        private Random random;

        private Random Rand => random ?? (random = new Random(Seed));

        public Tuple<int, Genome> GenomeRoulette()
        {
            var randomValue = Rand.NextDouble() * TotalFitness;
            for (var i = 0; i < Genomes.Count; i++)
            {
                randomValue -= Genomes[i].Fitness;
                if (randomValue <= 0) return new Tuple<int, Genome>(i, Genomes[i]);
            }
            return new Tuple<int, Genome>(Genomes.Count - 1, Genomes.Last());
        }

        public Tuple<Genome, Genome> Crossover(Genome mom, Genome dad)
        {
            if (mom.Genes.Count != dad.Genes.Count)
            {
                throw new ArgumentException("Incompatible parents");
            }

            var swapIndex = Rand.Next((int) Math.Round((mom.Genes.Count - 1) * CrossoverRate));

            var children = new Tuple<Genome, Genome>(new Genome(), new Genome());

            for (var i = 0; i < mom.Genes.Count; i++)
            {
                if (i > swapIndex)
                {
                    children.Item1.Genes.Add(dad.Genes[i]);
                    children.Item2.Genes.Add(mom.Genes[i]);
                }
                else
                {
                    children.Item1.Genes.Add(mom.Genes[i]);
                    children.Item2.Genes.Add(dad.Genes[i]);
                }
            }
            return children;
        }

        public Genome Mutate(Genome g)
        {
            for (var i = 0; i < g.Genes.Count; i++)
            {
                if (Rand.NextDouble() < MutationRate)
                {
                    g.Genes[i] = Rand.NextDouble();
                }
            }
            return g;
        }
    }
}
