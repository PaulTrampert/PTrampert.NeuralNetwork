using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Genome
    {
        public List<double> Genes { get; set; }

        public double Fitness { get; set; }

        public static bool operator <(Genome lhs, Genome rhs)
        {
            return lhs.Fitness < rhs.Fitness;
        }

        public static bool operator >(Genome lhs, Genome rhs)
        {
            return lhs.Fitness > rhs.Fitness;
        }
    }
}
