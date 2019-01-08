using System;

namespace cs_nn_fm
{
    public abstract class Dataset
    {
        public int InputNum { get; set; }
        public int OutputNum { get; set; }
        public double[][] dataSet { get; set; }
        protected int NumItems;
        protected int RandSeed;
        protected Random Rnd;
        protected double[][][] DataSet3D;
        public abstract double[] GetItems(int index);
        public abstract int GetLen();
//        public double[][] GetDataSet2D()
//        {
//            return DataSet2D;
//        }
    }

    public abstract class DataSet3D : Dataset
    {
        //TODO
    }

    public abstract class DataSet2D : Dataset
    {
        //TODO

    }


    class SinTrainData : DataSet2D
    {
//        private int numItems;
        //        double[][] trainData;

        public SinTrainData(int num_items, int seed)
        {
            RandSeed = seed;
            NumItems = num_items;
            Rnd = new Random(RandSeed);
            dataSet = new double[num_items][];
            for (int i = 0; i < NumItems; i++)
            {
                var x = 2 * Math.PI * Rnd.NextDouble();
                var sinX = Math.Sin(x);
                dataSet[i] = new[] { x, sinX };
            }
        }

        public override double[] GetItems(int index) //TODO
        {
            return dataSet[index];
        }

        public override int GetLen()
        {
            return dataSet.Length;
        }
    }

}