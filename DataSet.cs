using System;

namespace cs_nn_fm
{
    public abstract class Dataset
    {
//        protected Dataset(string root = "", string datatxt = null, )
//        {
//        }

        public int InputNum { get; set; }
        public int OutputNum { get; set; }
        public double[][] DataSet { get; set; }
        protected int NumItems;
        protected int RandSeed;
        protected Random Rnd;
        protected double[][][] DataSet3D;
        public abstract double[] GetItems(int index);
        public abstract int GetLen();
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

        public SinTrainData(int num_items = 1000, double[][] dataSet = null, int seed = 1, bool provideDataFlag = false)
        {
            RandSeed = seed;
            Rnd = new Random(RandSeed);
            if (provideDataFlag)
            {
                DataSet = dataSet;
                NumItems = DataSet.Length;
            }
            else
            {
                NumItems = num_items;
                DataSet = new double[num_items][];
                for (int i = 0; i < NumItems; i++)
                {
                    var x = 2 * Math.PI * Rnd.NextDouble();
                    var sinX = Math.Sin(x);
                    DataSet[i] = new[] {x, sinX};
                }
            }

 
        }

        public override double[] GetItems(int index) //按顺序读取每个元素的具体内容
        {
//            var index = Rnd.Next(NumItems-1);
            return DataSet[index];
        }

        public override int GetLen() //返回数据集的长度
        {
            return DataSet.Length;
        }
    }
}