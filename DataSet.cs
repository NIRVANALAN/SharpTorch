using System;

namespace cs_nn_fm
{
    public abstract class Dataset
    {
        protected int NumItems = 0;
        protected int rand_seed;
        protected Random Rnd;
        protected double[][] DataSet2D;
        protected double[][][] DataSet3D;
        
//        public int GetNumItems()
//        {
//            return NumItems;
//        }
//        public void Set_rand_seed(int seed)
//        {
//            this.rand_seed = seed;
//        }
//
//        public void SetNumItems(int NumItems)
//        {
//            this.NumItems = NumItems;
//        }

        public abstract double[] GetItems(int index);
        public abstract int GetLen();

        public double[][] GetDataSet2D()
        {
            return DataSet2D;
        }


    }
}