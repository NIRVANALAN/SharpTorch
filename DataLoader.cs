using System;

namespace cs_nn_fm
{
    public class DataLoader
    {
        private Dataset _dataSet;
        private int _batchSize;
        private bool _shuffle;
        private int _workers;
        private int _index = 0;

        public DataLoader(Dataset dataSet, int batchSize, bool shuffle = false, int workers = 1)
        {
            this._dataSet = dataSet;
            this._batchSize = batchSize;
            this._shuffle = shuffle;
            _workers = workers;
            if (_shuffle)
            {
                Shuffle(_dataSet);
            }
        }

        private void Shuffle(Dataset dataset)
        {
            var rnd = new Random(1);
            var allData = dataset.DataSet;
            for (int i = 0; i < allData.Length; i++)
            {
                var r = rnd.Next(i, allData.Length);
                var tmp = allData[r];
                allData[r] = allData[i];
                allData[i] = tmp;
            }
        }

        public double[][] Enumerate(bool label = false) //TODO label problem
        {
            var res = new double[_batchSize][];
            var end = _index + _batchSize;
            for (int i = 0; _index < end; i++)
            {
                res[i] = _dataSet.GetItems(_index);
                _index++;
            }
            return res;
        }

        //        public  TODO
    }
}