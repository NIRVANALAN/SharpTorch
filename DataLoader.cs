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
        public int loopTime { get; set; }
        private int[] sequence;

        public DataLoader(Dataset dataSet, int batchSize, bool shuffle = false, int workers = 1)
        {
            this._dataSet = dataSet;
            this._batchSize = batchSize;
            this._shuffle = shuffle;
            _workers = workers;
            if (_shuffle)
            {
                sequence = new int[_dataSet.GetLen()];
                for (int i = 0; i < sequence.Length; i++)
                {
                    sequence[i] = i;
                }

                Shuffle(sequence);
            }

            loopTime = (int) (dataSet.GetLen() / batchSize);
        }

        private void Shuffle(int[] sequence)
        {
            var rnd = new Random(1);
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        public double[][] Enumerate(bool label = false) //TODO label problem
        {
            if (_index >= _dataSet.GetLen())
            {
                _index = 0;
                Shuffle(sequence); //shuffle
            }

            var res = new double[_batchSize][];
            var end = _index + _batchSize;
            for (int i = 0; _index < end && _index < _dataSet.GetLen(); i++)
            {
                if (_shuffle)
                {
                    res[i] = _dataSet.GetItems(sequence[_index]);
                }
                else
                {
                    res[i] = _dataSet.GetItems(_index);
                }

                _index++;
            }

            return res;
        }

        //        public  TODO
    }
}