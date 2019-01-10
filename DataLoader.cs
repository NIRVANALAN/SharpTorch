namespace cs_nn_fm
{
    public class DataLoader
    {
        private Dataset dataset;
        private int batchSize;
        private bool shuffle;
        private int numWorkers;

        public DataLoader(Dataset dataset, int batchSize, bool shuffle=false, int workers=2)
        {
            this.dataset = dataset;
            this.batchSize = batchSize;
            this.shuffle = shuffle;
            numWorkers = workers;
        }

//        public  TODO
    }
}