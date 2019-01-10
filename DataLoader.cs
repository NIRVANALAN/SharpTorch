namespace cs_nn_fm
{
    public class DataLoader//在训练神经网络时，最好是对一个batch的数据进行操作，
        //同时还需要对数据进行shuffle和并行加速等。
        //对此，PyTorch提供了DataLoader帮助我们实现这些功能。
    {
        private Dataset dataset;
        private int batchSize;
        private bool shuffle;//是否将数据打乱
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