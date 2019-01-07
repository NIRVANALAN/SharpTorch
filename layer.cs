namespace cs_nn_fm
{
    public abstract class layer
    {
        protected int DIn;
        protected int DOut;

        public int GetDIn()
        {
            return DIn;
        }
        public int GetDOut()
        {
            return DOut;
        }

//        protected double[] Biases;
    }

    public abstract class ActivationLayer : layer
    {
//        protected double[] LayerSum;
        public abstract void Calculate(ref double[] LayerSum);
    }

    public abstract class PropogationLayer : layer
    {
        public double[,] Weights; // 2-d array to store Input-Output weights + biases
    }

    class ReLU : ActivationLayer
    {

//        public ReLU(double[] layerSum)
//        {
//            LayerSum = layerSum;
//            DIn = DOut = LayerSum.Length;
//        }

        public override void Calculate(ref double[] LayerSum)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.Relu(LayerSum[i]);
            }

//            return LayerSum;
        }
    }
    class Linear : PropogationLayer
    {
        public Linear(int numInput, int numOutput)
        {
            DIn = numInput;
            DOut = numOutput;
            Weights = Helper.MakeMatrix(DIn+1,DOut,0.0); // include biases
            Helper.InitializeWeights(ref Weights);

        }
    }
}