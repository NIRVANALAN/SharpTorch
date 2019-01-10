﻿namespace cs_nn_fm
{
    public abstract class Layer
    {
    }


    public abstract class ActivationLayer : Layer
    {
        public int Dimension { get; set; }
        //        protected double[] LayerSum;
        public abstract double Differentiate(double x);
        public abstract double[] Calculate(ref double[] LayerSum);
    }

    public abstract class PropogationLayer : Layer
    {
        public int DIn { get; set; }
        public int DOut { get; set; }
        public double[,] Weights; // 2-d array to store Input-Output weights + biases
        public double[,] Grads;
        public double[] Signals; // error gradients signals
        public double[,] PrevWeightsDelta; //for momentum

        protected PropogationLayer(int dIn, int dOut)
        {
            DIn = dIn;
            DOut = dOut;
        }
    }

    class HyperTan : ActivationLayer
    {
        public override double Differentiate(double x)
        {
            return (1 + x) * (1 - x); //derivative for hyperbolic tan
            throw new System.NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.HyperTan(LayerSum[i]);
            }

            return LayerSum;
        }
    }

    class ReLU : ActivationLayer
    {
        public override double Differentiate(double x)
        {
            return x > 0 ? 1.0 : 0;
        }

        public override double[] Calculate(ref double[] LayerSum)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.Relu(LayerSum[i]);
            }

            return LayerSum;
        }
    }



    class Linear : PropogationLayer
    {
        public Linear(int numInput, int numOutput) : base(numInput,numOutput)
        {
//            DIn = numInput;
//            DOut = numOutput;
            Weights = Helper.MakeMatrix(DIn + 1, DOut); // include biases
            Helper.InitializeWeights(ref Weights);
            Grads = Helper.MakeMatrix(DIn + 1, DOut);
            PrevWeightsDelta = Helper.MakeMatrix(DIn + 1, DOut);
            Signals = new double[DOut]; // gradients output signals
        }
    }
}