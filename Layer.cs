using System;

namespace cs_nn_fm
{

    public abstract class Layer
    {
    }

    public abstract class ActivationLayer : Layer
    {
        public int Dimension { get; set; }

        //        protected double[] LayerSum;
        public abstract double Differentiate(double x, double a = 0.0);
        public abstract double[] Calculate(ref double[] LayerSum, double a = 0.0);
    }

    public abstract class ConvLayer : Layer
    {
        // int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, int dilation=1,int group=1, bool bias=true,string padding_mode="zeros"
        public int in_channels { get; set; }
        public int out_channels { get; set; }
        public int kernel_size { get; set; }
        public int stride { get; set; } = 1;
        public int padding { get; set; } = 0;
        public int dilation { get; set; } = 1;
        public bool bias { get; set; } = true;
        public string padding_mode { get; set; } = "zeros";

        public double[][,] Weights; // 3-d array
        public double[][,] Grads;
        public double[][] Signals; // error gradients signals
        public double[][,] PrevWeightsDelta; //for momentum

        protected ConvLayer(int inChannels,int outChannels)
        {
            in_channels = inChannels;
            out_channels = outChannels;
        }

    }

    public abstract class LinearLayer : Layer
    {
        public int DIn { get; set; }
        public int DOut { get; set; }
        //public int 
        public double[,] Weights; // 2-d array to store Input-Output weights + biases
        public double[,] Grads;
        public double[] Signals; // error gradients signals
        public double[,] PrevWeightsDelta; //for momentum

        protected LinearLayer(int dIn, int dOut)
        {
            DIn = dIn;
            DOut = dOut;
        }
    }

//    public abstract class CostLayer : ActivationLayer
//    {
//
//    }
    //CostLayer below
    class SoftMax:ActivationLayer
    {
        public override double Differentiate(double x, double a = 0)
        {
            return (1 - x) * x;
//            throw new NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum, double a = 0)
        {
            return Activation.Softmax(LayerSum);
//            throw new NotImplementedException();
        }
    }


    // activationLayer below
    class HyperTan : ActivationLayer
    {
        public override double Differentiate(double x, double a)
        {
            return (1 + x) * (1 - x); //derivative for hyperbolic tan
            //throw new System.NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum, double a)
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
        public override double Differentiate(double x, double a)
        {
            return x > 0 ? 1.0 : 0;
        }

        public override double[] Calculate(ref double[] LayerSum, double a)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.Relu(LayerSum[i]);
            }

            return LayerSum;
        }
    }

    class Sigmoid : ActivationLayer
    {
        public override double Differentiate(double x, double a = 0)
        {
            return Activation.Sigmoid(x) * (1 - Activation.Sigmoid(x));
            //throw new NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum, double a = 0)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.Sigmoid(LayerSum[i]);
            }

            //throw new NotImplementedException();
            return LayerSum;
        }
    }

    class ELU : ActivationLayer
    {
        public override double Differentiate(double x, double a)
        {
            return x < 0 ? Activation.ELU(x, a) + a : 1;
            //throw new System.NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum, double a)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.ELU(LayerSum[i], a);
            }

            return LayerSum;
        }
    }

    class PRelu : ActivationLayer
    {
        public override double Differentiate(double x, double a = 0)
        {
            return x < 0 ? a : 1;
            //throw new System.NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum, double a = 0)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.PRelu(LayerSum[i], a);
            }

            return LayerSum;
            //throw new System.NotImplementedException();
        }
    }

    class ArcTan : ActivationLayer
    {
        public override double Differentiate(double x, double a = 0)
        {
            return 1 / (Math.Pow(x, 2) + 1);
            //throw new System.NotImplementedException();
        }

        public override double[] Calculate(ref double[] LayerSum, double a = 0)
        {
            for (int i = 0; i < LayerSum.Length; i++)
            {
                LayerSum[i] = Activation.ArcTan(LayerSum[i]);
            }

            return LayerSum;
            //throw new System.NotImplementedException();
        }
    }



    // propogation layer below
    class Conv2D : ConvLayer // todo
    {
        public Conv2D(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, int dilation=1,int group=1, bool bias=true,string padding_mode="zeros" ):
            base(in_channels, out_channels)
        {
            // kernel_num = out_channels, bias added， init needed
            //var kernels_weight = new double[out_channels, kernel_size + 1, kernel_size];
            for(var i = 0; i < out_channels; i++)
            {
                Weights[i] = Helper.MakeMatrix(kernel_size + 1, kernel_size);
                Grads[i] = Helper.MakeMatrix(kernel_size + 1, kernel_size);
                PrevWeightsDelta[i] = Helper.MakeMatrix(kernel_size + 1, kernel_size);
                Signals[i] = new double[kernel_size];

            }


        }
    }
    class Linear : LinearLayer
    {
        public Linear(int numInput, int numOutput) : base(numInput, numOutput)
        {
//            DIn = numInput;
//            DOut = numOutput;
            Weights = Helper.MakeMatrix(DIn + 1, DOut); // include biases
//            Helper.InitializeWeights(ref Weights); // initializeWeights globally
            Grads = Helper.MakeMatrix(DIn + 1, DOut);
            PrevWeightsDelta = Helper.MakeMatrix(DIn + 1, DOut);
            Signals = new double[DOut]; // gradients output signals
        }
    }

    class Dropout: Layer
    {
        public Dropout(double p)
        {

        }
    }
}