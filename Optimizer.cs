using System;
using System.Security.AccessControl;

namespace cs_nn_fm
{
    public abstract class Optimizer // for optimization
    {
        public Model Model;
        public Layer[] Layers;
        public double Lr; //learning rate
        public double Momentum;

        protected Optimizer(Model model, double lr)
        {
            Model = model;
            Lr = lr;
        }

        public abstract void Step();
        public void Zero_grad()
        {
            for (int layerIndex = 0; layerIndex < Layers.Length-1; layerIndex++)
            {
                if (Layers[layerIndex].GetType()==typeof(PropogationLayer))
                {
                    var cLayer = (PropogationLayer) Layers[layerIndex];
                    for (int i = 0; i < cLayer.DIn; i++)
                    {
                        for (int j = 0; j < cLayer.DOut; j++)
                        {
                            cLayer.Grads[i, j] = 0.0;//zero_grad
                        }
                    }

                }
            }
        }
    }

    public class SGD : Optimizer
    {
        public override void Step()
        {
            //===========update begin here=========
            for (int layerIndex = 0; layerIndex < Layers.Length - 1; layerIndex++)
            {
                if (Layers[layerIndex].GetType() == typeof(PropogationLayer))
                {
                    var cLayer = (PropogationLayer)Layers[layerIndex];
                    for (int i = 0; i < cLayer.DIn; i++)
                    {
                        for (int j = 0; j < cLayer.DOut; j++)
                        {
                            var delta = Lr * cLayer.Grads[i, j];
                            cLayer.Weights[i, j] -= delta;
                            cLayer.Weights[i, j] -= cLayer.PrevWeightsDelta[i, j] * Momentum;
                            cLayer.PrevWeightsDelta[i, j] = delta;
                        }
                    }
                } //propogationLayer
            } // weights update step
            throw new NotImplementedException();
        }
//        public static double[] SGD(Model model, Dataset data_set, int max_epochs, double lr, double momentum)
//        {
//            Console.WriteLine("Train using stochastic back propogation start:");
//            var sequence = new int[data_set.GetLen()];
//            var epoch = 0;
        //            var xValues = new double[_numInput]; // input vals
        //            var tValues = new double[_numOutput]; // target vals
//            for (int i = 0; i < sequence.Length; i++)
//            {
//                sequence[i] = i;
//            }

//            var errInterval = max_epochs / 50; // interval to check validation data
//            var allTrainData = data_set.dataSet;
        //            var trainSet = TODO
        //            var testSet = TODO
//            while (epoch < max_epochs)
//            {
//                epoch++;
//                if (epoch % errInterval == 0 && epoch < max_epochs)
//                {
//                    check err
//                    var trainErr = Evaluate.MSE(allTrainData, data_set.InputNum, data_set.OutputNum);
//                    Console.WriteLine("epoch= " + epoch + " acc = " + (1 - trainErr).ToString("F4"));
//                }
//            }
        public SGD(Model model, double lr, double momentum = 0.0) : base(model, lr)
        {
            Momentum = momentum;
        }
    }
}