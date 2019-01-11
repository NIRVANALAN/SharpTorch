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
            Layers = model.Layers;
            Lr = lr;
        }

        public abstract void Step();

        public void Zero_grad()
        {
            for (int layerIndex = 0; layerIndex < Layers.Length - 1; layerIndex++)
            {
                if (Layers[layerIndex].GetType() == typeof(PropogationLayer))
                {
                    var cLayer = (PropogationLayer) Layers[layerIndex];
                    for (int i = 0; i < cLayer.DIn; i++)
                    {
                        for (int j = 0; j < cLayer.DOut; j++)
                        {
                            cLayer.Grads[i, j] = 0.0; //zero_grad
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
            for (int layerIndex = 0; layerIndex < Layers.Length; layerIndex++)
            {
                if (Layers[layerIndex].GetType().BaseType == typeof(PropogationLayer))
                {
                    var cLayer = (PropogationLayer) Layers[layerIndex];
                    for (int i = 0; i < cLayer.DIn + 1; i++) //bug fix. bias weights not updated here
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

//            throw new NotImplementedException();
        }

        public SGD(Model model, double lr, double momentum = 0.0) : base(model, lr)
        {
            Momentum = momentum;
        }
    }
}