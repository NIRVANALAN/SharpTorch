using System;
using System.Linq;

namespace cs_nn_fm
{
    public abstract class Loss
    {
        public double LossSum;
        public Model Model;
        public Layer[] Layers;
        protected double item;

        public double[] PredictedValues;
//        public double Lr; //learning rate
//        public double Momentum;

        public double[] TargetValues; // predicted_val and real_value
//        protected double[,]

        protected Loss(Model model, double[] targetValues)
        {
//            this.Lr = lr;
            Model = model;
            TargetValues = targetValues;
//            Momentum = momentum;
            PredictedValues = model.Nodes[model.Nodes.Length - 1];
            Layers = model.Layers;
            foreach (var t in Layers) // init weights and signals
            {
                if (t.GetType().BaseType != typeof(PropogationLayer)) continue;
                var cLayer = (PropogationLayer) t;
//                cLayer.Weights = new double[cLayer.GetDIn()+1,cLayer.GetDOut()];
                cLayer.Signals = new double[cLayer.DOut];
                cLayer.PrevWeightsDelta = new double[cLayer.DIn + 1, cLayer.DOut]; // bias included
            }

            CostFunctionDerivative(); // calculate costFunction signal here : polymorphism
        } //ctor

        public abstract double Calculate();

        public abstract void CostFunctionDerivative();

        public double Item()
        {
            return item;
        }

        public void Backward()
        {
            var currentLayerIndex = Layers.Length - 1; // for output layer
            // output activation function
//            if (Layers[currentLayerIndex].GetType() == typeof(PropogationLayer))
//            {
//                // no cost layer at output
//                var derivative = 1.0; //dummy
//                var cLayer = (PropogationLayer) Layers[currentLayerIndex];
//                for (int i = 0; i < cLayer.DOut; i++)
//                {
//                    cLayer.Signals[i] = (PredictedValues[i] - TargetValues[i]) * derivative;
//                }
//            } 

            //============compute signal and Grads=============
            for (var i = currentLayerIndex; i >= 0; i--) // from output-layer 
            {
                if (Layers[i].GetType().BaseType == typeof(PropogationLayer)) // backward weights
                {
                    var cLayer = (PropogationLayer) Layers[i];
                    for (var j = 0; j < cLayer.DIn + 1; j++) // add bias node here
                    {
                        for (var k = 0; k < cLayer.DOut; k++)
                        {
                            cLayer.Grads[j, k] = cLayer.Signals[k] * Model.Nodes[i / 2][j]; //  i/2:layer->hidden nodes
                        }
                    } // calculate weightsGrad

//                    Layers[i] = cLayer;
                    continue;
                }

                if (Layers[i].GetType().BaseType != typeof(ActivationLayer)) // backward signal
                    throw new Exception("Do not accept Propogation layer in activation layer position: " +
                                        Layers[i].GetType());
                {
                    var cLayer = (ActivationLayer) Layers[i];
                    var hiddenLayer = (PropogationLayer) Layers[i - 1];
                    var outputLayer = (PropogationLayer) Layers[i + 1];
                    for (int j = 0; j < outputLayer.DIn; j++)
                    {
                        var sum = 0.0;
                        for (int k = 0; k < outputLayer.DOut; k++)
                        {
                            sum += outputLayer.Signals[k] * outputLayer.Weights[j, k];
                        }

                        // signal
                        var derivatives = cLayer.Differentiate(Model.Nodes[i][j]);
                        hiddenLayer.Signals[j] = sum * derivatives;
                    }

//                    Layers[i - 1] = hiddenLayer;
//                    Layers[i + 1] = outputLayer;
                }
            } // backward finish
//            //===========update should be via optimizer=========
//            for (int layer_index = 0; layer_index < Layers.Length - 1; layer_index++)
//            {
//                if (Layers[layer_index].GetType() == typeof(PropogationLayer))
//                {
//                    var cLayer = (PropogationLayer) Layers[layer_index];
//                    for (int i = 0; i < cLayer.DIn; i++)
//                    {
//                        for (int j = 0; j < cLayer.DOut; j++)
//                        {
//                            var delta = Lr * cLayer.Grads[i, j];
//                            cLayer.Weights[i, j] -= delta;
//                            cLayer.Weights[i, j] -= cLayer.PrevWeightsDelta[i, j] * Momentum;
//                            cLayer.PrevWeightsDelta[i, j] = delta;
//                        }
//                    }
//                } //propogationLayer
//            } // weights update loop

//            return 0.0; // tmp
        }
    }

    public class RegressionLoss : Loss
    {
        public RegressionLoss(Model model, double[] targetValues) : base(model, targetValues)
        {
        }

        public override double Calculate()
        {
            if (PredictedValues.Length != 1)
                throw new Exception("Not compatible dimension for RegressionLoss: " + PredictedValues.Length);
            {
                return PredictedValues[0] - TargetValues[0];
            }
            throw new NotImplementedException();
        }

        public override void CostFunctionDerivative()
        {
            var currentLayerIndex = Layers.Length - 1; // for output layer
            // output cost_function derivatives
            if (Layers[currentLayerIndex].GetType().BaseType != typeof(PropogationLayer))
                throw new Exception("Do not accept this type of Class here:" + Layers[currentLayerIndex].GetType());
            // Regression, no derivatives calculation....
            var derivative = 1.0; //dummy
            var cLayer = (PropogationLayer) Layers[currentLayerIndex];
//            for (int i = 0; i < cLayer.DOut; i++)
//            {
            item = cLayer.Signals[0] = (PredictedValues[0] - TargetValues[0]) * derivative;
//            }
        }
    }

    public class MSELoss : Loss
    {
        public override double Calculate()
        {
//            var sum = 0.0;
            if (PredictedValues.Length != TargetValues.Length)
                throw new Exception("y_pred and targetValues not compatible in MSELoss");
            LossSum += PredictedValues.Select((t, i) => Math.Pow((t - TargetValues[i]), 2)).Sum();
            return LossSum;
        }

        public override void CostFunctionDerivative()
        {
            throw new NotImplementedException();
        }

        public MSELoss(Model model, double[] targetValues) : base(model, targetValues)
        {
        }
    }
}