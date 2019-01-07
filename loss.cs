using System;
using System.Linq;

namespace cs_nn_fm
{
    public abstract class Loss
    {
        public double LossSum = 0.0;
        public Model Model;
        public Layer[] Layers;
        public double[] PredictedValues;
        public double[] TargetValues; // predicted_val and real_value
//        protected double[,]

        protected Loss(Model model, double[] tTargetValues)
        {
            Model = model;
            TargetValues = tTargetValues;
            PredictedValues = model.Nodes[model.Nodes.Length - 1];
            Layers = model.Layers;
            foreach (var t in Layers) // init weights and signals
            {
                if (t.GetType() != typeof(PropogationLayer)) continue;
                var cLayer = (PropogationLayer) t;
//                cLayer.Weights = new double[cLayer.GetDIn()+1,cLayer.GetDOut()];
                cLayer.Signals = new double[cLayer.DOut];
                cLayer.PrevWeightsDelta = new double[cLayer.DIn + 1, cLayer.DOut];// bias included
            }
        }

        public abstract double Calculate();

        public double Backward()
        {
            // implement future
            return 0.0;
        }
    }

    public class MSELoss : Loss
    {
        public override double Calculate()
        {
//            var sum = 0.0;
            if (PredictedValues.Length != TargetValues.Length) throw new Exception("y_pred and tTargetValues not compatible in MSELoss");
            LossSum += PredictedValues.Select((t, i) => Math.Pow((t - TargetValues[i]), 2)).Sum();
            return LossSum;
        }

        public MSELoss(Model model, double[] tTargetValues) : base(model, tTargetValues)
        {
        }
    }
}