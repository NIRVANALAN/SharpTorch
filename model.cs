using System;

namespace cs_nn_fm
{
    public class Model
    {
        public Layer[] Layers { get; }
        public int LayerNum;
        public double[][] LayerSum; // scratch array, should start from 1

        public double[][] Nodes { get; }

//        public double[] XValues;
        public double[] PredictedValues { get; set; }
        public int InputNum { get; set; }
        public int OutputNum { get; set; }

        public Model(Layer[] layers)
        {
            Layers = layers;
            LayerNum = 0;
//            var totalPropogationLayer = 0;
//            foreach (var t in layers)
//            {
//                if (t.GetType().BaseType == typeof(PropogationLayer))
//                    totalPropogationLayer++;
//            }

            Nodes = new double[layers.Length][];
            LayerSum = new double[layers.Length][];
            var flag = 1;
            foreach (var t in Layers) // check if the model is legal(propogation-Activation)
            {
                if (flag == 1 && t.GetType().BaseType == typeof(PropogationLayer))
                {
                    flag = 1 - flag;
                    continue;
                }
                if (flag==0&& t.GetType().BaseType == typeof(ActivationLayer))
                {
                    flag = 1 - flag;
                }

            }
            foreach (var t in Layers) // init the nodes
            {
                if (t.GetType().BaseType != typeof(PropogationLayer)) continue;
                var cLayer = (PropogationLayer) t;
                if (Nodes[LayerNum] == null)
                {
                    Nodes[LayerNum] = new double[cLayer.DIn + 1]; // add biases
                    Nodes[LayerNum][cLayer.DIn] = 1; // for bias
                }

                if (Nodes[LayerNum].Length == cLayer.DIn + 1) // check the LastLayer.DOut==ThisLayer.Dint
                {
                    LayerNum++;
//                    cLayer.Weights = Helper.MakeMatrix(cLayer.DIn + 1, cLayer.DOut);
//                    Helper.InitializeWeights(ref cLayer.Weights); // init weights
                    Nodes[LayerNum] = new double[cLayer.DOut + 1];
                    Nodes[LayerNum][cLayer.DOut] = 1; // for bias
                    LayerSum[LayerNum] = new double[cLayer.DOut + 1];
                    LayerSum[LayerNum][cLayer.DOut] = 1; //for bias
                }
                else
                {
                    throw new Exception("layer not compatible in layerNum: " + LayerNum);
                }
            }
        }

        public Model Forward(double[] XValues)
        {
            //check input dimension
            if (XValues.Length + 1 != Nodes[0].Length)
            {
                throw new Exception("Input x_value not compatible");
            }

            // copy x_input to nodes[0]
            for (int i = 0; i < XValues.Length; i++)
            {
                Nodes[0][i] = XValues[i]; // nn input
            }

//            nodes[0][x_values.Length] = 1; // have done above
            // forward
            var currentLayer = 0;
//            var activationNextFlag = false;
            foreach (var t in Layers)
            {
                if (t.GetType().BaseType == typeof(PropogationLayer))
                {
                    var cLayer = (PropogationLayer) t; // this is a Propogation Layer
                    for (int j = 0;
                        j < Nodes[currentLayer + 1].Length - 1;
                        j++) //input-next-layer -1 to remove bias-node
                    {
                        for (int i = 0; i < Nodes[currentLayer].Length; i++) //bias calculation included
                        {
                            LayerSum[currentLayer + 1][j] += Nodes[currentLayer][i] * cLayer.Weights[i, j];
                        }
                    }

                    currentLayer++; // point to next input-layer
//                    activationNextFlag = !activationNextFlag; // should be true in next circulation, proving that ActivationFunc is needed next
                    Nodes[currentLayer] = LayerSum[currentLayer]; // avoid no cost function after (in regression)
                    Nodes[currentLayer][cLayer.DOut] = 1; //should be 1,for bias
                    continue;
                }

                // activation
                if ((t.GetType().BaseType != typeof(ActivationLayer))) // check at last if
                    throw new Exception("Don't accept this type of Class in Forward: " + t.GetType());
                {
                    var cLayer = (ActivationLayer) t;
                    // polymorphism calculate activation
                    Nodes[currentLayer] = cLayer.Calculate(ref LayerSum[currentLayer]);
                    Nodes[currentLayer][Nodes[currentLayer].Length-1] = 1; //should be 1,for bias
                }
            } // forward finish

            PredictedValues = Nodes[currentLayer]; // store for later use
            return this; //result
        }
    }
}