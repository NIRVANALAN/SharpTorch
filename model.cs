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
        public int WeightsNum { get; set; }
        public int[] DIns;
        public int[] DOuts;
        public double[] InitialWeights;
        private double lo;
        private double hi;


        public Model(Layer[] layers, double lo=-0.001, double hi=0.001, double[] weights = null)
        {
            Layers = layers;
            this.lo = lo;
            this.hi = hi;
            InitialWeights = weights;
            LayerNum = 0;
            Nodes = new double[layers.Length][];
            LayerSum = new double[layers.Length][];
            var flag = 1;
            InputNum = ((LinearLayer) layers[0]).DIn;
            var propLayerIndex = 0;
            DIns = new int[layers.Length];
            DOuts = new int[layers.Length];
            foreach (var t in Layers) // check if the model is legal(propogation-Activation)
            {
                if (flag == 1 && t.GetType().BaseType == typeof(LinearLayer))
                {
                    var cLayer = (LinearLayer) t;
                    flag = 1 - flag;
                    DOuts[propLayerIndex] = OutputNum = cLayer.DOut;
                    DIns[propLayerIndex] = cLayer.DIn;
                    propLayerIndex++;
                    continue;
                }

                if (flag == 0 && t.GetType().BaseType == typeof(ActivationLayer))
                {
                    flag = 1 - flag;
                }
            }

            WeightsNum = 0;
            for (int i = 0; i < propLayerIndex; i++)
            {
                WeightsNum += (DIns[i] + 1) * DOuts[i]; //bias
            }

            //initialize weights
            if (InitialWeights == null)
            {
                InitializeWeights(lo,hi);
            }
            else
            {
                SetWeights(initialWeights: InitialWeights);
            }

            foreach (var t in Layers) // init the nodes
            {
                if (t.GetType().BaseType != typeof(LinearLayer)) continue;
                var cLayer = (LinearLayer) t;
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
//                    LayerSum[LayerNum][cLayer.DOut] = 1; //for bias
                }
                else
                {
                    throw new Exception("layer not compatible in layerNum: " + LayerNum);
                }
            }
        }

        private void InitializeWeights(double lo = -0.001, double hi = 0.001, int rnd_seed = 1)
        {
            var rnd = new Random(rnd_seed);
            InitialWeights = new double[WeightsNum];
            for (int i = 0; i < InitialWeights.Length; ++i)
                InitialWeights[i] = (hi - lo) * rnd.NextDouble() + lo; // [-0.001 to +0.001] by default
            SetWeights(InitialWeights); //setWeights globally
        }

        public void SetWeights(double[] initialWeights)
        {
            if (WeightsNum != initialWeights.Length)
            {
                throw new Exception("Bad weights array in SetWeights");
            }

            var w = 0;
            for (int i = 0; i < DIns.Length; i++)
            {
                for (int j = 0; j < DIns[i] + 1; j++) //add bias
                {
                    for (int k = 0; k < DOuts[i]; k++)
                    {
                        ((LinearLayer) Layers[i * 2]).Weights[j, k] = initialWeights[w++];
                    }
                }
            }
        }

        public double[] GetWeights()
        {
            var finalWeights = new double[WeightsNum];
            var w = 0;
            for (int i = 0; i < DIns.Length; i++)
            {
                for (int j = 0; j < DIns[i]+1; j++)
                {
                    for (int k = 0; k < DOuts[i]; k++)
                    {
                        finalWeights[w++] = ((LinearLayer) Layers[i * 2]).Weights[j, k];
                    }
                }
            }

            return finalWeights;
        }

        public Model Forward(double[] XValues)
        {
            foreach (var item in LayerSum) // zero_layerSum
            {
                if (item != null)
                {
                    Array.Clear(item, 0, item.Length);

                }
            }

            if (XValues.Length + 1 != Nodes[0].Length) //check input dimension
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
                if (t.GetType().BaseType == typeof(LinearLayer))
                {
                    var cLayer = (LinearLayer) t; // this is a Propogation Layer
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
                    Nodes[currentLayer][Nodes[currentLayer].Length - 1] = 1; //should be 1,for bias
                }
            } // forward finish

            PredictedValues = Nodes[currentLayer]; // store for later use
            return this; //result
        }
    }
}