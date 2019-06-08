using System;

namespace cs_nn_fm
{
    public class Model
    {
        public Layer[] Layers { get; }
        public int LayerNum;
        public double[][] LayerSum; // scratch array, should start from 1

        public double[][] LinearNodes { get; }
        public double [][,] ConvNodes { get; }

        //        public double[] XValues;
        public double[] PredictedLinearValues { get; set; }
        public double[][,] PredictedConvValues { get; set; }

        public int InputDimension { get; set; }
        public int OutputDimension { get; set; }

        public int LinearNodesWeightsNum { get; set; }
        public int ConvNdoesWeightsNum { get; set; }

        public int[] LinearDIns;
        public int[] LInearDOuts;

        public int[][,] ConvDIns;
        public int[][,] ConvDOuts;

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
            LinearNodes = new double[layers.Length][];
            LayerSum = new double[layers.Length][];
            var flag = 1;
            InputDimension = ((LinearLayer) layers[0]).DIn;
            var propLayerIndex = 0;
            LinearDIns = new int[layers.Length];
            LInearDOuts = new int[layers.Length];
            foreach (var t in Layers) // check if the model is legal(propogation-Activation)
            {
                if (flag == 1 && t.GetType().BaseType == typeof(LinearLayer))
                {
                    var cLayer = (LinearLayer) t;
                    flag = 1 - flag;
                    LInearDOuts[propLayerIndex] = OutputDimension = cLayer.DOut;
                    LinearDIns[propLayerIndex] = cLayer.DIn;
                    propLayerIndex++;
                    continue;
                }

                if (flag == 0 && t.GetType().BaseType == typeof(ActivationLayer))
                {
                    flag = 1 - flag;
                }
            }

            LinearNodesWeightsNum = 0;
            for (int i = 0; i < propLayerIndex; i++)
            {
                LinearNodesWeightsNum += (LinearDIns[i] + 1) * LInearDOuts[i]; //bias
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
                if (LinearNodes[LayerNum] == null)
                {
                    LinearNodes[LayerNum] = new double[cLayer.DIn + 1]; // add biases
                    LinearNodes[LayerNum][cLayer.DIn] = 1; // for bias
                }

                if (LinearNodes[LayerNum].Length == cLayer.DIn + 1) // check the LastLayer.DOut==ThisLayer.Dint
                {
                    LayerNum++;
//                    cLayer.Weights = Helper.MakeMatrix(cLayer.DIn + 1, cLayer.DOut);
//                    Helper.InitializeWeights(ref cLayer.Weights); // init weights
                    LinearNodes[LayerNum] = new double[cLayer.DOut + 1];
                    LinearNodes[LayerNum][cLayer.DOut] = 1; // for bias
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
            InitialWeights = new double[LinearNodesWeightsNum];
            for (int i = 0; i < InitialWeights.Length; ++i)
                InitialWeights[i] = (hi - lo) * rnd.NextDouble() + lo; // [-0.001 to +0.001] by default
            SetWeights(InitialWeights); //setWeights globally
        }

        public void SetWeights(double[] initialWeights)
        {
            if (LinearNodesWeightsNum != initialWeights.Length)
            {
                throw new Exception("Bad weights array in SetWeights");
            }

            var w = 0;
            for (int i = 0; i < LinearDIns.Length; i++)
            {
                for (int j = 0; j < LinearDIns[i] + 1; j++) //add bias
                {
                    for (int k = 0; k < LInearDOuts[i]; k++)
                    {
                        ((LinearLayer) Layers[i * 2]).Weights[j, k] = initialWeights[w++];
                    }
                }
            }
        }

        public double[] GetWeights()
        {
            var finalWeights = new double[LinearNodesWeightsNum];
            var w = 0;
            for (int i = 0; i < LinearDIns.Length; i++)
            {
                for (int j = 0; j < LinearDIns[i]+1; j++)
                {
                    for (int k = 0; k < LInearDOuts[i]; k++)
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

            if (XValues.Length + 1 != LinearNodes[0].Length) //check input dimension
            {
                throw new Exception("Input x_value dimension not compatible");
            }

            // copy x_input to nodes[0]
            for (int i = 0; i < XValues.Length; i++)
            {
                LinearNodes[0][i] = XValues[i]; // nn input
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
                        j < LinearNodes[currentLayer + 1].Length - 1;
                        j++) //input-next-layer -1 to remove bias-node
                    {
                        for (int i = 0; i < LinearNodes[currentLayer].Length; i++) //bias calculation included
                        {
                            LayerSum[currentLayer + 1][j] += LinearNodes[currentLayer][i] * cLayer.Weights[i, j];
                        }
                    }

                    currentLayer++; // point to next input-layer
//                    activationNextFlag = !activationNextFlag; // should be true in next circulation, proving that ActivationFunc is needed next
                    LinearNodes[currentLayer] = LayerSum[currentLayer]; // avoid no cost function after (in regression)
                    LinearNodes[currentLayer][cLayer.DOut] = 1; //should be 1,for bias
                    continue;
                }

                // activation
                if ((t.GetType().BaseType != typeof(ActivationLayer))) // check at last if
                    throw new Exception("Don't accept this type of Class in Forward: " + t.GetType());
                {
                    var cLayer = (ActivationLayer) t;
                    // polymorphism calculate activation
                    LinearNodes[currentLayer] = cLayer.Calculate(ref LayerSum[currentLayer]);
                    LinearNodes[currentLayer][LinearNodes[currentLayer].Length - 1] = 1; //should be 1,for bias
                }
            } // forward finish

            PredictedLinearValues = LinearNodes[currentLayer]; // store for later use
            return this; //result
        }
    }
}