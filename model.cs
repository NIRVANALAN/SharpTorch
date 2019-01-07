using System;

namespace cs_nn_fm
{
    public class Model
    {
        private layer[] _layers;
        private int _layerNum;
        private double[][] _layerSum; // should start from 1
        private double[][] nodes;

        public Model(layer[] layers)
        {
            _layers = layers;
            _layerNum = 0;
            foreach (var t in _layers) // init the nodes
            {
                if (t.GetType() != typeof(PropogationLayer)) continue;
                if (nodes[_layerNum] == null)
                {
                    nodes[_layerNum] = new double[t.GetDIn() + 1]; // add biases
                    nodes[_layerNum][t.GetDIn()] = 1; // for bias
                }

                if (nodes[_layerNum].Length == t.GetDIn()) // check the LastLayer.DOut==ThisLayer.Dint
                {
                    _layerNum++;
                    nodes[_layerNum] = new double[t.GetDOut() + 1];
                    nodes[_layerNum][t.GetDOut()] = 1; // for bias
                    _layerSum[_layerNum] = new double[t.GetDOut() + 1];
                }
                else
                {
                    throw new Exception("layer not compatible in layerNum: " + _layerNum.ToString());
                }
            }
        }

        public double[] Forward(double[] x_values)
        {
            //check input dimension
            if (x_values.Length + 1 != nodes[0].Length)
            {
                throw new Exception("Input x_value not compatible");
            }

            // copy x_input to nodes[0]
            for (int i = 0; i < x_values.Length; i++)
            {
                nodes[0][i] = x_values[i]; // nn input
            }

//            nodes[0][x_values.Length] = 1; // have done above
            // forward
            var currentLayer = 0;
            foreach (var t in _layers)
            {
                if (t.GetType() == typeof(PropogationLayer))
                {
                    var cLayer = (PropogationLayer) t; // this is a Propogation Layer
                    for (int j = 0; j < nodes[currentLayer+1].Length; j++)//input-next-layer
                    {
                        for (int i = 0; i < nodes[currentLayer].Length; i++)//input-layer
                        {
                            _layerSum[currentLayer + 1][j] += nodes[currentLayer][i] * cLayer.Weights[i, j];
                        }
                    }
                    currentLayer++; // point to next input-layer
                    continue;
                }

                // activation
                if (t.GetType() != typeof(Activation)) // check at last if
                    throw new Exception("Don't accept this type of Class in Forward");
                {
                    var cLayer = (ActivationLayer) t;
                    // polymorphism calculate activation
                    cLayer.Calculate(ref _layerSum[currentLayer]);
                }
            } // forward finish

            return nodes[currentLayer];//result
        }
    }
}