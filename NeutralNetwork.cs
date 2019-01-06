using System;

namespace cs_nn_fm {
    public class NeutralNetwork {
        private int _numInput;
        private int _numHidden;
        private int _numOutput;
        private double[] _inputs; // input nodes
        private double[] _hiddens;
        private double[] _outputs;
        private double[][] ih_weights; // Input-hidden weights
        private double[] _hBiases; // Input-hidden weights
        private double[][] _hoWeights; // hidden-output weights
        private double[] _oBiases;
        private Random _rnd;

        public NeutralNetwork(int numInput, int numHidden, int numOutput, int rndSeed)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;
            // init predictor
            this._inputs = new double[numInput];
            this._hiddens = new double[numHidden];
            this._outputs = new double[numOutput];
            // init weights
            this.ih_weights = MakeMatrix(numInput, numHidden, 0.0);
            this._hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this._hBiases = new double[numHidden];
            this._oBiases = new double[numOutput];
            // init rnd
            this._rnd = new Random(rndSeed);
            // all weights and biases
        }

        // helper methods
        public void SetWeight (double[] weights) { }
        public double[] GetWeights () {
            var numWeights = (_numInput * _numHidden) + _numHidden + (_numHidden * _numOutput) + _numOutput;
            var res = new double[numWeights];
            //i-h weighs + h biases + h-o weights + o hiases (order)
            var w = 0;
            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    res[w++] = ih_weights[i][j];

            for (int j = 0; j < _numHidden; ++j)
                res[w++] = _hBiases[j];

            for (int j = 0; j < _numHidden; ++j)
                for (int k = 0; k < _numOutput; ++k)
                    res[w++] = _hoWeights[j][k];

            for (int k = 0; k < _numOutput; ++k)
                res[w++] = _oBiases[k];
            return _outputs; // tmp
        }
        public double[] ComputeOutputs (double[] x_values) {
            // preliminary values
            var hSums = new double[_numHidden]; // scratch array
            var oSums = new double[_numOutput];

            for (int i = 0; i < _numInput; i++) {
                this._inputs[i] = x_values[i]; // copy independent vars into input nodes
            }
            for (int j = 0; j < _numHidden; j++)
                for (int i = 0; i < _numInput; i++) {
                    hSums[j] += this._inputs[i] * this.ih_weights[i][j]; // full-connect network
                }
            // add the bias
            for (int j = 0; j < _numHidden; j++)
                hSums[j] += this._hBiases[j];
            //activation
            for (int j = 0; j < _numHidden; j++) {
                this._hiddens[j] = Activation.HyperTan (hSums[j]);
            }
            for (int k = 0; k < _numOutput; k++)
                for (int j = 0; j < _numHidden; j++) {
                    oSums[k] += _hiddens[j] * _hoWeights[j][k];
                }
            for (int k = 0; k < _numOutput; k++) {
                oSums[k] += _oBiases[k];
            }
            // no softmax activation in regression applied. Just copy
            Array.Copy (oSums, this._outputs, _outputs.Length);
            double[] resRes = new double[_numOutput];
            Array.Copy (this._outputs, resRes, resRes.Length); // copy res_res to output[]
            return resRes;
        }

        private static double[][] MakeMatrix (int rows, int cols, double init_val) //helper method
        {
            var res = new double[rows][];
            for (int r = 0; r < res.Length; r++) {
                res[r] = new double[cols];
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res[i][j] = init_val;
                }
            }
            return res;
        }
        public double[] Train (double[][] train_data, int max_epochs, double lr, double momentum) {
            // back-prop specific arrays
            var hoGrads = MakeMatrix (_numHidden, _numOutput, 0.0); // hidden-output grad
            var obGrads = new double[_numOutput]; // output bias grad

            var ihGrad = MakeMatrix (_numInput, _numHidden, 0.0); // input-hidden grad
            var hbGrad = new double[_numHidden]; // hidden-bias grad

            // signal
            var oSignals = new double[_numOutput]; // signals == gradients w/o associated input terms
            var hSignals = new double[_numHidden];

            //backprop-momentum specific array
            var ihPrevWeightsDelta = MakeMatrix (_numInput, _numHidden, 0.0);
            var hPrevBiasesDelta = new double[_numHidden];
            var hoPrevWeightsDelta = MakeMatrix (_numHidden, _numOutput, 0.0);
            var oPrevBiasisDelta = new double[_numOutput];
            // train NN using lr and momentum
            var epoch = 0;
            var xValues = new double[_numInput]; // input vals
            var tValues = new double[_numOutput]; // target vals

            var sequence = new int[train_data.Length];
            for (int i = 0; i < sequence.Length; i++) {
                sequence[i] = i;
            }

            var errInterval = max_epochs / 10; // interval to check validation data
            while (epoch < max_epochs) { // every epoch
                epoch++;
                if (epoch % errInterval == 0 && epoch < max_epochs) {
                    // check err
                    var trainErr = 0.0;
                    System.Console.WriteLine ("epoch= " + epoch + "training error = " + trainErr.ToString ("F4"));
                }
                Shuffle (sequence); // shuffle the order

                for (int ii = 0; ii < train_data.Length; ii++) {
                    int idx = sequence[ii];
                    Array.Copy (train_data[idx], xValues, _numInput);
                    Array.Copy (train_data[idx], _numInput, tValues, 0, _numOutput);
                    ComputeOutputs (xValues); // res_outupt has been copied to output[]

                    // i=inputs j=hidden(s) k=outputs
                    // 1. compute output nodes signals
                    for (int k = 0; k < _numOutput; k++) {
                        var derivatives = 1.0; // dummy
                        oSignals[k] = (tValues[k] - _outputs[k]) * derivatives;
                    }
                    // 2. compute h-to-o weights gradients using output signals
                    for (int j = 0; j < _numHidden; j++) {
                        for (int k = 0; k < _numOutput; k++) {
                            hoGrads[j][k] = oSignals[k] * _hiddens[j];
                        }
                    }
                    // 2'. compute the output biases grads using output signals
                    for (int k = 0; k < _numOutput; k++) {
                        obGrads[k] = oSignals[k] * 1.0; // dummy
                    }
                    // 3.comput hidden nodes signals
                    for (int j = 0; j < _numHidden; j++) {
                        var sum = 0.0;
                        for (int k = 0; k < _numOutput; k++) {
                            sum += oSignals[k] * _hoWeights[j][k];
                        }
                        var derivatives = (1 + _hiddens[j]) * (1 - _hiddens[j]); // for tanh
                        hSignals[j] = sum * derivatives;
                    }
                    // 4. compute input-hidden weights grads
                    for (int i = 0; i < _numInput; i++) {
                        for (int j = 0; j < _numHidden; j++) {
                            ihGrad[i][j] = ih_weights[i][j] * _inputs[i];
                        }
                    }
                    // 4.b compute input-hidden biases grads
                    for (int j = 0; j < _numHidden; j++) {
                        hbGrad[j] = hSignals[j] * 1.0; // dummy 1.0 input
                    }
                    // ========begin update here==========
                    //1. update input-hidden weights
                    for (int i = 0; i < _numInput; i++) {
                        for (int j = 0; j < _numHidden; j++) {
                            var delta = lr * ihGrad[i][j];
                            ih_weights[i][j] += delta;
                            // momentum involved
                            ih_weights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta;
                        }
                    }
                    //2. update hidden biases
                    for (int j = 0; j < _numHidden; j++) {
                        var delta = hbGrad[j] * lr;
                        _hBiases[j] += delta;
                        _hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }
                    //3.update hidden-output weights
                    for (int j = 0; j < _numHidden; j++) {
                        for (int k = 0; k < _numOutput; k++) {
                            var delta = hoGrads[j][k] * lr;
                            _hoWeights[j][k] += delta;
                            _hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }
                    //4.update output biases
                    for (int k = 0; k < _numOutput; k++) {
                        var delta = obGrads[k] * lr;
                        obGrads[k] += delta;
                        obGrads[k] += oPrevBiasisDelta[k] * momentum;
                        oPrevBiasisDelta[k] = delta;
                    }
                } //each training item
            } // end while(each epoch)
            var bestWeights = this.GetWeights ();
            return bestWeights;
        }

        private void Shuffle (int[] sequence) {
            for (int i = 0; i < sequence.Length; i++) {
                var r = _rnd.Next (i, sequence.Length);
                var tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }
        private double Error (double[][] data) {
            // MSE : average squared error per training item
            var sumSquaredErr = 0.0;
            var xValues = new double[_numInput]; // intput x(first input_num vals in train_data)
            var tValues = new double[_numOutput]; // output y(last num_output vals in train_data)
            for (int i = 0; i < data.Length; i++) {
                Array.Copy (data[i], xValues, _numInput);
                Array.Copy (data[i], _numInput, tValues, 0, _numOutput);
                var yValues = this.ComputeOutputs (xValues);
                for (int j = 0; j < _numOutput; j++) {
                    var err = tValues[j] - yValues[j]; // calc
                    sumSquaredErr += Math.Pow (err, 2);
                }
            }
            return sumSquaredErr / data.Length;
        } //Error
    } // class NeuralNetwork
} //ns