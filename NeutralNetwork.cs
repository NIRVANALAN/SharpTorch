using System;

namespace ann {
    public class NeutralNetwork {
        private int num_input;
        private int num_hidden;
        private int num_output;
        private int rnd_seed;
        private double[] inputs; // input nodes
        private double[] hiddens;
        private double[] outputs;
        private double[][] ih_weights; // Input-hidden weights
        private double[] h_biases; // Input-hidden weights
        private double[][] ho_weights; // hidden-output weights
        private double[] o_biases;
        private Random rnd;
        public NeutralNetwork (int num_input, int num_hidden, int num_output, int seed) {

        }
        // helper methods
        public void SetWeight (double[] weights) { }
        public double[] GetWeights () {
            return outputs; // tmp
        }
        public double[] ComputeOutputs (double[] x_values) {
            // preliminary values
            var h_sums = new double[num_hidden]; // scratch array
            var o_sums = new double[num_output];

            for (int i = 0; i < num_input; i++) {
                this.inputs[i] = x_values[i]; // copy independent vars into input nodes
            }
            for (int j = 0; j < num_hidden; j++)
                for (int i = 0; i < num_input; i++) {
                    h_sums[j] += this.inputs[i] * this.ih_weights[i][j]; // full-connect network
                }
            // add the bias
            for (int j = 0; j < num_hidden; j++)
                h_sums[j] += this.h_biases[j];
            //activation
            for (int j = 0; j < num_hidden; j++) {
                this.hiddens[j] = HyperTan (h_sums[j]);
            }
            for (int k = 0; k < num_output; k++)
                for (int j = 0; j < num_hidden; j++) {
                    o_sums[k] += hiddens[j] * ho_weights[j][k];
                }
            for (int k = 0; k < num_output; k++) {
                o_sums[k] += o_biases[k];
            }
            // no softmax activation in regression applied. Just copy
            Array.Copy (o_sums, this.outputs, outputs.Length);
            double[] res_result = new double[num_output];
            Array.Copy (this.outputs, res_result, res_result.Length);
            return res_result;
        }

        private static double HyperTan (double x) // hyperbolic tan
        {
            if (x < -20)
                return -1.0;
            else if (x > 20)
                return 1.0; // approximation is correct to 30 decimals
            else return Math.Tanh (x);

        }
        private static double Sigmoid (double x) {
            return (1 / 1 + Math.Exp (-x));
        }
        private static double[][] MakeMatrix (int rows, int cols, double init_val) //helper method
        {
            var result = new double[rows][];
            for (int r = 0; r < result.Length; r++) {
                result[r] = new double[cols];
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] = init_val;
                }
            }
            return result;
        }
        public double[] Train (double[][] train_data, int max_epochs, double lr, double momentum) {
            // back-prop specific arrays
            var ho_grads = MakeMatrix (num_hidden, num_output, 0.0);
            var ob_grads = new double[num_output];

            var ih_grad = MakeMatrix (num_input, num_hidden, 0.0);
            var ih_bias = new double[num_hidden];

            // signal
            var o_signals = new double[num_output]; // signals == gradients w/o associated input terms
            var h_signals = new double[num_hidden];

            //backprop-momentum specific array
            var ih_prev_weights_delta = MakeMatrix (num_input, num_hidden, 0.0);
            var h_prev_biases_delta = new double[num_hidden];
            var hp_prev_weights_delta = MakeMatrix (num_hidden, num_output, 0.0);
            var o_prev_biasis_delta = new double[num_output];
            // train NN using lr and momentum
            var epoch = 0;
            var x_values = new double[num_input]; // input vals
            var t_values = new double[num_output]; // target vals

            var sequence = new int[train_data.Length];
            for (int i = 0; i < sequence.Length; i++) {
                sequence[i] = i;
            }

            var err_intercal = max_epochs / 10; // interval to check validation data
            while (epoch < max_epochs) { // every epoch
                epoch++;
                if (epoch % err_intercal == 0 && epoch < max_epochs) {
                    // check err
                    var train_err = 0.0;
                    System.Console.WriteLine ("epoch= " + epoch + "training error = " + train_err.ToString ("F4"));
                }
                shuffle (sequence); // shuffle the order

                for (int ii = 0; ii < train_data.Length; ii++) {
                    int idx = sequence[ii];
                    Array.Copy (train_data[idx], x_values, num_input);
                    Array.Copy (train_data[idx], num_input, t_values, 0, num_output);
                    ComputeOutputs (x_values);

                    // i=inputs j=hidden(s) k=outputs
                }

                // 1. compute output nodes signals
                for (int k = 0; k < num_output; k++) {
                    var derivatives = 1.0; // dummy
                    o_signals[k] = (t_values[k] - outputs[k]) * derivatives;
                }
                // 2. compute h-to-o weights gradients using output signals
                for (int j = 0; j < num_hidden; j++) {
                    for (int k = 0; k < num_output; k++) {
                        ho_grads[j][k] = o_signals[k] * hiddens[j];
                    }
                }
                // 2'. compute the output biases grads using output signals
                for (int k = 0; k < num_output; k++) {
                    ob_grads[k] = o_signals[k] * 1.0;
                }
                // 3.comput hidden nodes signals
                for (int j = 0; j < num_hidden; j++) {
                    var sum = 0.0;
                    for (int k = 0; k < num_output; k++) {
                        sum += o_signals[k] * ho_weights[j][k];
                    }
                    var derivatives = (1 + hiddens[j]) * (1 - hiddens[j]); // for tanh
                    h_signals[j] = sum * derivatives;
                }

            }

            return outputs; //tmp
        }

        private void shuffle (int[] sequence) {
            for (int i = 0; i < sequence.Length; i++) {
                var r = rnd.Next (i, sequence.Length);
                var tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }
        private double Error (double[][] data) {
            // MSE : average squared error per training item
            var sum_squared_err = 0.0;
            var x_values = new double[num_input]; // intput x(first input_num vals in train_data)
            var t_values = new double[num_output]; // output y(last output_num vals in train_data)
            for (int i = 0; i < data.Length; i++) {
                Array.Copy (data[i], x_values, num_input);
                Array.Copy (data[i], num_input, t_values, 0, num_output);
                var y_vals = this.ComputeOutputs (x_values);
                for (int j = 0; j < num_output; j++) {
                    var err = t_values[j] - y_vals[j]; // calc
                    sum_squared_err += Math.Pow (err, 2);
                }
            }
            return sum_squared_err / data.Length;
        } //Error
    } // class NeuralNetwork
} //ns