using System;

namespace ann {
    class Program {
        static void Main (string[] args) {
            Console.WriteLine ("NN Regression for predicting sin(x) begin:");
            // create training data 
            int num_items = 100;
            var train_data = new double[num_items][];
            var rnd = new Random (1);
            for (int i = 0; i < num_items; i++) {
                double x = 6.4 * rnd.NextDouble ();
                double sin_x = Math.Sin (x);
                train_data[i] = new double[] { x, sin_x };
            }
            System.Console.WriteLine ("\nTraining data:");
            ShowMatrix (train_data, 10, 4, true);
            // create NN 
            var num_input = 1;
            var num_hidden = 12;
            var num_output = 1;
            var rnd_seed = 0;
            System.Console.WriteLine ("\n Creating a" + num_input + "-" + num_hidden + "-" + num_output + "regression neural network");
            var nn = new NeutralNetwork (num_input, num_hidden, num_output, rnd_seed);
            var max_epochs = 1000;
            var lr = 0.004;
            var momentum = 0.001; // need more test
            Console.WriteLine ("\nSetting maxEpochs = " + max_epochs);
            Console.WriteLine ("Setting learnRate = " + lr.ToString ("F4"));
            Console.WriteLine ("Setting momentum  = " + momentum.ToString ("F4"));
            // train NN 
            System.Console.WriteLine ("stochastic back-propogation training start:");
            // var weights = nn.Train (train_data, max_epochs, lr, momentum);
            System.Console.WriteLine ("Finished training");
            System.Console.WriteLine ("Final model weights:");

            // Evaluate NN
        }
        public static void ShowMatrix (double[][] matrix, int numRows, int decimals, bool indices) {
            int len = matrix.Length.ToString ().Length; // refractor?
            for (int i = 0; i < numRows; i++) {
                if (indices)
                    System.Console.Write ("[" + i.ToString ().PadLeft (len) + "] ");
                for (int j = 0; j < matrix[i].Length; j++) {
                    var v = matrix[i][j];
                    if (v >= 0.0) //refractor?
                        System.Console.Write (" "); //'+'
                    System.Console.Write (v.ToString ("F" + decimals) + "  ");
                }
                System.Console.WriteLine ("");
            }
            if (numRows < matrix.Length) {
                System.Console.WriteLine (". . .");
                int last_row = matrix.Length - 1;
                if (indices)
                    Console.Write ("[" + last_row.ToString ().PadLeft (len) + "]");
                for (int j = 0; j < matrix[last_row].Length; j++) {
                    var v = matrix[last_row][j];
                    if (v >= 0.0)
                        System.Console.Write ("  ");
                    System.Console.Write (v.ToString ("F" + decimals) + "  ");;
                }
            }
            System.Console.WriteLine ("\n");
        }
    }
}