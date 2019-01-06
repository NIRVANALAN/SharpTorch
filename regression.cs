using System;

namespace cs_nn_fm {
    class Program {
        static void Main (string[] args) {
            Console.WriteLine ("NN Regression for predicting sin(x) begin:");
            // create training data 
            const int numItems = 100;
            var trainData = new double[numItems][];
            var rnd = new Random (1);
            for (int i = 0; i < numItems; i++) {
                double x = 6.4 * rnd.NextDouble ();
                double sinX = Math.Sin (x);
                trainData[i] = new double[] { x, sinX };
            }
            System.Console.WriteLine ("\nTraining data:");
            HelperClass.ShowMatrix (trainData, 10, 4, true);
            // create NN 
            const int numInput = 1;
            var numHidden = 12;
            var numOutput = 1;
            var rndSeed = 0;
            System.Console.WriteLine ("\n Creating a " + numInput + "-" + numHidden + "-" + numOutput + " regression neural network");
            var nn = new NeutralNetwork (numInput, numHidden, numOutput, rndSeed);
            const int max_epochs = 1000;
            const double lr = 0.004;
            const double momentum = 0.001; // need more test
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
    }

}