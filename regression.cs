using System;

namespace cs_nn_fm {
    class SinTrainData: Dataset
    {
        private int numItems;
//        double[][] trainData;

        public SinTrainData(int numitems, int seed)
        {
            this.rand_seed = seed;
            this.numItems = numitems;
            Rnd = new Random(rand_seed);
            DataSet2D = new double[numitems][];
            for (int i = 0; i < numItems; i++)
            {
                var x = 2*Math.PI * Rnd.NextDouble();
                var sinX = Math.Sin(x);
                DataSet2D[i] = new[] { x, sinX };
            }

        }

        public override double[] GetItems(int index)
        {
            return DataSet2D[index];
            throw new NotImplementedException();
        }

        public override int GetLen()
        {
            return DataSet2D.Length;
            throw new NotImplementedException();
        }

    }
    internal class Program {
        static void Main (string[] args) {
            Console.WriteLine ("NN Regression for predicting sin(x) begin:");
            // create training data 
            var sTrainData = new SinTrainData(500,2);
            System.Console.WriteLine ("\nTraining data:");
            Helper.ShowMatrix (sTrainData.GetDataSet2D(), 10, 4, true);
            // create NN 
            const int numInput = 1;
            var numHidden = 12;
            var numOutput = 1; // regression
            var rndSeed = 0;
            System.Console.WriteLine ("\n Creating a " + numInput + "-" + numHidden + "-" + numOutput + " regression neural network");
            var nn = new NeutralNetwork (numInput, numHidden, numOutput, rndSeed);
            const int numMaxEpochs = 1000;
            const double lr = 0.005;
            const double momentum = 0.001; // need more test
            Console.WriteLine ("\nSetting maxEpochs = " + numMaxEpochs);
            Console.WriteLine ("Setting learnRate = " + lr.ToString ("F4"));
            Console.WriteLine ("Setting momentum  = " + momentum.ToString ("F4"));
            // train NN 
            var weights = nn.Train (sTrainData, numMaxEpochs, lr, momentum);
            // show model
            System.Console.WriteLine ("Final model weights:");
            Helper.ShowVector(weights,4,4,true);

            // Test Example
            var y = nn.ComputeOutputs(new[] { Math.PI });
            Console.WriteLine("\nActual sin(PI) =  0.0   Predicted =  " + y[0].ToString("F6"));

            y = nn.ComputeOutputs(new[] { Math.PI / 2 });
            Console.WriteLine("\nActual sin(PI / 2)  =  1.0   Predicted =  " + y[0].ToString("F6"));

            y = nn.ComputeOutputs(new[] { 3 * Math.PI / 2.0 });
            Console.WriteLine("\nActual sin(3*PI / 2) = -1.0   Predicted = " + y[0].ToString("F6"));
        }
    }

}