using System;

namespace cs_nn_fm
{
    internal class Program
    {
        public void ManualPropSin()
        {
            Console.WriteLine("NN Regression for predicting sin(x) begin:");
            // create training data 
            var sTrainData = new SinTrainData(500);
            System.Console.WriteLine("\nTraining data:");
            Helper.ShowMatrix(sTrainData.DataSet, 10, 4, true);
            // create NN 
            const int numInput = 1;
            var numHidden = 12;
            var numOutput = 1; // regression
            var rndSeed = 0;
            System.Console.WriteLine("\n Creating a " + numInput + "-" + numHidden + "-" + numOutput +
                                     " regression neural network");
            var nn = new nn(numInput, numHidden, numOutput, rndSeed);
            const int numMaxEpochs = 1000;
            const double lr = 0.005;
            const double momentum = 0.001; // need more test
            Console.WriteLine("\nSetting maxEpochs = " + numMaxEpochs);
            Console.WriteLine("Setting learnRate = " + lr.ToString("F4"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F4"));
            // train NN 
            var weights = nn.Train(sTrainData, numMaxEpochs, lr, momentum);
            // show model
            System.Console.WriteLine("Final model weights:");
            Helper.ShowVector(weights, 4, 4, true);

            // Test Example
            var y = nn.Forward(new[] {Math.PI});
            Console.WriteLine("\nActual sin(PI) =  0.0   Predicted =  " + y[0].ToString("F6"));

            y = nn.Forward(new[] {Math.PI / 2});
            Console.WriteLine("\nActual sin(PI / 2)  =  1.0   Predicted =  " + y[0].ToString("F6"));

            y = nn.Forward(new[] {3 * Math.PI / 2.0});
            Console.WriteLine("\nActual sin(3*PI / 2) = -1.0   Predicted = " + y[0].ToString("F6"));
        }

        public void RegressionUsingFrameWork()
        {
            var inputLayer = new Linear(1, 12);
            var activationLayer1TanH = new HyperTan();
            var hiddenLayer = new Linear(12, 1);
            var model = new Model(new Layer[] {inputLayer, activationLayer1TanH, hiddenLayer}); // create model
            const int numEpochs = 500;
            const double learning_rate = 0.005;
            const double momentum = 0.001; // need more test
            // generate dataSet 
            var sTrainData = new SinTrainData(); // Inherit from DataSet class:generate 1000 items default
            //generate trainData, testData
            Helper.SplitTrainTest(sTrainData, 0.8, 1, trainData: out var trainSet, testData: out var testSet);

            var optimizer = new SGD(model, learning_rate, momentum); // train via sgd
            var dataLoader = new DataLoader(trainSet, 1, true);
            //train using SGD
            for (int i = 0; i < numEpochs; i++)
            {
                var data = dataLoader.Enumerate(); //[1][x,y]
                Helper.SplitInputOutput(data, out var inputData, out var outputData);
                var yPred = model.Forward(inputData[0]); //xValue
                //compute and print loss
                var loss = new RegressionLoss(yPred, outputData[0]); //tValue
                //  Console.WriteLine(loss.Item()); // print loss
                if (i % 10 == 0 && i > 0) //print epoch & error info
                {
                    var mse = Evaluate.MSE(model, testSet);
                    Console.WriteLine("epoch = " + i + " acc = " + (1 - mse).ToString("F4"));
//                    Console.WriteLine(" MSE: " + mse.ToString());
                }

                optimizer.Zero_grad(); // refresh buffer before .backward()
                loss.Backward(); // calculate grads
                optimizer.Step(); // update weights
            }

            // test
        }


        static void Main(string[] args)
        {
            var program = new Program();
//            program.ManualPropSin();
            program.RegressionUsingFrameWork();
        }
    }
}