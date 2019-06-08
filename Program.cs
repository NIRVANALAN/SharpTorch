using System;

namespace cs_nn_fm
{
    internal class Program
    {
        public void ManualPropSin(SinTrainData sTrainData, int numMaxEpochs, bool shuffle_flag,
            out double[] InitialWeights, out double[] FinalWeights, bool debug = false)
        {
            Console.WriteLine("NN Regression for predicting sin(x) begin:");
            // create training data 
//            var sTrainData = new SinTrainData(500);
            System.Console.WriteLine("\nTraining data:");
            Helper.ShowMatrix(sTrainData.DataSet, 5, 4, true);
            // create NN 
            const int numInput = 1;
            var numHidden = 12;
            var numOutput = 1; // regression
            var rndSeed = 0;
            System.Console.WriteLine("\n Creating a " + numInput + "-" + numHidden + "-" + numOutput +
                                     " regression neural network");
            var nn = new nn(numInput, numHidden, numOutput, rndSeed);
//            const int numMaxEpochs = 1000;
            const double lr = 0.005;
            const double momentum = 0.001; // need more test
            Console.WriteLine("\nSetting maxEpochs = " + numMaxEpochs);
            Console.WriteLine("Setting learnRate = " + lr.ToString("F4"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F4"));
            // train NN 
            InitialWeights = new double[37]; //just for test
            InitialWeights = nn.GetWeights();
            Console.WriteLine("Initial weights");
            Helper.ShowVector(InitialWeights, 6, 8, true);
            FinalWeights = nn.Train(sTrainData, numMaxEpochs, lr, momentum, shuffle_flag, ref InitialWeights, debug);
            if (debug)
            {
                Console.WriteLine("final weights after training");
                Helper.ShowVector(FinalWeights);
            }
            // show model
//            System.Console.WriteLine("Final model weights:");
//            Helper.ShowVector(weights, 4, 4, true);

            // Test Example
//            var y = nn.Forward(new[] {Math.PI});
//            Console.WriteLine("\nActual sin(PI) =  0.0   Predicted =  " + y[0].ToString("F6"));

//            y = nn.Forward(new[] {Math.PI / 2});
//            Console.WriteLine("\nActual sin(PI / 2)  =  1.0   Predicted =  " + y[0].ToString("F6"));

//            y = nn.Forward(new[] {3 * Math.PI / 2.0});
//            Console.WriteLine("\nActual sin(3*PI / 2) = -1.0   Predicted = " + y[0].ToString("F6"));
        }

        public void RegressionUsingSGD(SinTrainData sinTrainData, bool shuffle_flag, int numEpochs,
            ref double[] initialWeights, out double[] finalWeights, bool provideInitialWeights = false)
        {
            var inputLayer = new Linear(1, 13);
            var activationLayer1TanH = new HyperTan();
            var hiddenLayer1 = new Linear(13, 13);
            var activationLayer2TanH = new HyperTan();
            var hiddenLayer2 = new Linear(13, 1);
            var model = new Model(new Layer[] {inputLayer, activationLayer1TanH, hiddenLayer1, activationLayer2TanH, hiddenLayer2});
//            var model = new Model(new Layer[] {inputLayer, activationLayer1TanH, hiddenLayer2});
            if (provideInitialWeights)
            {
                model.SetWeights(initialWeights);
            }
            const double learning_rate = 0.005;
            const double momentum = 0.001; // need more test
            // generate dataSet 
//            var sinTrainData = new SinTrainData(); // Inherit from DataSet class:generate 1000 items default
            //generate trainData, testData
            Helper.SplitTrainTest(sinTrainData, 0.8, 1, trainData: out var trainSet, testData: out var testSet);

            var optimizer = new SGD(model, learning_rate, momentum); // train via sgd
            var dataLoader = new DataLoader(trainSet, 1, shuffle_flag);
            //train using SGD
            var epoch = 0;
            var epochInternal = 10;
            while (epoch < numEpochs)
            {
                epoch++;
                if (epoch % epochInternal == 0 && epoch > 0) //print epoch & error info
                {
                    var mse = Evaluate.MSE(model, testSet);
                    Console.WriteLine("epoch = " + epoch + " acc = " + (1 - mse).ToString("F4"));
                    //                    Console.WriteLine(" MSE: " + mse.ToString());
                }

                for (int i = 0; i < dataLoader.loopTime; i++)
                {
                    var data = dataLoader.Enumerate(); //[1][x,y]
//                    Console.WriteLine(i);
                    Helper.SplitInputOutput(data, out var inputData, out var outputData);
                    var yPred = model.Forward(inputData[0]); //xValue
//                    Console.WriteLine("framework output:" + yPred.PredictedLinearValues[0]);
                    //compute and print loss
                    var loss = new RegressionLoss(yPred, outputData[0]); //tValue
                    //  Console.WriteLine(loss.Item()); // print loss
                    optimizer.Zero_grad(); // refresh buffer before .backward()
                    loss.Backward(); // calculate grads
                    optimizer.Step(); // update weights
                }
            }

            finalWeights = model.GetWeights();


            // test
            var y = model.Forward(new[] {Math.PI}).PredictedLinearValues;
            Console.WriteLine("\nActual sin(PI) =  0.0   Predicted =  " + y[0].ToString("F6"));
        }

        public void testUnit()
        {
            var testDataSet = new double[5][];
            testDataSet[0] = new double[] {Math.PI, Math.Sin(Math.PI)};
            testDataSet[1] = new double[] {Math.PI / 2, Math.Sin(Math.PI / 2)};
            testDataSet[2] = new double[] {Math.PI / 3, Math.Sin(Math.PI / 3)};
            testDataSet[3] = new double[] {Math.PI * 2, Math.Sin(Math.PI * 2)};
            testDataSet[4] = new double[] {Math.PI * 3 / 2, Math.Sin(Math.PI * 3 / 2)};
            //            var sTrainData = new SinTrainData(dataSet: testDataSet, provideDataFlag: true);
        }


        public void ClassificationUsingSoftmax()
        {
            var inputLayer = new Linear(1, 12);
            var activationLayer1TanH = new HyperTan();
            var hiddenLayer = new Linear(12, 1);
            var softmaxLayer = new SoftMax();
            var model = new Model(new Layer[] {inputLayer, activationLayer1TanH, hiddenLayer, softmaxLayer});
            // implement yourself~~

        }

        static void Main(string[] args)//TODO BGD
        {
            var epochNum = 100;
            var program = new Program();
            var sTrainData = new SinTrainData(1000);
            var initialWeights = new double[1];//tmp
//            program.ManualPropSin(sTrainData, epochNum, false, InitialWeights: out var initialWeights,
//                FinalWeights: out var finalWeights, debug: false);
//            Console.WriteLine("");
            program.RegressionUsingSGD(sTrainData, true, epochNum, initialWeights: ref initialWeights,
                finalWeights: out var finalWeights);

        }
    }
}