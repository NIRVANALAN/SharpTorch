# CSharp_NN_framework
## example ##
Here we define a model
```
var inputLayer = new Linear(1, 13);
var activationLayer1TanH = new HyperTan();
var hiddenLayer1 = new Linear(13, 13);
var activationLayer2TanH = new HyperTan();
var hiddenLayer2 = new Linear(13, 1);
var model = new Model(new Layer[] {inputLayer, activationLayer1TanH, hiddenLayer1, activationLayer2TanH, hiddenLayer2});
```
initiate the weights and define momentum & lr
```
 if (provideInitialWeights)
{
model.SetWeights(initialWeights);
}
const double learning_rate = 0.005;
const double momentum = 0.001; // need more test
```
define optimizer and dataloader
```
var optimizer = new SGD(model, learning_rate, momentum); // train via sgd
var dataLoader = new DataLoader(trainSet, 1, shuffle_flag);
```
Then, we can train the network in pyTorch way, using SGD and MSELOSS
```
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
//                    Console.WriteLine("framework output:" + yPred.PredictedValues[0]);
      //compute and print loss
      var loss = new RegressionLoss(yPred, outputData[0]); //tValue
      //  Console.WriteLine(loss.Item()); // print loss
      optimizer.Zero_grad(); // refresh buffer before .backward()
      loss.Backward(); // calculate grads
      optimizer.Step(); // update weights
  }
}
```
---

This framework implemented common activation function and support regression and classification. 
Optimization method including SGD and ADAM
Run program.cs to see the output :)
···
program.RegressionUsingSGD(sTrainData, true, epochNum, initialWeights: ref initialWeights,
    finalWeights: out var finalWeights);
···
Hope you will enjoy it
