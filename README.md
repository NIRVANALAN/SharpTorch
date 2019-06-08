# CSharp_NN_framework
This project allow you to define the structure of your network in a layer by layer manner, and you can choose what activation function to use per layer. We implemented forward propogation, cost function like 
- MSE
- SoftMax 


to support training, and backward propogation. You can apply this framework on your own regression and classification work in a rather simple way.

## Here is a form of the corresponding relationship between the components of neural network and the modules of our codes, to help you gain a better understanding. ##

| component of NN | module of the code |
| :-------------- | :----------------- |


## In the follow, we will give an example as regard to how to use this framework. ##
Here we define a model
```
var inputLayer = new Linear(1, 13);
var activationLayer1TanH = new HyperTan();
var hiddenLayer1 = new Linear(13, 13);
var activationLayer2TanH = new HyperTan();
var hiddenLayer2 = new Linear(13, 1);
var model = new Model(new Layer[] {inputLayer, activationLayer1TanH, hiddenLayer1, activationLayer2TanH, hiddenLayer2});
```
initiate the weights and define momentum & learning rate
```
 if (provideInitialWeights)
{
model.SetWeights(initialWeights);
}
const double learning_rate = 0.005;
const double momentum = 0.001;
```
define optimizer and dataloader
```
var optimizer = new SGD(model, learning_rate, momentum);
var dataLoader = new DataLoader(trainSet, 1, shuffle_flag);
```
Then, we can train the network in pyTorch way, using SGD and MSELOSS
```
while (epoch < numEpochs)
{
  epoch++;
  if (epoch % epochInternal == 0 && epoch > 0)
  {
      var mse = Evaluate.MSE(model, testSet);
      Console.WriteLine("epoch = " + epoch + " acc = " + (1 - mse).ToString("F4"));
  }

  for (int i = 0; i < dataLoader.loopTime; i++)
  {
      var data = dataLoader.Enumerate();
      Helper.SplitInputOutput(data, out var inputData, out var outputData);
      var yPred = model.Forward(inputData[0]);
      var loss = new RegressionLoss(yPred, outputData[0]);
      optimizer.Zero_grad();
      before .backward()
      loss.Backward();
      optimizer.Step();
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
