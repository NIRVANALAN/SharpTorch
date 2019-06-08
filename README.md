# CSharp_NN_framework
This project allow you to define the structure of your network in a layer by layer manner, and you can choose what activation function to use per layer. We implemented forward propogation, cost function such as 
- MSE
- SoftMax 


to support training, and backward propogation and other helper functions. You can apply this framework on your own regression and classification work in a rather simple way.

## Here is a form of succinct description to what functions all the modules of our codes perform. Hope it will help you gain a better understanding of this framework. ##

| Component                          | Description                                                                                                                                                                |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DataLoader.cs, Dataset.cs, help.cs | Load dataset and perform data pre-process like shuffle, split training set, dev set and test set, initialize weight matrices and so on.                                    |
| Evaluate.cs                        | Define cost functions, you can add your own codes regarding evaluation of the performance of models here.                                                                  |
| layer.cs                           | Define the structure of each layer and the activation function of each layer such as Relu, L-Relu, sigmoid, tanh and so on. We also plan to implement drop out later here. |
| loss.cs                            | Define the calculation function to compute loss on training sets.                                                                                                          |
| model.cs                           | Aggregate all layers defined previously, implement the complete forward propogation and backward propogation.                                                              |
| NN.cs                              | Implement the whole process of training.                                                                                                                                   |
| Optimizer.cs                       | Define different ways to optimize the final result.                                                                                                                        |
| Program.cs                         | A test case to validify that our program is useful.                                                                                                                        |

## In the follow, we will give demos as regard to how to use this framework. ##
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
## Installation
---
### Check your C# version
As C# is a cross-platform languge, you can run this code in both *nix & windows platform. Both .net core and .NET >=4.6 are supported.
### Get the C#_NN_Framework Source
```
git clone https://github.com/leegolan/CSharp_NN_framework.git 
// Run program.cs to see the output :)
```
Then, you can run the test case in program.cs, which use our framework to create several linear layers for predicting the value of Sin(X) out of X from training set.
---
# The Team

We write this framework out of our passion and love for deep learning & neutral network, which implemented common activation function and support regression and classification. Basically, the project is aiming at implementing simple but robust linear layer network, using forward and back-prop to train model.

This repo is designed, contributed and currently maintained by [YushiLAN](https://github.com/NIRVANALAN) and [XiaotongLi](https://github.com/yellowducklet). I think this repo can be a good beginner's guide to Neutral Network and DeepLearning, which concisely explained the math principle of forward&backward propgation. Dataloader and popular activation functions are also added, which we hope would help.

# LICENSE
This repo is MIT-style licensed, as found in the LICENSE file.

