
using System;

namespace cs_nn_fm
{
    public class Evaluate // TODO need more test
    {
        public static double MSE(Model model, Dataset testSet, string reduction = "sum")// use testSet to test,TODO use dataLoader in the future
        {
            // MSE : average squared error per training item
//            var data = model.Nodes;
            var sumSquaredErr = 0.0;
            var inputNum = model.InputNum;
            var outputNum = model.OutputNum;
            var xValues = new double[inputNum];
            var tValues = new double[outputNum]; // output y(last num_output vals in train_data)
            var data = testSet.DataSet;
            for (int i = 0; i < data.Length; i++)
            {
                Array.Copy(data[i], xValues, inputNum);
                Array.Copy(data[i], inputNum, tValues, 0, outputNum);
                var yValues = model.Forward(xValues).PredictedValues; // get PredictedValues from current weights
                for (int j = 0; j < outputNum; j++)
                {
                    var err = tValues[j] - yValues[j]; // calc
                    sumSquaredErr += Math.Pow(err, 2);
                }
            }

            return sumSquaredErr / data.Length;
        } //Error

        public static double SoftmaxAcc(Model model, Dataset testSet)
        {
        // percentage correct using winner-takes all
        int numCorrect = 0;
        int numWrong = 0;
        var numInput = model.InputNum;
        var numOutput = model.OutputNum;
        var testData = testSet.DataSet;
        double[] xValues = new double[numInput]; // inputs
        double[] tValues = new double[numOutput]; // targets
        double[] yValues; // computed Y

            for (int i = 0; i<testData.Length; ++i)
        {
            Array.Copy(testData[i], xValues, numInput); // get x-values
            Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
            yValues = model.Forward(xValues).PredictedValues;
            int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
            int tMaxIndex = MaxIndex(tValues);

            if (maxIndex == tMaxIndex)
                ++numCorrect;
            else
                ++numWrong;
        }
        return (numCorrect* 1.0) / (numCorrect + numWrong);
        }
        private static int MaxIndex(double[] vector) // helper for SoftmaxAccy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
    }
}