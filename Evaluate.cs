
using System;

namespace cs_nn_fm
{
    public class Evaluate // TODO need more test
    {
        public static double MSE(Model model, Dataset testSet)// use testSet to test,TODO use dataLoader in the future
        {
            // MSE : average squared error per training item
//            var data = model.Nodes;
//            均方方差
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
    }
}