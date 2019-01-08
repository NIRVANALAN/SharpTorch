using System;

namespace cs_nn_fm
{
    public class Evaluate
    {
        public static double MSE(double[][] data, int input_num, int output_num)
        {
            // MSE : average squared error per training item
            var sumSquaredErr = 0.0;
            var xValues = new double[input_num]; // intput x(first input_num vals in train_data)
            var tValues = new double[output_num]; // output y(last num_output vals in train_data)
            for (int i = 0; i < data.Length; i++)
            {
                Array.Copy(data[i], xValues, input_num);
                Array.Copy(data[i], input_num, tValues, 0, output_num);
                var yValues = Forward(xValues);
                for (int j = 0; j < output_num; j++)
                {
                    var err = tValues[j] - yValues[j]; // calc
                    sumSquaredErr += Math.Pow(err, 2);
                }
            }

            return sumSquaredErr / data.Length;
        } //Error
    }
}