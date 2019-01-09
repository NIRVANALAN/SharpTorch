using System;

namespace cs_nn_fm
{
    public class Helper
    {
        public static void SplitTrainTest(double[][] allData, double trainPct,
            int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int) (totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }

            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // SplitTrainTest

        public static void InitializeWeights(ref double[,] weights, double lo = -0.001, double hi = 0.001,
            int rnd_seed = 1)
        {
            var rnd = new Random(rnd_seed);
            var row = weights.GetLength(0);
            var col = weights.GetLength(1);
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    weights[i, j] = lo + (hi - lo) * rnd.NextDouble();
                }
            }
        }

        public static double[,] MakeMatrix(int rows, int cols, double init_val = 0.0) //helper method
        {
            var res = new double[rows, cols];
//            for (int r = 0; r < res.Length; r++)
//            {
//                res[r] = new double[cols];
//            }

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    res[i, j] = init_val;
                }
            }

            return res;
        }

        public static void ShowVector(double[] vector, int decimals, int line_len, bool new_line)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                if (i % line_len == 0 && i > 0) // avoid the state 
                    System.Console.WriteLine("");
                System.Console.Write(i.ToString("F" + decimals) + " ");
            }

            if (new_line == true)
                System.Console.WriteLine("");
        }

        public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length; // refractor?
            for (int i = 0; i < numRows; i++)
            {
                if (indices)
                    System.Console.Write("[" + i.ToString().PadLeft(len) + "] ");
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    var v = matrix[i][j];
                    if (v >= 0.0) //refractor?
                        System.Console.Write(" "); //'+'
                    System.Console.Write(v.ToString("F" + decimals) + "  ");
                }

                System.Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                System.Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]");
                for (int j = 0; j < matrix[lastRow].Length; j++)
                {
                    var v = matrix[lastRow][j];
                    if (v >= 0.0)
                        System.Console.Write("  ");
                    System.Console.Write(v.ToString("F" + decimals) + "  ");
                    ;
                }
            }

            System.Console.WriteLine("\n");
        }
    }
}