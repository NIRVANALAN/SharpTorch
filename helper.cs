using System;

namespace cs_nn_fm
{
    public class Helper
    {
        public static double[][] MakeMatrix(int rows, int cols, double init_val) //helper method
        {
            var res = new double[rows][];
            for (int r = 0; r < res.Length; r++)
            {
                res[r] = new double[cols];
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    res[i][j] = init_val;
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