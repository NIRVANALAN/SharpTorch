using System;
namespace ann {
    public class HelperClass {
        public static void showVector (double[] vector, int decimals, int line_len, bool new_line) {
            for (int i = 0; i < vector.Length; i++) {
                if (i % line_len == 0 && i > 0) // avoid the state 
                    System.Console.WriteLine ("");
                System.Console.Write (i.ToString ("F" + decimals) + " ");
            }
            if (new_line == true)
                System.Console.WriteLine ("");
        }
        public static void ShowMatrix (double[][] matrix, int numRows, int decimals, bool indices) {
            int len = matrix.Length.ToString ().Length; // refractor?
            for (int i = 0; i < numRows; i++) {
                if (indices)
                    System.Console.Write ("[" + i.ToString ().PadLeft (len) + "] ");
                for (int j = 0; j < matrix[i].Length; j++) {
                    var v = matrix[i][j];
                    if (v >= 0.0) //refractor?
                        System.Console.Write (" "); //'+'
                    System.Console.Write (v.ToString ("F" + decimals) + "  ");
                }
                System.Console.WriteLine ("");
            }
            if (numRows < matrix.Length) {
                System.Console.WriteLine (". . .");
                int last_row = matrix.Length - 1;
                if (indices)
                    Console.Write ("[" + last_row.ToString ().PadLeft (len) + "]");
                for (int j = 0; j < matrix[last_row].Length; j++) {
                    var v = matrix[last_row][j];
                    if (v >= 0.0)
                        System.Console.Write ("  ");
                    System.Console.Write (v.ToString ("F" + decimals) + "  ");;
                }
            }
            System.Console.WriteLine ("\n");
        }
    }
}