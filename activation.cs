using System;

namespace cs_nn_fm {
    internal class Activation {
        public static double Clamp(double x, double min)
        {
            return x>min ? x : min;
        }
        public static double Relu(double x)
        {
            return Clamp(x, 0); // use clamp
        } // clamp{
        public static double PRelu(double x, double a) { //
            return x < 0 ? a * x : x;
        }
        public static double ELU(double x, double a) { //
            return x < 0 ? a * (Math.Pow(Math.E, x) - 1) : x;
        }
        public static double HyperTan (double x) // hyperbolic tan£¬Ë«ÇúÕýÇÐ
        {
            if (x < -45)
                return -1.0;
            return x > 45 ? 1.0 : Math.Tanh (x);
        }
        public static double Sigmoid (double x) { // sigmoid activation
            return 1 / (1 + Math.Exp (-x));
        }
        public static double ArcTan(double x) { //·´ÕýÇÐ
            return Math.Atan(x);
        }
    }
}