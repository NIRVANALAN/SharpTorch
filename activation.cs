using System;
namespace ann {
    class Activation {
        public static double HyperTan (double x) // hyperbolic tan
        {
            if (x < -20)
                return -1.0;
            else if (x > 20)
                return 1.0; // approximation is correct to 30 decimals
            else return Math.Tanh (x);

        }
        public static double Sigmoid (double x) { // sigmoid activation
            return (1 / 1 + Math.Exp (-x));
        }
    }
}