using System;

namespace app
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Random rng = new System.Random();
            DataPoint[] completeData = new DataPoint[10000];
            for(int i = 0;i<5000;i++){
                double [] inputData = new double[2];
                double randomNumber = rng.NextDouble() * 10000;
                
                inputData[1] = randomNumber*randomNumber;
                completeData[i] = new DataPoint(inputData,1,2);
            }
            
            for(int i = 5000;i<10000;i++){
                
                double [] inputData = new double[2];
                double randomNumber = rng.NextDouble() * 10000;
                double randomNumber2 = rng.NextDouble() * 10000;
                inputData[0] = randomNumber;
                inputData[1] = randomNumber2;
                completeData[i] = new DataPoint(inputData,0,2);
            }


        }
    }
}
