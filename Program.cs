using System;

namespace app
{
    class Program
    {
        static int[] LayerSize = {2,5,5,2};
        static double[] Test = {2,4};
        static double[] Test2 = {3,30};
        static double[] Test3 = {4,16};
        static double[] Test4 = {1,6};
        static void Main(string[] args)
        {
            

            NN firstNetwork = new NN(LayerSize);
            //firstNetwork.Learn(creatData(),0.0001);
            //firstNetwork.Learn(creatData(),0.0001);
            //firstNetwork.Learn(creatData(),0.0001);
            
            for (int i = 0; i < 100000; i++)
            {
                DataPoint[] testData = creatData();
                firstNetwork.Learn(testData,0.1);
                firstNetwork.cost(testData);
            }
            

            firstNetwork.Classify(Test);
            firstNetwork.Classify(Test2);
            firstNetwork.Classify(Test3);
            firstNetwork.Classify(Test4); 
            
            



        }
        public static DataPoint[] creatData(){
            System.Random rng = new System.Random();
            DataPoint[] completeData = new DataPoint[100];
            for(int i = 0;i<100;i++){
                double [] inputData = new double[2];
                int randomNumberInt = rng.Next(10);
                double randomNumber = (double)randomNumberInt;
                
                
                

                if(rng.Next(2)==1){
                    
                    inputData[0] = randomNumber;
                    inputData[1] = randomNumber*randomNumber;
                    completeData[i] = new DataPoint(inputData,1,2);
                }
                else{
                    int randomNumberInt2 = rng.Next(10);
                    if (randomNumberInt2 == randomNumberInt)
                    {
                        randomNumberInt2 = rng.Next(10);
                    }
                    double randomNumber2 = (double)randomNumberInt2;
                    inputData[0] = randomNumber;
                    inputData[1] = randomNumber*randomNumber2;
                    completeData[i] = new DataPoint(inputData,0,2);
                }
                
            }
            
            
            return completeData;
        }
    }
}
