using System;

namespace app
{
    class Program
    {
        static int[] LayerSize = {200,50,50,50,10};
        static double[] Test = new double[784];
        static double test;
        static double[] Test2 = {3,30};
        static double[] Test3 = {4,16};
        static double[] Test4 = {1,6};
        static void Main(string[] args)
        {
            DataPoint[] trainData = creatDataPictures();
            DataPoint[] realTestData = creatDataPicturesTest();
            /*for (int i = 0; i < 784; i++)
            {
                Console.WriteLine(trainData[11].inputs[i]);
            } */
            
            
            NN firstNetwork = new NN(LayerSize);
            firstNetwork.Learn(trainData,1);
           //firstNetwork.classifyAll(creatDataPicturesTest());
           Console.WriteLine(firstNetwork.cost(trainData));
          
            //firstNetwork.Learn(creatData(),0.0001);
            //firstNetwork.Learn(creatData(),0.0001);
            
            for (int i = 0; i < 100000; i++)
            {
                DataPoint[] testData = batchMaker(trainData,100);
                firstNetwork.Learn(testData,2);
                if(i%500 == 0){
                    Console.WriteLine(firstNetwork.cost(testData));
                }
                
            }

            foreach (DataPoint data in realTestData)
            {
                Console.WriteLine(firstNetwork.Classify(data.inputs));
                Console.WriteLine("real:" + data.label);

            }
            Console.WriteLine(firstNetwork.cost(realTestData));


            

            
            

        

        }
        public static DataPoint[] creatData(){
            System.Random rng = new System.Random();
            DataPoint[] completeData = new DataPoint[60000];
            for(int i = 0;i<60000;i++){
                double [] inputData = new double[2];
                int randomNumberInt = rng.Next(100);
                double randomNumber = (double)randomNumberInt;
                
                
                

                if(rng.Next(2)==1){
                    
                    inputData[0] = randomNumber;
                    inputData[1] = randomNumber+5;
                    completeData[i] = new DataPoint(inputData,1,2);
                }
                else{
                    int randomNumberInt2 = rng.Next(100);
                    if (randomNumberInt2 == randomNumberInt)
                    {
                        randomNumberInt2 = rng.Next(100);
                    }
                    double randomNumber2 = (double)randomNumberInt2;
                    inputData[0] = randomNumber;
                    inputData[1] = randomNumber+randomNumber2;
                    completeData[i] = new DataPoint(inputData,0,2);
                }
                
            }
            
            
            return completeData;
        }

        public static DataPoint[] creatDataPictures(){
            
            DataPoint[] completeDataTrainng = new DataPoint[60000];
            int i = 0;
            foreach (var image in MnistReader.ReadTrainingData()){
                double [] inputDataImage = new double[image.Data.Length];
                inputDataImage = Extensions.conv2dTo1d(image.Data,784,28,28);                
                completeDataTrainng[i] = new DataPoint(inputDataImage,image.Label,10);
                i++;
            }
            
            return completeDataTrainng;
        }

        public static DataPoint[] creatDataPicturesTest(){
            //System.Random rng = new System.Random();
            DataPoint[] completeDataTrainng = new DataPoint[10000];
            int i = 0;
            foreach (var image in MnistReader.ReadTestData()){
                double [] inputDataImage = new double[image.Data.Length];
                inputDataImage = Extensions.conv2dTo1d(image.Data,784,28,28);
                completeDataTrainng[i] = new DataPoint(inputDataImage,image.Label,10);
                i++;
            }
            return completeDataTrainng;
        }


        public static DataPoint[] batchMaker(DataPoint[] fullData, int batchSize){
            DataPoint[] batchData = new DataPoint[batchSize];
            System.Random rng = new System.Random();
            for (int i = 0; i < batchSize; i++)
            {
                int rndIndex = rng.Next(fullData.Length);
                batchData[i] = fullData[rndIndex];
            }
            return batchData;
        }

    }
}
