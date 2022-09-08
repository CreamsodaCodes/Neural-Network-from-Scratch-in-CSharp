using System;

namespace app
{
    class CreatTestCases
    {
        public static int[] Test(int amount)
        {
           int[] testcases = new int[amount];
           for(int i=0;i<amount;i++){
            testcases[i] = i*i;
           }
           return testcases;
        }
    }
}