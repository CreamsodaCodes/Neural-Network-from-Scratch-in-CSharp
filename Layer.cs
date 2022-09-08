using System;

namespace app
{
    class Layer
    {
        // Cost gradient with respect to weights and with respect to biases
	    double[,] costGradientW;
	    double[] costGradientB;
        double[] safeweightedInputs;
        public int numNodesIn; int numNodesOut;
        double[,] weights;
        double[] bias;
        public double[] inputs;

        public Layer(int numNodesIn, int numNodesOut){
            this.numNodesIn = numNodesIn;
            this.numNodesOut = numNodesOut;
            safeweightedInputs = new double[numNodesOut];
            weights = new double[numNodesIn,numNodesOut];
            bias = new double[numNodesOut];
            costGradientW = new double[numNodesIn,numNodesOut];
            costGradientB = new double[numNodesOut];
            InitializeRandomWeights();
        }

        public double[] calculateOutputs(double[] inputs){
            double[] weightedInputs = new double[numNodesOut];
            this.inputs = inputs;
            for(int nodeOut = 0; nodeOut<numNodesOut;nodeOut++){
                double weightedInput = bias[nodeOut];
                for(int nodeIn=0; nodeIn<numNodesIn;nodeIn++){
                    weightedInput += inputs[nodeIn] * weights[nodeIn,nodeOut];
                    
                }
                
                weightedInputs[nodeOut] = activationFunction(weightedInput);
                //Console.WriteLine(weightedInputs[nodeOut]);
            }
            safeweightedInputs = weightedInputs;
            return weightedInputs;
        }


        public double[] CalculateOutputLayerNodeValues(double[] expectedOutputs){
            double[] nodeValues = new double[expectedOutputs.Length];

            for(int i = 0;i<nodeValues.Length;i++){
                double costDerivative = NodeCostDerivative(safeweightedInputs[i], expectedOutputs[i]);
                double activationDerivative = ActivationDerivative(safeweightedInputs[i]);
                nodeValues[i] = activationDerivative*costDerivative;
            }
            return nodeValues;
        }



        public double activationFunction(double weightedInput){
            return 1/(1+Math.Exp(-weightedInput));
        }
        public double ActivationDerivative(double weightedInput){
            double activation = weightedInput;
            return activation*(1-activation);
        }

        public double NodeCost(double outputActivation, double expectedOutput){
            double error = outputActivation-expectedOutput;
            return error * error;
        }
        public double NodeCostDerivative(double outputActivation, double expectedOutput){
            return 2*(outputActivation-expectedOutput);
        }

        public void UpdateGradients(double[] nodeValues){
            for(int nodeOut=0;nodeOut<numNodesOut;nodeOut++){

                for(int nodeIn=0;nodeIn<numNodesIn;nodeIn++){
                    //inputs noch nirgends gespeichert?!? erledigt
                    double derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut];
                    //auch nicht fespeichert?!
                    costGradientW[nodeIn, nodeOut] += derivativeCostWrtWeight;
                }
            
            double derivativeCostWrtBias = 1*nodeValues[nodeOut];
            // wo gespeichert?!?! (wo banane :)
            costGradientB[nodeOut] += derivativeCostWrtBias;
            }
        }

        public void ApplyGradients(double learnRate)
        {
            //Console.WriteLine("layer.inputs[0]");
            for(int nodeOut=0;nodeOut<numNodesOut;nodeOut++){
                bias[nodeOut] -= costGradientB[nodeOut]*learnRate;
                for(int nodeIn=0;nodeIn<numNodesIn;nodeIn++){
                    weights[nodeIn,nodeOut] -= costGradientW[nodeIn, nodeOut]* learnRate;
                }
            }
        }

        public double[] CalculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues){
            double[] newNodeValues = new double[numNodesOut];

            for(int newNodeIndex=0; newNodeIndex<newNodeValues.Length;newNodeIndex++){
                double newNodeValue=0;
                for(int oldNodeIndex=0;oldNodeIndex<oldNodeValues.Length;oldNodeIndex++){
                    double weightedInputDerivative = oldLayer.weights[newNodeIndex,oldNodeIndex];
                    newNodeValue += weightedInputDerivative*oldNodeValues[oldNodeIndex];
                }
                //wo weighted Inputs
                newNodeValue *= ActivationDerivative(safeweightedInputs[newNodeIndex]);
                newNodeValues[newNodeIndex]=newNodeValue;
            }
            return newNodeValues;
        }

        public void InitializeRandomWeights(){
            System.Random rng = new System.Random();

            for(int nodeIn=0;nodeIn<numNodesIn;nodeIn++){
                for(int nodeOut=0;nodeOut<numNodesOut;nodeOut++){
                    double randomValue = rng.NextDouble() * 2 - 1;
                    weights[nodeIn,nodeOut]= randomValue / Math.Sqrt(numNodesIn);
                }
            }
        }

        public void ClearGradient(){
            Array.Clear(costGradientW, 0, costGradientW.Length);
            Array.Clear(costGradientB, 0, costGradientB.Length);
            //Console.WriteLine(costGradientW[0,1]);
        }
        


    }
}