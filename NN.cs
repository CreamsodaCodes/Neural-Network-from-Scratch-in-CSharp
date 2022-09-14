using System;

namespace app
{
    class NN
    {
        Layer[] layers;
        public NN(params int[] layerSize){
            layers = new Layer[layerSize.Length-1];
            for(int i=0;i<layers.Length;i++){
                layers[i] = new Layer(layerSize[i],layerSize[i+1]);
            }
            
        }

    public double[] CalculateOutputs(double[] inputs)
	{
		foreach (Layer layer in layers)
		{
			inputs = layer.calculateOutputs(inputs);
		}
		return inputs;
	}

    public void classifyAll(DataPoint[] data){
        int i = 0;
        foreach (DataPoint dataPoint in data){
            if((Classify(dataPoint.inputs)==(int)dataPoint.label)){
                Console.WriteLine("Nice");
                i++;
            }
        }
        
        Console.WriteLine(i);
        Console.WriteLine(data.Length);
    }

    public int Classify(double[] inputs){
        double[] outputs = CalculateOutputs(inputs);
        
        /*System.Console.WriteLine(outputs[0]);
        System.Console.WriteLine(outputs[1]);*/
        Console.WriteLine(MaxValueIndex(outputs));
        return MaxValueIndex(outputs);
    }

    double cost(DataPoint dataPoint){
        double[] outputs = CalculateOutputs(dataPoint.inputs);
        Layer outputLayer = layers[layers.Length-1];
        double cost=0;

        for(int nodeOut=0;nodeOut<outputs.Length;nodeOut++){
        
            cost += outputLayer.NodeCost(outputs[nodeOut],dataPoint.expectedOutputs[nodeOut]);
        }
        
        return cost;
    }
    public double cost(DataPoint[] data){
        double totalCost = 0;
        foreach (DataPoint dataPoint in data){
            totalCost += cost(dataPoint);
            
        }
        
        return totalCost/data.Length;
    }

    public int MaxValueIndex(double[] values)
	{
		double maxValue = double.MinValue;
		int index = 0;
		for (int i = 0; i < values.Length; i++)
		{
			if (values[i] > maxValue)
			{
				maxValue = values[i];
				index = i;
			}
		}

		return index;
	}

    void UpdateAllGradients(DataPoint dataPoint){
        
        CalculateOutputs(dataPoint.inputs);
        
        Layer outputLayer = layers[layers.Length-1];
        
        double[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs);
        
        outputLayer.UpdateGradients(nodeValues);
        

        for(int hindenLayerIndex= layers.Length-2; hindenLayerIndex>=0;hindenLayerIndex--){
        Layer hiddenLayer = layers[hindenLayerIndex];
        nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hindenLayerIndex+1], nodeValues);
        hiddenLayer.UpdateGradients(nodeValues);
        }
    }

    public void Learn(DataPoint[] trainingBatch, double learnRate){
        foreach (DataPoint dataPoint in trainingBatch)
        {
            
            UpdateAllGradients(dataPoint);
        }
        
        //wo?
        ApplyAllGradients(learnRate / trainingBatch.Length);
        //wo??
        ClearAllGradients();
        
    }

    public void ApplyAllGradients(double learnRate){
        foreach (Layer layer in layers)
        {
            
            layer.ApplyGradients(learnRate);
        }

    }
    public void ClearAllGradients(){
        foreach (Layer layer in layers)
        {
            layer.ClearGradient();
        }
    }
    }
}