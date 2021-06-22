using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    class ForwardPropagation
    {
        static float positionSum = 0.0f;

        static public void forwardPropagate(ref CNN net) {
            //----------first convolution layer
            for (int i = 0; i<net.filterLayers[0].squares.Length; i++)
            {
                applyFilter(net.input, net.filterLayers[0].squares[i], ref net.convolutedLayers[0].squares[i]);
                net.convolutedLayers[0].squares[i].applyBiases();
                reluSquare(net.convolutedLayers[0].squares[i], ref net.activatedConvolutedLayers[0].squares[i]);
                maxPool(net.activatedConvolutedLayers[0].squares[i], ref net.downsampledLayers[0].squares[i]);
            }

            //--------------second convolution
            int filterPos = 0;            
            for (int c = 0; c < net.convolutedLayers[1].squares.Length; c++)
            {
                for (int f = 0; f < 6; f++)
                {
                    applyFilterAdditively(net.downsampledLayers[0].squares[filterPos % 6], net.filterLayers[1].squares[filterPos], ref net.convolutedLayers[1].squares[c]);
                    filterPos++;
                }
                net.convolutedLayers[1].squares[c].applyBiases();
                reluSquare(net.convolutedLayers[1].squares[c], ref net.activatedConvolutedLayers[1].squares[c]);//apply relu to square
                maxPool(net.activatedConvolutedLayers[1].squares[c], ref net.downsampledLayers[1].squares[c]);
            }

            //-------------fully connected
            int neuronCount = net.hiddenNeurons[0].values.Length;
            int filterCount = net.downsampledLayers[1].squares.Length;

            for (int i=0; i<neuronCount; i++) {
                net.hiddenNeurons[0].values[i]=0.0f;//reset for this iteration
                for (int j = 0; j < filterCount; j++) {                    
                    int filterHeight = net.downsampledLayers[1].squares[0].width;
                    int filterWidth = net.downsampledLayers[1].squares[0].width;

                    //iterate through square
                    for (int y = 0; y < filterHeight; y++) {                        
                        for (int x = 0; x < filterWidth; x++)
                        {
                            //weightPos iterates through each filter as you would read a book. horizontal-vertical-page-book (page being each filter, book being set of filters per neuron)
                            net.hiddenNeurons[0].values[i] += net.weights[0].values[i*filterCount*filterHeight * filterWidth + j*filterHeight*filterWidth + y* filterWidth + x] * net.downsampledLayers[1].squares[j].values[x,y];
                        }                        
                    }                    
                }
                net.hiddenNeurons[0].values[i] += net.biases[0].values[i];//Add bias to neuron after summing weighted inputs
                net.activatedHiddenNeurons[0].values[i] = (float)sigmoid(net.hiddenNeurons[0].values[i]);//after adding all values for current neuron, run it through sigmoid
            }

            //fully connected layer 2
            for (int i = 0; i < net.hiddenNeurons[1].values.Length; i++)
            {
                net.hiddenNeurons[1].values[i] = 0.0f;//reset
                for (int j = 0; j < net.hiddenNeurons[0].values.Length; j++)
                {
                    net.hiddenNeurons[1].values[i] += net.weights[1].values[j * net.hiddenNeurons[1].values.Length + i] * net.activatedHiddenNeurons[0].values[j];
                }
                net.hiddenNeurons[1].values[i] += net.biases[1].values[i];//Add bias to neuron after summing weighted inputs
                net.activatedHiddenNeurons[1].values[i] = (float)sigmoid(net.hiddenNeurons[1].values[i]);//after adding all values for current neuron, run it through sigmoid
            }

            //fully connected output
            for (int i = 0; i < net.outputs.values.Length; i++) {
                net.outputs.values[i] = 0.0f;//reset
                for (int j = 0; j < net.hiddenNeurons[1].values.Length; j++) {
                    net.outputs.values[i] += net.weights[2].values[j* net.outputs.values.Length+i] * net.activatedHiddenNeurons[1].values[j];                    
                }
                net.outputs.values[i] += net.biases[2].values[i];
                //No bias in output layer
                net.activatedOutputs.values[i] = net.outputs.values[i];//Copy over outputs so that softmax can be applied
            }
            //softMax(ref net.activatedOutputs);

            int highestPos = -1;
            float highestVal = float.MinValue;
            for (int i=0; i<net.outputs.values.Length; i++) {
                if (net.outputs.values[i] > highestVal) {
                    highestVal = net.outputs.values[i];
                    highestPos = i;
                }
            }
            net.numberGuess = highestPos;
        }


        static public void applyFilter(Square input, Square filter, ref Square convolution)
        {
            for (int posy = 0; posy < convolution.width; posy++)
            {
                for (int posx = 0; posx < convolution.width; posx++)
                {
                    positionSum = 0.0f;

                    for (int y = 0; y < filter.width; y++)
                    {
                        for (int x = 0; x < filter.width; x++)
                        {
                            positionSum += input.values[posx + x, posy + y] * filter.values[x, y];
                        }
                    }
                    convolution.values[posx, posy] = positionSum;
                }
            }
        }

        static public void applyFilterAdditively(Square input, Square filter, ref Square convolution)
        {
            for (int posy = 0; posy < convolution.width; posy++)
            {
                for (int posx = 0; posx < convolution.width; posx++)
                {
                    positionSum = 0.0f;

                    for (int y = 0; y < filter.width; y++)
                    {
                        for (int x = 0; x < filter.width; x++)
                        {
                            positionSum += input.values[posx + x, posy + y] * filter.values[x, y];
                        }
                    }
                    convolution.values[posx, posy] += positionSum;
                }
            }
        }

        static public double sigmoid(float num) {
            double epow = Math.Pow(Math.E, -num);
            double log = Math.Log(Double.MaxValue);
            if (num > Math.Log(Double.MaxValue)) {
                //Console.WriteLine("Nan zone");
            }
            double output = 1 / (1 + Math.Exp(-num));
            if(output==0)
                output = 1 / (1 + Math.Exp(709));
            return output;
        }

        static public void maxPool(Square inputLayer, ref Square downsampled) {
            for (int posy = 0; posy < downsampled.width; posy++)
            {
                for (int posx = 0; posx < downsampled.width; posx++)
                {
                    float highest = float.MinValue;
                    for (int y = 0; y < 2; y++)
                    {
                        for (int x = 0; x < 2; x++)
                        {
                            if (inputLayer.values[posx*2 + x, posy*2 + y] > highest)
                                highest = inputLayer.values[posx*2 + x, posy*2 + y];
                        }
                    }
                    downsampled.values[posx, posy] = highest;
                }
            }
        }

        static public void reluSquare(Square input, ref Square ouput) {
            for (int y = 0; y < input.width; y++) {
                for (int x=0; x<input.width; x++) {
                    ouput.values[x, y] = input.values[x, y] >= 0 ? input.values[x, y] : 0.01f* input.values[x, y];//if less than zero set to zero else keep.
                }
            }
        }

        static public void softMax(ref SingleDimension input) {
            float sum = 0.0f;
            for (int i = 0; i < input.values.Length; i++) {
                sum+=(float)Math.Pow(Math.E, input.values[i]);
            }
            for (int i = 0; i < input.values.Length; i++)
            {
                float pow = (float)Math.Pow(Math.E, input.values[i]);
                float value = pow / sum;
                input.values[i] = value;
            }
        }

    }
}
