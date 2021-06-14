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
            //first convolution layer
            for (int i = 0; i<net.filterLayers[0].squares.Length; i++)
            {
                applyFilter(net.input, net.filterLayers[0].squares[i], ref net.convolutedLayers[0].squares[i]);
                maxPool(net.convolutedLayers[0].squares[i], ref net.downsampledLayers[0].squares[i]);
            }

            //second convolution
            int filterPos = 0;
            for (int c = 0; c < net.convolutedLayers[1].squares.Length; c++)
            {
                for (int f = 0; f < 6; f++)
                {
                    applyFilterAdditively(net.downsampledLayers[0].squares[filterPos % 6], net.filterLayers[1].squares[filterPos], ref net.convolutedLayers[1].squares[c]);
                    filterPos++;
                }
                reluSquare(ref net.convolutedLayers[1].squares[c]);//apply relu to square
                maxPool(net.convolutedLayers[1].squares[c], ref net.downsampledLayers[1].squares[c]);
            }

            //fully connected
            int weightPos = 0;
            for (int i=0; i<net.hiddenNeurons[0].values.Length; i++) {
                for (int j = 0; j < net.downsampledLayers[1].squares.Length; j++) {
                    //iterate through square
                    for (int y = 0; y < net.downsampledLayers[1].squares[0].width; y++) {
                        for (int x = 0; x < net.downsampledLayers[1].squares[0].width; x++)
                        {
                            net.hiddenNeurons[0].values[i] += net.weights[0].values[weightPos++] * net.downsampledLayers[1].squares[j].values[x,y];
                        }                        
                    }                    
                }
                net.activatedHiddenNeurons[0].values[i] = sigmoid(net.hiddenNeurons[0].values[i]);//after adding all values for current neuron, run it through sigmoid
            }

            //fully connected layer 2
            weightPos = 0;
            for (int i = 0; i < net.hiddenNeurons[1].values.Length; i++)
            {
                for (int j = 0; j < net.hiddenNeurons[0].values.Length; j++)
                {
                    net.hiddenNeurons[1].values[i] += net.weights[1].values[weightPos++] * net.activatedHiddenNeurons[0].values[j];
                }
                net.activatedHiddenNeurons[1].values[i] = sigmoid(net.hiddenNeurons[1].values[i]);//after adding all values for current neuron, run it through sigmoid
            }

            //fully connected output
            weightPos = 0;
            for (int i = 0; i < net.outputs.values.Length; i++) {
                for (int j = 0; j < net.hiddenNeurons[1].values.Length; j++) {
                    net.outputs.values[i] += net.weights[2].values[weightPos++] * sigmoid(net.activatedHiddenNeurons[1].values[j]);                    
                }
                net.softMaxOutputs.values[i] = net.outputs.values[i];
            }
            softMax(ref net.softMaxOutputs);
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
                    float value = (positionSum + convolution.biases[posx, posy]);
                    convolution.values[posx, posy] = value>=0.0f?value:0.01f* value;
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
                    float value = (positionSum + convolution.biases[posx, posy]);
                    convolution.values[posx, posy] += value;
                }
            }
        }

        static public float sigmoid(float num) {
            float epow = (float)Math.Pow(Math.E, num);
            return 1 / (1 + epow);
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

        static public void reluSquare(ref Square input) {
            for (int y = 0; y < input.width; y++) {
                for (int x=0; x<input.width; x++) {
                    input.values[x, y] = input.values[x, y] >= 0 ? input.values[x, y] : 0.01f* input.values[x, y];//if less than zero set to zero else keep.
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
                input.values[i] = (float)Math.Pow(Math.E, input.values[i]) / sum;
            }
        }

    }
}
