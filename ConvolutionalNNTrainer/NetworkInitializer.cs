using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    class NetworkInitializer
    {
        static Random rnd;

        static int firstFilterLayerCount = 6;
        static int firstFilterWidth = 5;

        static int firstConvolutionWidth = 28;

        static int firstDownsampleWidth = 14;

        //static int secondFilterCount = 59;        
        static int secondFilterWidth = 5;
        static int secondConvolutionCount = 16;
        static int secondFilterCount = firstFilterLayerCount * secondConvolutionCount;

        static int secondConvolutionWidth = 10;

        static int secondDownsampleWidth = 5;

        static int hiddenLayerCount = 2;
        static int firstHiddenLayerNeuronCount = 120;
        static int secondHiddenLayerNeuronCount = 100;

        static int firstWeightCount = secondDownsampleWidth * secondDownsampleWidth * //amount of pixels per downsampled image
                                    secondConvolutionCount * //amount of downsampled images
                                    firstHiddenLayerNeuronCount; //amount of neurons in next layer

        static int secondWeightCount = firstHiddenLayerNeuronCount * secondHiddenLayerNeuronCount;//amount of neurons in first layer multiplied by amount in second layer

        static int outputNeuronCount = 10;

        static int thirdWeightCount = secondHiddenLayerNeuronCount * outputNeuronCount;//amount of neurons in last layer multiplied by amount of output neurons

        public static void setInputs(ref CNN network, Bitmap input) {
            network.input.width = input.Width;
            network.input.values = new float[input.Width, input.Width];

            for (int y=0; y<network.input.width; y++) {
                for (int x=0; x<network.input.width; x++) {
                    Color color = input.GetPixel(x, y);
                    float gray = (color.R + color.G + color.B) / 3;//average all three colour parameters to get greyscale

                    network.input.values[x, y] = (gray/255)*2-1;//convert to percentage of full colour so that it is value between 1 and 0
                }
            }
        }

        public static void InitializeNetwork(ref CNN network){
            network.filterLayers = new SquareLayer[2];
            network.convolutedLayers = new SquareLayer[2];
            network.activatedConvolutedLayers = new SquareLayer[2];

            //First filters
            network.filterLayers[0] = new SquareLayer();
            network.filterLayers[0].squares = new Square[firstFilterLayerCount];//Amount of filters
            for (int i = 0; i < firstFilterLayerCount; i++)
            {
                network.filterLayers[0].squares[i] = new Square();
                network.filterLayers[0].squares[i].width = firstFilterWidth;//width of square filter
                network.filterLayers[0].squares[i].values = new float[firstFilterWidth, firstFilterWidth];
            }

            //convolved post filter
            network.convolutedLayers[0] = new SquareLayer();
            network.convolutedLayers[0].squares = new Square[firstFilterLayerCount];//amount of convolved images
            network.activatedConvolutedLayers[0] = new SquareLayer();
            network.activatedConvolutedLayers[0].squares = new Square[firstFilterLayerCount];//amount of convolved images
            for (int i = 0; i < firstFilterLayerCount; i++)
            {
                network.convolutedLayers[0].squares[i] = new Square();
                network.convolutedLayers[0].squares[i].width = firstConvolutionWidth;//width of square convolved images
                network.convolutedLayers[0].squares[i].values = new float[firstConvolutionWidth, firstConvolutionWidth];
                network.convolutedLayers[0].squares[i].biases = new float[firstConvolutionWidth, firstConvolutionWidth];

                network.activatedConvolutedLayers[0].squares[i] = new Square();
                network.activatedConvolutedLayers[0].squares[i].width = firstConvolutionWidth;//width of square convolved images
                network.activatedConvolutedLayers[0].squares[i].values = new float[firstConvolutionWidth, firstConvolutionWidth];
                network.activatedConvolutedLayers[0].squares[i].biases = new float[firstConvolutionWidth, firstConvolutionWidth];
            }

            //first downsample
            network.downsampledLayers = new SquareLayer[2];
            network.downsampledLayers[0] = new SquareLayer();
            network.downsampledLayers[0].squares = new Square[firstFilterLayerCount];//amount of downsampled images
            for (int i = 0; i < firstFilterLayerCount; i++) {
                network.downsampledLayers[0].squares[i] = new Square();
                network.downsampledLayers[0].squares[i].width = firstDownsampleWidth;//width of square downsampled images
                network.downsampledLayers[0].squares[i].values = new float[firstDownsampleWidth, firstDownsampleWidth];
            }

            //second filters
            network.filterLayers[1] = new SquareLayer();
            network.filterLayers[1].squares = new Square[secondFilterCount];//amount of filters
            for (int i = 0; i < secondFilterCount; i++)
            {                
                network.filterLayers[1].squares[i] = new Square();
                if (isSkippedFilter(i))
                {
                    network.filterLayers[1].squares[i].width=0;
                    continue;
                }
                network.filterLayers[1].squares[i].width = secondFilterWidth;//width of each square filter in layer
                network.filterLayers[1].squares[i].values = new float[secondFilterWidth, secondFilterWidth];
            }

            //second convolved post filter
            network.convolutedLayers[1] = new SquareLayer();
            network.convolutedLayers[1].squares = new Square[secondConvolutionCount];//Amount of images after convolution
            network.activatedConvolutedLayers[1] = new SquareLayer();
            network.activatedConvolutedLayers[1].squares = new Square[secondConvolutionCount];//Amount of images after convolution
            for (int i = 0; i < secondConvolutionCount; i++)
            {
                network.convolutedLayers[1].squares[i] = new Square();
                network.convolutedLayers[1].squares[i].width = secondConvolutionWidth;//width of images after convolution
                network.convolutedLayers[1].squares[i].values = new float[secondConvolutionWidth, secondConvolutionWidth];
                network.convolutedLayers[1].squares[i].biases = new float[secondConvolutionWidth, secondConvolutionWidth];

                network.activatedConvolutedLayers[1].squares[i] = new Square();
                network.activatedConvolutedLayers[1].squares[i].width = secondConvolutionWidth;//width of images after convolution
                network.activatedConvolutedLayers[1].squares[i].values = new float[secondConvolutionWidth, secondConvolutionWidth];
                network.activatedConvolutedLayers[1].squares[i].biases = new float[secondConvolutionWidth, secondConvolutionWidth];
            }

            //second downsample
            network.downsampledLayers[1] = new SquareLayer();
            network.downsampledLayers[1].squares = new Square[secondConvolutionCount];//amount of downsampled images
            for (int i = 0; i < secondConvolutionCount; i++)
            {
                network.downsampledLayers[1].squares[i] = new Square();
                network.downsampledLayers[1].squares[i].width = secondDownsampleWidth;//width of square downsampled images
                network.downsampledLayers[1].squares[i].values = new float[secondDownsampleWidth, secondDownsampleWidth];
            }

            //first fully connected
            network.hiddenNeurons = new SingleDimension[hiddenLayerCount];
            network.activatedHiddenNeurons = new SingleDimension[hiddenLayerCount];
            network.biases = new SingleDimension[hiddenLayerCount+1];
            network.biases[0].values = new float[firstHiddenLayerNeuronCount];
            network.hiddenNeurons[0].values = new float[firstHiddenLayerNeuronCount];
            network.activatedHiddenNeurons[0].values = new float[firstHiddenLayerNeuronCount];
            network.weights = new SingleDimension[hiddenLayerCount+1];
            network.weights[0].values = new float[firstWeightCount];


            //second fully connected
            network.biases[1].values = new float[secondHiddenLayerNeuronCount];
            network.hiddenNeurons[1].values = new float[secondHiddenLayerNeuronCount];
            network.activatedHiddenNeurons[1].values = new float[secondHiddenLayerNeuronCount];
            network.weights[1].values = new float[secondWeightCount];

            //output layer and weights
            network.outputs.values = new float[outputNeuronCount];
            network.softMaxOutputs.values = new float[outputNeuronCount];
            network.weights[2].values = new float[thirdWeightCount];            
        }

        static public void randomizeWeights(ref CNN network) {
            rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < firstFilterLayerCount; i++)
            {
                for (int y=0; y<firstFilterWidth; y++) {
                    for (int x=0; x<firstFilterWidth; x++) {
                        network.filterLayers[0].squares[i].values[x, y] = NextGaussian();                        
                    }
                }

                for (int y = 0; y < firstConvolutionWidth; y++)
                {
                    for (int x = 0; x < firstConvolutionWidth; x++)
                    {
                        network.convolutedLayers[0].squares[i].biases[x, y] = 0.0f;
                    }
                }   
            }

            for (int i = 0; i < secondFilterCount; i++) {
                for (int y = 0; y < network.filterLayers[1].squares[i].width; y++)
                {
                    for (int x = 0; x < network.filterLayers[1].squares[i].width; x++)
                    {
                        network.filterLayers[1].squares[i].values[x, y] = NextGaussian();
                    }
                }
            }

            for (int i = 0; i < secondConvolutionCount; i++)
            {
                for (int y = 0; y < secondConvolutionWidth; y++)
                {
                    for (int x = 0; x < secondConvolutionWidth; x++)
                    {
                        network.convolutedLayers[1].squares[i].biases[x, y] = 0.0f;
                    }
                }
            }

            for (int i=0; i<firstWeightCount; i++) {
                network.weights[0].values[i] = NextGaussian();
            }
            for (int i = 0; i < firstHiddenLayerNeuronCount; i++) {
                network.biases[0].values[i] = 0.0f;
            }
            for (int i = 0; i < secondWeightCount; i++)
            {
                network.weights[1].values[i] = NextGaussian();
            }
            for (int i = 0; i < secondHiddenLayerNeuronCount; i++)
            {
                network.biases[1].values[i] = 0.0f;
            }
            for (int i = 0; i < thirdWeightCount; i++)
            {
                network.weights[2].values[i] = NextGaussian();
            }
        }

        /*
         
         */
        public static float NextGaussian()
        {
            float v1, v2, s;
            do
            {
                v1 = 2.0f * (float)rnd.NextDouble() - 1.0f;
                v2 = 2.0f * (float)rnd.NextDouble() - 1.0f;
                s = v1 * v1 + v2 * v2;
            } while (s >= 1.0f || s == 0f);

            s = (float)Math.Sqrt((-2.0f * Math.Log(s)) / s);

            return v1 * s;
        }

        static public bool isSkippedFilter(int pos) {
            switch (pos) {
                case 7:
                    return true;
                case 10:
                    return true;
                case 14:
                    return true;
                case 17:
                    return true;
                case 18:
                    return true;
                case 21:
                    return true;
                case 25:
                    return true;
                case 26:
                    return true;
                case 32:
                    return true;
                case 33:
                    return true;
                case 39:
                    return true;
                case 40:
                    return true;
                case 46:
                    return true;
                case 47:
                    return true;
                case 48:
                    return true;
                case 53:
                    return true;
                case 54:
                    return true;
                case 55:
                    return true;
                case 61:
                    return true;
                case 62:
                    return true;
                case 63:
                    return true;
                case 68:
                    return true;
                case 69:
                    return true;
                case 70:
                    return true;
                case 75:
                    return true;
                case 76:
                    return true;
                case 77:
                    return true;
                case 78:
                    return true;
                case 82:
                    return true;
                case 83:
                    return true;
                case 84:
                    return true;
                case 85:
                    return true;
                case 89:
                    return true;
                case 90:
                    return true;
                case 91:
                    return true;
                case 92:
                    return true;
            }
            return false;
        }
    }
}
