using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    /// <summary>
    /// Class for initializing CNN values
    /// </summary>
    class NetworkInitializer
    {
        private static Random rnd;

        private readonly static int firstFilterLayerCount = 6;
        private readonly static int firstFilterWidth = 5;

        private readonly static int firstConvolutionWidth = 28;

        private readonly static int firstDownsampleWidth = 14;


        private readonly static int secondFilterWidth = 5;
        private readonly static int secondConvolutionCount = 16;
        private readonly static int secondFilterCount = firstFilterLayerCount * secondConvolutionCount;

        private readonly static int secondConvolutionWidth = 10;

        private readonly static int secondDownsampleWidth = 5;

        private readonly static int hiddenLayerCount = 2;
        private readonly static int firstHiddenLayerNeuronCount = 240;//120;
        private readonly static int secondHiddenLayerNeuronCount = 200;//100;

        private readonly static int firstWeightCount = secondDownsampleWidth * secondDownsampleWidth * //amount of pixels per downsampled image
                                    secondConvolutionCount * //amount of downsampled images
                                    firstHiddenLayerNeuronCount; //amount of neurons in next layer

        private readonly static int secondWeightCount = firstHiddenLayerNeuronCount * secondHiddenLayerNeuronCount;//amount of neurons in first layer multiplied by amount in second layer

        private readonly static int outputNeuronCount = 11;

        private readonly static int thirdWeightCount = secondHiddenLayerNeuronCount * outputNeuronCount;//amount of neurons in last layer multiplied by amount of output neurons

        /// <summary>
        /// Set the input image (Bitmap) in the CNN object.
        /// </summary>
        /// <param name="network">Convolutional neural network object to be set</param>
        /// <param name="input">Input image</param>
        public static void setInputs(ref CNN network, Bitmap input) {
            network.input.width = input.Width;
            network.input.values = new float[input.Width, input.Width];

            for (int y=0; y<network.input.width; y++) {
                for (int x=0; x<network.input.width; x++) {
                    Color color = input.GetPixel(x, y);
                    float gray = (color.R + color.G + color.B) / 3;//average all three colour parameters to get greyscale

                    network.input.values[x, y] = gray > 0 ? 1.0f : 0.0f;//(gray/255)*2-1;//convert to percentage of full colour so that it is value between 1 and 0
                }
            }
        }

        /// <summary>
        /// Set the input image as a byte array in the CNN object.
        /// </summary>
        /// <param name="network">Convolutional neural network object to be set</param>
        /// <param name="input">Input bytes</param>
        public static void setInputs(ref CNN network, byte[,] input)
        {
            network.input.width = 32;
            network.input.values = new float[32,32]; ;

            for (int y = 0; y < network.input.width; y++)
            {
                for (int x = 0; x < network.input.width; x++)
                {
                    if (x < 2 || y < 2 || x >= 30 || y >= 30) {
                        network.input.values[x, y] = -1;
                        continue;
                    }
                    float num = input[x - 2, y - 2];
                    num = num / 255;
                    network.input.values[x, y] = num * 2 - 1;//convert to percentage of full colour so that it is value between 1 and 0
                }
            }
        }

        /// <summary>
        /// Initializes the structure of the neural network object according to the parameters in this class.
        /// </summary>
        /// <param name="network">Network to be initialized</param>
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
            network.biases[2].values = new float[outputNeuronCount];
            network.outputs.values = new float[outputNeuronCount];
            network.activatedOutputs.values = new float[outputNeuronCount];
            network.weights[2].values = new float[thirdWeightCount];            
        }

        /// <summary>
        /// Resets the values that are stored in the network object from the previous training 
        /// </summary>
        /// <param name="network">Network object to be reset.</param>
        public static void resetNetworkNeurons(ref CNN network)
        {
            //convolved post filter
            for (int i = 0; i < firstFilterLayerCount; i++)
            {
                network.convolutedLayers[0].squares[i].values = new float[firstConvolutionWidth, firstConvolutionWidth];

                network.activatedConvolutedLayers[0].squares[i].values = new float[firstConvolutionWidth, firstConvolutionWidth];
            }

            //first downsample
            for (int i = 0; i < firstFilterLayerCount; i++)
            {
                network.downsampledLayers[0].squares[i].values = new float[firstDownsampleWidth, firstDownsampleWidth];
            }

            //second convolved post filter
            for (int i = 0; i < secondConvolutionCount; i++)
            {
                network.convolutedLayers[1].squares[i].values = new float[secondConvolutionWidth, secondConvolutionWidth];

                network.activatedConvolutedLayers[1].squares[i].values = new float[secondConvolutionWidth, secondConvolutionWidth];
            }

            //second downsample
            for (int i = 0; i < secondConvolutionCount; i++)
            {
                network.downsampledLayers[1].squares[i].values = new float[secondDownsampleWidth, secondDownsampleWidth];
            }

            //first fully connected
            network.hiddenNeurons = new SingleDimension[hiddenLayerCount];
            network.activatedHiddenNeurons = new SingleDimension[hiddenLayerCount];
            network.hiddenNeurons[0].values = new float[firstHiddenLayerNeuronCount];
            network.activatedHiddenNeurons[0].values = new float[firstHiddenLayerNeuronCount];


            //second fully connected
            network.hiddenNeurons[1].values = new float[secondHiddenLayerNeuronCount];
            network.activatedHiddenNeurons[1].values = new float[secondHiddenLayerNeuronCount];

            //output layer and weights
            network.outputs.values = new float[outputNeuronCount];
            network.activatedOutputs.values = new float[outputNeuronCount];
        }

        /// <summary>
        /// Randomizes the weights and biases for their initial state prior to training.
        /// </summary>
        /// <param name="network">Netowrk to be randomized</param>
        static public void randomizeWeights(ref CNN network) {
            rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < firstFilterLayerCount; i++)
            {
                for (int y=0; y<firstFilterWidth; y++) {
                    for (int x=0; x<firstFilterWidth; x++) {
                        network.filterLayers[0].squares[i].values[x, y] = randValue();                        
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
                        network.filterLayers[1].squares[i].values[x, y] = randValue();
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
                network.weights[0].values[i] = randValue();
            }
            for (int i = 0; i < firstHiddenLayerNeuronCount; i++) {
                network.biases[0].values[i] = 0.0f;
            }
            for (int i = 0; i < secondWeightCount; i++)
            {
                network.weights[1].values[i] = randValue();
            }
            for (int i = 0; i < secondHiddenLayerNeuronCount; i++)
            {
                network.biases[1].values[i] = 0.0f;
            }
            for (int i = 0; i < thirdWeightCount; i++)
            {
                network.weights[2].values[i] = randValue();
            }
        }

        /// <summary>
        /// Creates a random value between -0.5 and 0.5.
        /// </summary>
        /// <returns>Random value</returns>
        public static float randValue() {
            return (float)rnd.NextDouble() - 0.5f;
        }

        /// <summary>
        /// Checks if this is a filter to be skipped.
        /// </summary>
        /// <param name="pos">Position of filter</param>
        /// <returns>Whether or not this filter is skipped</returns>
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
