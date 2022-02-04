using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    /// <summary>
    /// Struct representing a single layer of values in the neural network.
    /// </summary>
    struct SingleDimension {
        public float[] values;
    }

    /// <summary>
    /// Struct represending a square matrix in the neural network. Either an image or filter.
    /// </summary>
    struct Square{
        public int width;
        public float[,] values;
        public float[,] biases;
        public void applyBiases() {
            for (int y = 0; y < values.GetLength(0); y++) {
                for (int x = 0; x < values.GetLength(0); x++){
                    values[x, y] += biases[x, y];
                }
            }
        }
    };

    /// <summary>
    /// Struct representing a single layer of Square objects in the network.
    /// </summary>
    struct SquareLayer {
        public Square[] squares;
    }

    /// <summary>
    /// Class representing a convolutional neural network and all the values within.
    /// </summary>
    class CNN
    {
        public Square input;

        public SquareLayer[] filterLayers;
        public SquareLayer[] convolutedLayers;
        public SquareLayer[] activatedConvolutedLayers;
        public SquareLayer[] downsampledLayers;

        public SingleDimension[] weights;
        public SingleDimension[] biases;
        public SingleDimension[] hiddenNeurons;
        public SingleDimension[] activatedHiddenNeurons;

        public SingleDimension outputs;
        public SingleDimension activatedOutputs;

        public int numberGuess = -1;

        public CNN() {
        }

        /// <summary>
        /// Subtract from weights an biases by the amount stored in the passed CNN object at their coresponding positions.
        /// </summary>
        /// <param name="adjustNet"></param>
        public void adjustWeights(CNN adjustNet) {
            for (int i=0; i<weights[0].values.Length; i++) {
                weights[0].values[i] -= adjustNet.weights[0].values[i];
            }
            for (int i = 0; i < weights[1].values.Length; i++)
            {
                weights[1].values[i] -= adjustNet.weights[1].values[i];
            }
            for (int i = 0; i < weights[2].values.Length; i++)
            {
                weights[2].values[i] -= adjustNet.weights[2].values[i];
            }

            for (int i = 0; i < biases[0].values.Length; i++)
            {
                biases[0].values[i] -= adjustNet.biases[0].values[i];
            }
            for (int i = 0; i < biases[1].values.Length; i++)
            {
                biases[1].values[i] -= adjustNet.biases[1].values[i];
            }

            for (int i = 0; i < filterLayers[0].squares.Length; i++)
            {

                filterLayers[0].squares[i].width = adjustNet.filterLayers[0].squares[i].width;//Set widths equal so that it copies 0 values on skipped filters.

                for (int y = 0; y < adjustNet.filterLayers[0].squares[i].width; y++)
                {
                    for (int x = 0; x < adjustNet.filterLayers[0].squares[i].width; x++)
                    {
                        filterLayers[0].squares[i].values[x, y] -= adjustNet.filterLayers[0].squares[i].values[x, y];
                    }
                }
            }

            for (int i = 0; i < filterLayers[1].squares.Length; i++) {

                filterLayers[1].squares[i].width = adjustNet.filterLayers[1].squares[i].width;//Set widths equal so that it copies 0 values on skipped filters.

                for (int y=0; y<adjustNet.filterLayers[1].squares[i].width; y++) {
                    for (int x = 0; x < adjustNet.filterLayers[1].squares[i].width; x++) {
                        filterLayers[1].squares[i].values[x, y] -= adjustNet.filterLayers[1].squares[i].values[x, y];                        
                    }
                }
            }

                
            for (int j = 0; j < convolutedLayers[0].squares.Length; j++) {                  
                for (int y = 0; y < convolutedLayers[0].squares[0].width; y++) {                    
                    for (int x = 0; x < convolutedLayers[0].squares[0].width; x++) {            
                        convolutedLayers[0].squares[j].biases[x, y] -= adjustNet.convolutedLayers[0].squares[j].biases[x, y];          
                    }        
                }       
            }
            for (int j = 0; j < convolutedLayers[1].squares.Length; j++)
            {
                for (int y = 0; y < convolutedLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < convolutedLayers[1].squares[0].width; x++)
                    {
                        convolutedLayers[1].squares[j].biases[x, y] -= adjustNet.convolutedLayers[1].squares[j].biases[x, y];
                    }
                }
            }
        }


    }
}
