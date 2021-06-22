using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    struct SingleDimension {
        public float[] values;
    }

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

    struct SquareLayer {
        public Square[] squares;
    }

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

            for (int i = 0; i < filterLayers[1].squares.Length; i++) {

                filterLayers[1].squares[i].width = adjustNet.filterLayers[1].squares[i].width;//Set widths equal so that it copies 0 values on skipped filters.

                for (int y=0; y<adjustNet.filterLayers[1].squares[i].width; y++) {
                    for (int x = 0; x < adjustNet.filterLayers[1].squares[i].width; x++) {
                        filterLayers[1].squares[i].values[x, y] -= adjustNet.filterLayers[1].squares[i].values[x, y];                        
                    }
                }
            }

            for (int i = 0; i < convolutedLayers.Length; i++) {
                for (int j = 0; j < convolutedLayers[i].squares.Length; j++) {
                    for (int y = 0; y < convolutedLayers[i].squares[j].width; y++) {
                        for (int x = 0; x < convolutedLayers[i].squares[j].width; x++) {
                            convolutedLayers[i].squares[j].biases[x, y] -= adjustNet.convolutedLayers[i].squares[j].biases[x, y];
                        }
                    }
                }
            }

        }
    }
}
