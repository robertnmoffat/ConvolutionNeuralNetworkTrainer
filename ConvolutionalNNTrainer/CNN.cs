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
        public SingleDimension softMaxOutputs;

        public CNN() {
        }
    }
}
