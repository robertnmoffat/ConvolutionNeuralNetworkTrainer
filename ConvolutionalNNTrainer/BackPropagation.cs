using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    /// <summary>
    /// Contains the methods used in backpropagating a neural network.
    /// </summary>
    class BackPropagation
    {

        /// <summary>
        /// Backpropagates errors through a neural network.
        /// </summary>
        /// <param name="net">Network to be backpropagated.</param>
        /// <param name="errors">Network to store errors in while backpropagating.</param>
        /// <param name="errorsAvg">Network containing averaged adjustments to weights and biases.</param>
        /// <param name="batchSize">Batch size to average adjustments by.</param>
        /// <param name="correctDigit">The currently desired output of the network.</param>
        /// <param name="learningRate">Learning rate to multiply adjustments by.</param>
        /// <returns></returns>
        static public float backPropagate(in CNN net, ref CNN errors, ref CNN errorsAvg, int batchSize, int correctDigit, float learningRate) {

            float[] desiredOutput = new float[11];            
            desiredOutput[correctDigit] = 1.0f;

            //---------------------calculate total error
            float totalError = 0.0f;
            for (int i=0; i<net.activatedOutputs.values.Length; i++) {
                float y = desiredOutput[i];
                float o = net.activatedOutputs.values[i];
                totalError += (float)Math.Pow((o-y),2)/11;
            }

            //---------------------calculate error in outputs
            for (int i = 0; i < net.outputs.values.Length; i++) {
                errors.activatedOutputs.values[i] = 1;
                errors.outputs.values[i] = costFunctionDerivative(desiredOutput[i], net.activatedOutputs.values[i]);
                errors.outputs.values[i] *= (float)sigmoidDerivative(net.activatedOutputs.values[i]);
                errorsAvg.biases[2].values[i] += errors.activatedOutputs.values[i] * errors.outputs.values[i] / batchSize * learningRate;
            }

            //------------------calculate error in third layer of weights
            for (int i = 0; i < net.outputs.values.Length; i++)
            {
                for (int j = 0; j < net.hiddenNeurons[1].values.Length; j++)
                {
                    //derivative of current weight * softmax derivative * cross entropy derivative
                    errors.weights[2].values[j * net.outputs.values.Length + i] = net.hiddenNeurons[1].values[j] * errors.outputs.values[i] * errors.activatedOutputs.values[i];
                    errorsAvg.weights[2].values[j * net.outputs.values.Length + i] += errors.weights[2].values[j * net.outputs.values.Length + i] / batchSize * learningRate;//store of average weights                    
                }
            }

            //-----------------error in second hidden layer
            for (int i = 0; i < net.hiddenNeurons[1].values.Length; i++)
            {
                for (int j = 0; j < net.outputs.values.Length; j++)
                {
                    errors.activatedHiddenNeurons[1].values[i] += net.weights[2].values[i * net.outputs.values.Length + j] * errors.outputs.values[j] * errors.activatedOutputs.values[j];
                }
                errors.hiddenNeurons[1].values[i] = (float)sigmoidDerivative(net.hiddenNeurons[1].values[i])* errors.activatedHiddenNeurons[1].values[i];
                errorsAvg.biases[1].values[i] += errors.hiddenNeurons[1].values[i] / batchSize * learningRate;
                //--------CHANGED TO SIG DERIV OF NET.HIDDEN FROM ERROR.ACTIV, THEN MULT BY PREV ERROR. CHANGED BIAS TO SUM RATHER THAN NEG.
            }

            //------------------error in second layer of weights
            for (int j = 0; j < net.hiddenNeurons[0].values.Length; j++) {
                for (int i = 0; i < net.hiddenNeurons[1].values.Length; i++)
                {
                    errors.weights[1].values[j * net.hiddenNeurons[1].values.Length + i] = net.hiddenNeurons[0].values[j] * errors.hiddenNeurons[1].values[i];
                    errorsAvg.weights[1].values[j * net.hiddenNeurons[1].values.Length + i] += errors.weights[1].values[j * net.hiddenNeurons[1].values.Length + i] / batchSize * learningRate;
                }
            }

            //------------------error in first hidden layer
            for (int i = 0; i < net.hiddenNeurons[0].values.Length; i++)
            {
                for (int j = 0; j < net.hiddenNeurons[1].values.Length; j++)
                {
                    errors.activatedHiddenNeurons[0].values[i] += net.weights[1].values[i * net.outputs.values.Length + j] * errors.hiddenNeurons[1].values[j];
                }
                errors.hiddenNeurons[0].values[i] = (float)sigmoidDerivative(net.hiddenNeurons[0].values[i])* errors.activatedHiddenNeurons[0].values[i];
                errorsAvg.biases[0].values[i] += errors.hiddenNeurons[0].values[i] / batchSize * learningRate;
            }

            //------------------error in the first layer of weights
            int neuronCount = net.hiddenNeurons[0].values.Length;
            int sampleCount = net.downsampledLayers[1].squares.Length;
            int sampleHeight = net.downsampledLayers[1].squares[0].width;
            int sampleWidth = net.downsampledLayers[1].squares[0].width;

            for (int j = 0; j < sampleCount; j++)
            {
                //iterate through square
                for (int y = 0; y < sampleHeight; y++)
                {
                    for (int x = 0; x < sampleWidth; x++)
                    {
                        for (int i = 0; i < neuronCount; i++) {
                            errors.weights[0].values[i * sampleCount * sampleHeight * sampleWidth + j * sampleHeight * sampleWidth + y * sampleWidth + x] = errors.hiddenNeurons[0].values[i] * net.downsampledLayers[1].squares[j].values[x,y];
                            errorsAvg.weights[0].values[i * sampleCount * sampleHeight * sampleWidth + j * sampleHeight * sampleWidth + y * sampleWidth + x] += errors.weights[0].values[i * sampleCount * sampleHeight * sampleWidth + j * sampleHeight * sampleWidth + y * sampleWidth + x] / batchSize * learningRate;
                        }
                    }
                }
            }
            
            
            //---------------error in last downsample layer
            for (int j = 0; j < sampleCount; j++)
            {
                //iterate through square
                for (int y = 0; y < sampleHeight; y++)
                {
                    for (int x = 0; x < sampleWidth; x++)
                    {
                        for (int i = 0; i < neuronCount; i++)
                        {
                            errors.downsampledLayers[1].squares[j].values[x, y] += net.weights[0].values[i * sampleCount * sampleHeight * sampleWidth + j * sampleHeight * sampleWidth + y * sampleWidth + x] * errors.hiddenNeurons[0].values[i];                            
                        }
                        //iterate over possible positions of downsample source
                        for (int sy = 0; sy < 2; sy++)
                        {
                            for (int sx = 0; sx < 2; sx++)
                            {
                                float sourceValue = net.activatedConvolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy];//Potential source position for downsample
                                float sampleValue = net.downsampledLayers[1].squares[j].values[x, y];//Downsample value
                                if (sourceValue == sampleValue)//source position found
                                { 
                                    errors.activatedConvolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] = errors.downsampledLayers[1].squares[j].values[x, y];
                                    //if the values before and after activation are not equal then leaky relu was applied (* 0.1)
                                    if (net.convolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] != net.activatedConvolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy])
                                    {
                                        errors.convolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] = errors.activatedConvolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] * 0.0f;
                                    }
                                    else {//else it was positive and relu simply multiplied by 1
                                        if (net.convolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] != 0.0f)
                                            errors.convolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] = errors.activatedConvolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy];
                                        else
                                            errors.convolutedLayers[1].squares[j].values[x * 2 + sx, y * 2 + sy] = 0;
                                    }
                                    sx = 2;
                                    sy = 2;
                                }
                            }
                        }
                    }
                }
            }

            //---------------error in second filter convolution derivative

            sampleCount = net.downsampledLayers[0].squares.Length;
            int convolutionCount = errors.convolutedLayers[1].squares.Length;
            Square filterErrorWeights = new Square();
            int filterWidth = net.filterLayers[1].squares[0].width;            
            filterErrorWeights.width = filterWidth;
            int filterPos = 0;

            for (int i = 0; i < convolutionCount; i++)
            {
                for (int j = 0; j < sampleCount; j++)
                {
                    if (net.filterLayers[1].squares[filterPos].width != 0)//If width equals zero, there is no filter in this position.
                    {
                        applyFilterAdditivelyWithRates(net.downsampledLayers[0].squares[j], ref errors.convolutedLayers[1].squares[i], ref errorsAvg.filterLayers[1].squares[filterPos], batchSize, learningRate, ref errorsAvg.convolutedLayers[1].squares[i]);
                        Square mirroredFilter = mirrorFilter(in net.filterLayers[1].squares[filterPos]);//Flip filter to be applied to error to generate previous layer error.
                        //-----Error in first downsample layer-----
                        applyFilterAdditivelyWithPadding(errors.convolutedLayers[1].squares[i], mirroredFilter, ref errors.downsampledLayers[0].squares[j]);
                    }
                    else {
                        errorsAvg.filterLayers[1].squares[filterPos].width = 0;//Set error filter width to 0 so that it is known not to use this filter.
                    }
                    filterPos++;
                }
            }

            //---------------Error from downsample to first convolution
            sampleCount = net.downsampledLayers[0].squares.Length;
            sampleHeight = net.downsampledLayers[0].squares[0].width;
            sampleWidth = sampleHeight;

            for (int j = 0; j < sampleCount; j++)
            {
                //iterate through square
                for (int y = 0; y < sampleHeight; y++)
                {
                    for (int x = 0; x < sampleWidth; x++)
                    {
                        //iterate over possible positions of downsample source
                        for (int sy = 0; sy < 2; sy++)
                        {
                            for (int sx = 0; sx < 2; sx++)
                            {
                                float sourceValue = net.activatedConvolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy];//Potential source position for downsample
                                float sampleValue = net.downsampledLayers[0].squares[j].values[x, y];//Downsample value
                                if (sourceValue == sampleValue)//source position found
                                {
                                    errors.activatedConvolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] = errors.downsampledLayers[0].squares[j].values[x, y];
                                    //if the values before and after activation are not equal then leaky relu was applied (* 0.1)
                                    if (net.convolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] != net.activatedConvolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy])
                                    {
                                        errors.convolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] = errors.activatedConvolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] * 0.0f;
                                    }
                                    else
                                    {//else it was positive and relu simply multiplied by 1
                                        if (net.convolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] != 0.0f)
                                            errors.convolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] = errors.activatedConvolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy];
                                        else
                                            errors.convolutedLayers[0].squares[j].values[x * 2 + sx, y * 2 + sy] = 0;
                                    }
                                    sx = 2;
                                    sy = 2;//set to skip checking rest of area.
                                }
                            }
                        }
                    }
                }
            }


            //--------------Error in first layer of convolutional filter weights
            convolutionCount = net.convolutedLayers[0].squares.Length;
            for (int i = 0; i < convolutionCount; i++)
            {  
                applyFilterAdditivelyWithRates(net.input, ref errors.convolutedLayers[0].squares[i], ref errorsAvg.filterLayers[0].squares[i], batchSize, learningRate, ref errorsAvg.convolutedLayers[0].squares[i]);

            }



            return totalError;
        }

        /// <summary>
        /// Returns the derivative of the cost function.
        /// </summary>
        /// <param name="desiredValue">What the output should be.</param>
        /// <param name="output">What the output was.</param>
        /// <returns>The value of the derivative of the cost function.</returns>
        static private float costFunctionDerivative(float desiredValue, float output) {
            return 2.0f * 1.0f / 11.0f * (output - desiredValue);
        }

        /// <summary>
        /// Returns the derivative of the sigmoid function.
        /// </summary>
        /// <param name="x">Activation value after sigmoid was applied during forward propagation.</param>
        /// <returns>The derivative value of the sigmoid function.</returns>
        static private double sigmoidDerivative(float x) {
            return ForwardPropagation.sigmoid(x) * (1 - ForwardPropagation.sigmoid(x));
        }

        /// <summary>
        /// Applies a filter through convolution
        /// </summary>
        /// <param name="input">Input Square to have the filter applied to.</param>
        /// <param name="filter">Filter Square to be applied to the input.</param>
        /// <param name="convolution">Square containing the values post convolution.</param>
        /// <param name="batchSize">Amount of rounds in the training batch to average the values by.</param>
        /// <param name="learningRate">Learning rate to be multiplied to the values.</param>
        /// <param name="biases">Biases to be applied to the filter.</param>
        static private void applyFilterAdditivelyWithRates(Square input, ref Square filter, ref Square convolution, int batchSize, float learningRate, ref Square biases)
        {
            float positionSum = 0.0f;
            for (int posy = 0; posy < convolution.width; posy++)
            {
                for (int posx = 0; posx < convolution.width; posx++)
                {
                    positionSum = 0.0f;

                    for (int y = 0; y < filter.width; y++)
                    {
                        for (int x = 0; x < filter.width; x++)
                        {
                            positionSum += input.values[posx + x, posy + y] * filter.values[x, y]/batchSize*learningRate;
                        }
                    }
                    convolution.values[posx, posy] += positionSum;
                    biases.biases[posx, posy] = filter.values[posx, posy];
                }
            }
        }

        /// <summary>
        /// Applies a filter through convolution with one space of padding.
        /// </summary>
        /// <param name="input">Input Square to have the filter applied to.</param>
        /// <param name="filter">Filter Square to be applied to the input.</param>
        /// <param name="convolution">Square containing the values post convolution.</param>
        static private void applyFilterAdditivelyWithPadding(in Square input, in Square filter, ref Square convolution)
        {
            float positionSum = 0.0f;
            for (int posy = 0; posy < convolution.width; posy++)
            {
                for (int posx = 0; posx < convolution.width; posx++)
                {
                    positionSum = 0.0f;

                    for (int y = 0; y < filter.width; y++)
                    {
                        for (int x = 0; x < filter.width; x++)
                        {
                            if (posx + x-(filter.width-1) < 0 || posx + x >= convolution.width)
                                continue;
                            if (posy + y- (filter.width - 1) < 0 || posy + y >= convolution.width)
                                continue;
                            positionSum += input.values[posx + x- (filter.width - 1), posy + y- (filter.width - 1)] * filter.values[x, y];
                        }
                    }
                    convolution.values[posx, posy] += positionSum;
                }
            }
        }


        /// <summary>
        /// Mirrors filter horizontally and vertically.
        /// </summary>
        /// <param name="filter">Filter to be mirrored</param>
        /// <returns>The mirrored filter.</returns>
        private static Square mirrorFilter(in Square filter) {
            Square mirrored = new Square();
            mirrored.width = filter.width;
            mirrored.values = new float[mirrored.width, mirrored.width];
            for (int y=0; y<filter.width; y++) {
                for (int x=0; x<filter.width; x++) {
                    mirrored.values[filter.width - x-1, filter.width - y-1] = filter.values[x, y];
                }
            }
            return mirrored;
        }
    }
}
