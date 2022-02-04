using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    /// <summary>
    /// Class containing static methods to load and save convolutional neural network.
    /// </summary>
    class FileIO
    {
        /// <summary>
        /// Converts 4 bytes to a float.
        /// </summary>
        /// <param name="bytes">Array containing the bytes to convert.</param>
        /// <param name="startPos">Position of first of 4 bytes to convert.</param>
        /// <returns></returns>
        public static float convertBytesToFloat(byte[] bytes, int startPos) {
            byte[] ar;
            ar = new[] { bytes[startPos], bytes[startPos+1], bytes[startPos+2], bytes[startPos+3] };
            float num = BitConverter.ToSingle(ar, 0);
            return num;
        }

        /// <summary>
        /// Converts a float to 4 bytes.
        /// </summary>
        /// <param name="bytes">Byte array to hold the bytes.</param>
        /// <param name="pos">Starting position in the array to add the bytes.</param>
        /// <param name="toAdd">Float to be converted.</param>
        public static void addFloatToByteArray(ref byte[] bytes, ref int pos, float toAdd)
        {
            byte[] floatHolder = new byte[4];
            floatHolder = BitConverter.GetBytes(toAdd);

            for (int i = 0; i < 4; i++)
            {
                bytes[pos++] = floatHolder[i];
            }
        }

        /// <summary>
        /// Loads a neural network file into a CNN object.
        /// </summary>
        /// <param name="path">Path to the neural net file.</param>
        /// <returns>CNN object representing the saved neural net.</returns>
        public static CNN loadNet(string path) {
            CNN network = new CNN();
            NetworkInitializer.InitializeNetwork(ref network);

            byte[] bytes = File.ReadAllBytes(path);
            int byteOffset=0;

            char version = (char)bytes[byteOffset++];

            for (int l = 0; l < network.filterLayers[0].squares.Length; l++)
            {
                for (int y = 0; y < network.filterLayers[0].squares[0].width; y++)
                {
                    for (int x = 0; x < network.filterLayers[0].squares[0].width; x++)
                    {
                        network.filterLayers[0].squares[l].values[x, y] = convertBytesToFloat(bytes, byteOffset);
                        byteOffset += 4;
                    }
                }
            }
            for (int l = 0; l < network.filterLayers[1].squares.Length; l++)
            {
                for (int y = 0; y < network.filterLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < network.filterLayers[1].squares[0].width; x++)
                    {
                        if (network.filterLayers[1].squares[l].values != null)
                            network.filterLayers[1].squares[l].values[x, y] = convertBytesToFloat(bytes, byteOffset);
                        byteOffset += 4;
                    }
                }
            }

            //----------------convoluted layers biases-------------
            for (int l = 0; l < network.convolutedLayers[0].squares.Length; l++)
            {
                for (int y = 0; y < network.convolutedLayers[0].squares[0].width; y++)
                {
                    for (int x = 0; x < network.convolutedLayers[0].squares[0].width; x++)
                    {
                        network.convolutedLayers[0].squares[l].biases[x, y] = convertBytesToFloat(bytes, byteOffset);
                        byteOffset += 4;
                    }
                }
            }
            for (int l = 0; l < network.convolutedLayers[1].squares.Length; l++)
            {
                for (int y = 0; y < network.convolutedLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < network.convolutedLayers[1].squares[0].width; x++)
                    {
                        network.convolutedLayers[1].squares[l].biases[x, y] = convertBytesToFloat(bytes, byteOffset);
                        byteOffset += 4;
                    }
                }
            }

            //---------------weight layer values----------------
            for (int i = 0; i < network.weights[0].values.Length; i++)
            {
                network.weights[0].values[i] = convertBytesToFloat(bytes, byteOffset);
                byteOffset += 4;
            }
            for (int i = 0; i < network.weights[1].values.Length; i++)
            {
                network.weights[1].values[i] = convertBytesToFloat(bytes, byteOffset);
                byteOffset += 4;
            }
            for (int i = 0; i < network.weights[2].values.Length; i++)
            {
                network.weights[2].values[i] = convertBytesToFloat(bytes, byteOffset);
                byteOffset += 4;
            }

            //-------------bias layer values------------------
            for (int i = 0; i < network.biases[0].values.Length; i++)
            {
                network.biases[0].values[i] = convertBytesToFloat(bytes, byteOffset);
                byteOffset += 4;
            }
            for (int i = 0; i < network.biases[1].values.Length; i++)
            {
                network.biases[1].values[i] = convertBytesToFloat(bytes, byteOffset);
                byteOffset += 4;
            }

            return network;
        }

        /// <summary>
        /// Save a CNN object to file.
        /// </summary>
        /// <param name="network">CNN object of network to save.</param>
        /// <param name="path">Path and filename of where to save network.</param>
        public static void saveNet(CNN network, string path)
        {

            byte version = (byte)2;

            byte[] bytes;
            int totalSize = 1;//starts at 1 for version number

            totalSize += network.filterLayers[0].squares.Length * network.filterLayers[0].squares[0].width * network.filterLayers[0].squares[0].width * 4;
            totalSize += network.filterLayers[1].squares.Length * network.filterLayers[1].squares[0].width * network.filterLayers[1].squares[0].width * 4;

            totalSize += network.convolutedLayers[0].squares.Length * network.convolutedLayers[0].squares[0].width * network.convolutedLayers[0].squares[0].width * 4;
            totalSize += network.convolutedLayers[1].squares.Length * network.convolutedLayers[1].squares[0].width * network.convolutedLayers[1].squares[0].width * 4;

            totalSize += network.weights[0].values.Length * 4;
            totalSize += network.weights[1].values.Length * 4;
            totalSize += network.weights[2].values.Length * 4;

            totalSize += network.hiddenNeurons[0].values.Length * 4;
            totalSize += network.hiddenNeurons[1].values.Length * 4;

            totalSize += network.biases[0].values.Length * 4;
            totalSize += network.biases[1].values.Length * 4;

            bytes = new byte[totalSize];
            int byteOffset = 1;

            bytes[0] = version;
            //-------------------Filter layers weights-------------
            for (int l = 0; l < network.filterLayers[0].squares.Length; l++) {
                for (int y = 0; y < network.filterLayers[0].squares[0].width; y++) {
                    for (int x = 0; x < network.filterLayers[0].squares[0].width; x++) {
                        addFloatToByteArray(ref bytes, ref byteOffset, network.filterLayers[0].squares[l].values[x, y]);
                    }
                }
            }
            for (int l = 0; l < network.filterLayers[1].squares.Length; l++)
            {
                for (int y = 0; y < network.filterLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < network.filterLayers[1].squares[0].width; x++)
                    {
                        float num;
                        if (network.filterLayers[1].squares[l].width == 0)
                            num = 0.0f;
                        else
                            num = network.filterLayers[1].squares[l].values[x, y];
                        addFloatToByteArray(ref bytes, ref byteOffset, num);
                    }
                }
            }

            //----------------convoluted layers biases-------------
            for (int l = 0; l < network.convolutedLayers[0].squares.Length; l++)
            {
                for (int y = 0; y < network.convolutedLayers[0].squares[0].width; y++)
                {
                    for (int x = 0; x < network.convolutedLayers[0].squares[0].width; x++)
                    {
                        addFloatToByteArray(ref bytes, ref byteOffset, network.convolutedLayers[0].squares[l].biases[x, y]);
                    }
                }
            }
            for (int l = 0; l < network.convolutedLayers[1].squares.Length; l++)
            {
                for (int y = 0; y < network.convolutedLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < network.convolutedLayers[1].squares[0].width; x++)
                    {
                        addFloatToByteArray(ref bytes, ref byteOffset, network.convolutedLayers[1].squares[l].biases[x, y]);
                    }
                }
            }

            //---------------weight layer values----------------
            for (int i = 0; i < network.weights[0].values.Length; i++) {
                addFloatToByteArray(ref bytes, ref byteOffset, network.weights[0].values[i]);
            }
            for (int i = 0; i < network.weights[1].values.Length; i++)
            {
                addFloatToByteArray(ref bytes, ref byteOffset, network.weights[1].values[i]);
            }
            for (int i = 0; i < network.weights[2].values.Length; i++)
            {
                addFloatToByteArray(ref bytes, ref byteOffset, network.weights[2].values[i]);
            }

            //-------------bias layer values------------------
            for (int i = 0; i < network.biases[0].values.Length; i++)
            {
                addFloatToByteArray(ref bytes, ref byteOffset, network.biases[0].values[i]);
            }
            for (int i = 0; i < network.biases[1].values.Length; i++)
            {
                addFloatToByteArray(ref bytes, ref byteOffset, network.biases[1].values[i]);
            }

            File.WriteAllBytes(path, bytes);

            CNN loadedNet = loadNet(path);
            compareNets(network, loadedNet);
        }

        /// <summary>
        /// Compares to CNN objects to ensure that they have matching parameters.
        /// </summary>
        /// <param name="net1">First network.</param>
        /// <param name="net2">Second network.</param>
        public static void compareNets(CNN net1, CNN net2)
        {
            if (net1.filterLayers.Length != net2.filterLayers.Length)
                throw new Exception();

            for (int f = 0; f < net1.filterLayers[0].squares.Length; f++)
            {
                for (int y = 0; y < net1.filterLayers[0].squares[0].width; y++)
                {
                    for (int x = 0; x < net1.filterLayers[0].squares[0].width; x++)
                    {
                        if (net1.filterLayers[0].squares[f].values[x, y] != net2.filterLayers[0].squares[f].values[x, y])
                            throw new Exception();
                    }
                }
            }
            for (int f = 0; f < net1.filterLayers[1].squares.Length; f++)
            {
                for (int y = 0; y < net1.filterLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < net1.filterLayers[1].squares[0].width; x++)
                    {
                        if (net1.filterLayers[1].squares[f].width == 0)
                            continue;
                        if (net1.filterLayers[1].squares[f].values[x, y] != net2.filterLayers[1].squares[f].values[x, y])
                            throw new Exception();
                    }
                }
            }

            for (int l = 0; l < net1.convolutedLayers[0].squares.Length; l++)
            {
                for (int y = 0; y < net1.convolutedLayers[0].squares[0].width; y++)
                {
                    for (int x = 0; x < net1.convolutedLayers[0].squares[0].width; x++)
                    {
                        if (net1.convolutedLayers[0].squares[l].biases[x, y] != net2.convolutedLayers[0].squares[l].biases[x, y])
                            throw new Exception();
                    }
                }
            }
            for (int l = 0; l < net1.convolutedLayers[1].squares[0].width; l++)
            {
                for (int y = 0; y < net1.convolutedLayers[1].squares[0].width; y++)
                {
                    for (int x = 0; x < net1.convolutedLayers[1].squares[0].width; x++)
                    {
                        if (net1.convolutedLayers[1].squares[l].biases[x, y] != net2.convolutedLayers[1].squares[l].biases[x, y])
                            throw new Exception();
                    }
                }
            }

            for (int i = 0; i < net1.weights[0].values.Length; i++)
            {
                if (net1.weights[0].values[i] != net2.weights[0].values[i])
                    throw new Exception();
            }
            for (int i = 0; i < net1.weights[1].values.Length; i++)
            {
                if (net1.weights[1].values[i] != net2.weights[1].values[i])
                    throw new Exception();
            }
            for (int i = 0; i < net1.weights[2].values.Length; i++)
            {
                if (net1.weights[2].values[i] != net2.weights[2].values[i])
                    throw new Exception();
            }

            //-------------bias layer values------------------
            for (int i = 0; i < net1.biases[0].values.Length; i++)
            {
                if (net1.biases[0].values[i] != net2.biases[0].values[i])
                    throw new Exception();
            }
            for (int i = 0; i < net1.biases[1].values.Length; i++)
            {
                if (net1.biases[1].values[i] != net2.biases[1].values[i])
                    throw new Exception();
            }
        }

    }

    
}
