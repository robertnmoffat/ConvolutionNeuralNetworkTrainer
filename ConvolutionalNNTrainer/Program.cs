using Accord.IO;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ConvolutionalNNTrainer
{
    static class Program
    {

        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);
            //Application.Run(new Form1());

            runNet();
        }

        public static void runNet() {
            var reader = new MatReader("train_32x32.mat");
            foreach (var field in reader.Fields)
                Console.WriteLine(field.Key);

            MatNode matx = reader["y"];
            byte[,] order = (byte[,])matx.Value;

            CNN errorsAvg = new CNN();
            NetworkInitializer.InitializeNetwork(ref errorsAvg);

            CNN net = new CNN();
            NetworkInitializer.InitializeNetwork(ref net);
            NetworkInitializer.randomizeWeights(ref net);

            float crossEntropy = 0.0f;
            float totalErr = 0.0f;
            int epochCounter = 0;
            int batchsize = 32;
            float trainingRate = 0.02f;
            float correct = 0;
            float guesstotal = 0;
            int trainingRounds = 0;
            float lastErr = 0.0f;
            Bitmap bitmap;
            while (true)
            {
                correct = 0;
                guesstotal = 0;
                for (int i = 0; i < 70000; i++)
                {
                    bitmap = new Bitmap(Image.FromFile(Application.StartupPath + @"\digits\" + order[i, 0] + @"\" + i + ".png"));
                    NetworkInitializer.resetNetworkNeurons(ref net);
                    //NetworkInitializer.InitializeNetwork(ref net);
                    //NetworkInitializer.randomizeWeights(ref net);
                    NetworkInitializer.setInputs(ref net, bitmap);
                    ForwardPropagation.forwardPropagate(ref net);
                    if (net.numberGuess == order[i, 0])
                    {
                        correct++;
                    }
                    guesstotal++;
                    CNN errors = new CNN();
                    NetworkInitializer.InitializeNetwork(ref errors);
                    crossEntropy += BackPropagation.backPropagate(net, ref errors, ref errorsAvg, 100, order[i, 0], trainingRate);
                    if (float.IsNaN(crossEntropy))
                    {
                        Console.WriteLine("Is NAN");
                    }

                    epochCounter++;
                    if (epochCounter > batchsize)
                    {
                        float error = crossEntropy / batchsize;
                        if (error > (totalErr / i) * 2)
                            trainingRate /= error - (totalErr / i);
                        else if (lastErr > error)
                            trainingRate += 0.001f;
                        else
                            trainingRate -= 0.001f;

                        lastErr = error;

                        if (trainingRate < 0.00001f)
                            trainingRate = 0.00001f;

                        Console.WriteLine("Error:" + error);
                        totalErr += crossEntropy;
                        Console.WriteLine("ErrAvg:" + totalErr / i + ", Accuracy:" + (correct / guesstotal) + ", Epoch:" + trainingRounds + ", TrainingRate:" + trainingRate);
                        //outputText = ("ErrAvg:" + totalErr / i + ", Accuracy:" + (correct / guesstotal) + ", Epoch:" + trainingRounds + ", TrainingRate:" + trainingRate);
                        net.adjustWeights(errorsAvg);

                        errorsAvg = new CNN();
                        NetworkInitializer.InitializeNetwork(ref errorsAvg);
                        epochCounter = 0;
                        crossEntropy = 0.0f;
                        //trainingRate *= 0.9f;
                        //correct = 0;
                        //guesstotal = 0;
                    }
                }
                trainingRounds++;
            }
            Console.WriteLine("Done.");
            for (int i = 0; i < 100; i++)
            {
                bitmap = new Bitmap(Image.FromFile(Application.StartupPath + @"\digits\5\73034.png"));
                NetworkInitializer.resetNetworkNeurons(ref net);
                //NetworkInitializer.InitializeNetwork(ref net);
                //NetworkInitializer.randomizeWeights(ref net);
                NetworkInitializer.setInputs(ref net, bitmap);
                ForwardPropagation.forwardPropagate(ref net);
                Console.WriteLine("Guess: " + net.activatedOutputs.ToString());
            }
        }
    }
}
