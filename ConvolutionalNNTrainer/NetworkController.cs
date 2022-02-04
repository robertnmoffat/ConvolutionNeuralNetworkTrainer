using Accord.IO;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using MNIST.IO;
using System.IO;

namespace ConvolutionalNNTrainer
{
    /// <summary>
    /// Class which controls the training of CNN objects.
    /// </summary>
    class NetworkController
    {
        private static float totalError = 0.0f;
        private static float totalErr = 0.0f;
        private static int epochCounter = 0;
        private static int batchsize = 4;//32;
        public float trainingRate = 1.0f;
        private static float correct = 0;
        private static float guesstotal = 0;
        private static int trainingRounds = 0;
        private static Bitmap bitmap;
        public bool isNan = false;

        public string outputText="";
        public string realAccuracy = "";
        public Bitmap displayBitmap;

        private bool loadNet = false;

        public NetworkController() { }

        /// <summary>
        /// Method which handles training of the CNN object.
        /// </summary>
        public void runNet()
        {
            var reader = new MatReader("train_32x32.mat");
            foreach (var field in reader.Fields)
                Console.WriteLine(field.Key);

            MatNode matx = reader["y"];
            byte[,] order = (byte[,])matx.Value;

            CNN errorsAvg = new CNN();
            NetworkInitializer.InitializeNetwork(ref errorsAvg);

            CNN net = new CNN();
            NetworkInitializer.InitializeNetwork(ref net);
            if(loadNet)//If desired, load previously saved network object.
                net = FileIO.loadNet(Application.StartupPath + @"\savedNet");
            else//else randomize a fresh network object.
                NetworkInitializer.randomizeWeights(ref net);


            byte[] labels = FileReaderMNIST.LoadLabel("./train-labels-idx1-ubyte.gz");
            Random rand = new Random(DateTime.Now.Millisecond);

            string[][] filenames = new string[10][];
            for (int i = 0; i < 10; i++)
            {
                filenames[i]  = Directory.GetFiles(Application.StartupPath + @"\roastNums\"+i);
            }

            while (true)
            {
                correct = 0;
                guesstotal = 0;
                for (int i = 0; i < 8745; i++)
                {
                    int digit = labels[i];

                    //Increase batch size during training to slowly increase averaging of training as it approaches correctness.
                    if (i % 1000 == 0)
                        batchsize++;

                    //For every 10 rounds train on an MNIST image
                    if (i % 10 == 0)
                    {
                        bitmap = new Bitmap(Image.FromFile(Application.StartupPath + @"\MNIST\" + digit + @"\" + i + ".png"));
                    }
                    else {
                        digit = rand.Next(0, 11);//Randomly select a training input.
                        if (digit == 10)//10 represents "not a number", use pixel gibberish.
                        {
                            bitmap = BitmapTools.generateRandom32Bitmap();
                            displayBitmap = bitmap;
                        }
                        else
                        {
                            int folderSize = filenames[digit].Length;
                            int filenum = rand.Next(0, folderSize);
                            string filename = filenames[digit][filenum];
                            bitmap = new Bitmap(Image.FromFile(filename));
                            bitmap = BitmapTools.filterBitmap(bitmap);
                        }
                    }
                    //Every fifth round, randomly shift the input to reduce overfitting.
                    if(i%5==0)
                        bitmap = BitmapTools.ResizeBitmap(bitmap, 32+(rand.Next(0,11)-5), 32 + (rand.Next(0, 11) - 5), rand.Next(0,11)-5, rand.Next(0, 11) - 5);
                    
                    //Set up the network for next training round.
                    NetworkInitializer.resetNetworkNeurons(ref net);
                    NetworkInitializer.setInputs(ref net, bitmap);
                    ForwardPropagation.forwardPropagate(ref net);
                    if (net.numberGuess == digit)//order[i, 0])
                    {
                        correct++;
                    }
                    guesstotal++;
                    CNN errors = new CNN();
                    NetworkInitializer.InitializeNetwork(ref errors);
                    totalError += BackPropagation.backPropagate(net, ref errors, ref errorsAvg, batchsize, digit, trainingRate);
                    if (float.IsNaN(totalError))
                    {
                        Console.WriteLine("Is NAN");
                        net = FileIO.loadNet(Application.StartupPath + @"\savedNet");
                        isNan = true;
                        totalError = 0.0f;
                        totalErr = 0.0f;
                        errorsAvg = new CNN();
                        NetworkInitializer.InitializeNetwork(ref errorsAvg);
                    }

                    epochCounter++;
                    if (epochCounter > batchsize)//Batch complete
                    {
                        //Increase training rate slowly to speed up training.
                        trainingRate += 0.00001f;

                        float error = totalError / batchsize;

                        Console.WriteLine("Error:" + error);
                        totalErr += totalError;
                        Console.WriteLine("ErrAvg:" + totalErr / i + ", Accuracy:" + (correct / guesstotal) + ", Epoch:" + (trainingRounds+(i/8745.0f)) + ", TrainingRate:" + trainingRate);
                        outputText = ("Error:" + error+" ErrAvg:" + totalErr / i + ", Accuracy:" + (correct / guesstotal) + ", Epoch:" + (trainingRounds + (i / 8745.0f)) + ", TrainingRate:" + trainingRate);
                        net.adjustWeights(errorsAvg);

                        errorsAvg = new CNN();
                        NetworkInitializer.InitializeNetwork(ref errorsAvg);
                        epochCounter = 0;
                        totalError = 0.0f;
                    }
                }
                //End of epoch. Save net and test against test data.
                trainingRounds++;
                totalErr = 0;

                FileIO.saveNet(net, Application.StartupPath + @"\savedNet");

                correct = 0;
                for (int i = 0; i < 100; i++)
                {
                    int digit = rand.Next(0, 10);
                    bitmap = new Bitmap(Image.FromFile(Application.StartupPath + @"\roastNums\" + digit + @"\" + rand.Next(1,10) + ".png"));
                    bitmap = BitmapTools.ResizeBitmap(bitmap, 32, 32,0,0);
                    bitmap = BitmapTools.filterBitmap(bitmap);
                    displayBitmap = bitmap;
                    NetworkInitializer.resetNetworkNeurons(ref net);
                    NetworkInitializer.setInputs(ref net, bitmap);
                    ForwardPropagation.forwardPropagate(ref net);
                    if (net.numberGuess == digit) {
                        correct += 1;
                    }
                }
                realAccuracy = ""+(correct/100);
            }
        }



    }
}
