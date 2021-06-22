using Accord.IO;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ConvolutionalNNTrainer
{
    public partial class Form1 : Form
    {

        Thread thread1;
        private System.Windows.Forms.Timer _timer = new System.Windows.Forms.Timer();
        static string outputText;

        public Form1()
        {
            InitializeComponent();

            _timer.Interval = 500;
            _timer.Tick += TimerTick;
            _timer.Enabled = true;
        }

        void TimerTick(object sender, EventArgs e)
        {
            textBox1.Text = outputText;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }
        

        public class ThreadWork
        {
            public static void DoWork()
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
                NetworkInitializer.randomizeWeights(ref net);

                float crossEntropy = 0.0f;
                float totalErr = 0.0f;
                int epochCounter = 0;
                int batchsize = 32;
                float trainingRate = 0.02f;
                float correct = 0;
                float guesstotal = 0;
                int trainingRounds = 0;
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
                            if (crossEntropy / batchsize > (totalErr / i) * 2)
                                trainingRate /= crossEntropy / batchsize - (totalErr / i);
                            else
                                trainingRate += 0.001f;
                            if (trainingRate < 0.01f)
                                trainingRate = 0.01f;

                            Console.WriteLine("Error:" + crossEntropy / batchsize);
                            totalErr += crossEntropy;
                            Console.WriteLine("ErrAvg:" + totalErr / i + ", Accuracy:" + (correct / guesstotal) + ", Epoch:" + trainingRounds + ", TrainingRate:" + trainingRate);
                            outputText = ("ErrAvg:" + totalErr / i + ", Accuracy:" + (correct / guesstotal) + ", Epoch:" + trainingRounds + ", TrainingRate:" + trainingRate);
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

        private void button1_Click(object sender, EventArgs e)
        {
            thread1 = new Thread(ThreadWork.DoWork);
            thread1.IsBackground = true;
            thread1.Start();
        }
    }
}
