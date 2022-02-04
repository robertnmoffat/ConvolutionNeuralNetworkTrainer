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
        private static bool threadIsStarted = false;
        static NetworkController controller;

        Thread thread1;
        private System.Windows.Forms.Timer _timer = new System.Windows.Forms.Timer();

        public Form1()
        {
            InitializeComponent();

            //Start timer to update the output text.
            _timer.Interval = 500;
            _timer.Tick += TimerTick;
            _timer.Enabled = true;
        }

        private void TimerTick(object sender, EventArgs e)
        {
            if (controller != null)
            {
                textBox1.Text = controller.outputText;
                textBox3.Text = controller.realAccuracy;
                if (controller.displayBitmap != null) {
                    //pictureBox1.Image = controller.displayBitmap;
                }
                if (controller.isNan == true)
                {
                    controller.trainingRate -= 0.01f;
                    controller.isNan = false;
                }
                
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }
        

        public class ThreadWork
        {
            public static void DoWork()
            {
                controller = new NetworkController();
                controller.runNet();    
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (!threadIsStarted)
            {
                //Start training on a new thread.
                thread1 = new Thread(ThreadWork.DoWork);
                thread1.IsBackground = true;
                thread1.Start();

                threadIsStarted = true;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            controller.trainingRate = float.Parse(textBox2.Text);
        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
