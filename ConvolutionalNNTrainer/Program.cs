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
            Bitmap bitmap = new Bitmap(Image.FromFile("39355.png"));

            CNN net = new CNN();
            NetworkInitializer.setInputs(ref net, bitmap);
            NetworkInitializer.InitializeNetwork(ref net);
            NetworkInitializer.randomizeWeights(ref net);

            ForwardPropagation.forwardPropagate(ref net);
            Console.WriteLine(net.softMaxOutputs.values);
        }

        
    }
}
