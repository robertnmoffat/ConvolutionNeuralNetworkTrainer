using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNNTrainer
{
    /// <summary>
    /// Class containing static methods used in bitmap manipulations in the program.
    /// </summary>
    class BitmapTools
    {
        /// <summary>
        /// Changes the resolution of a bitmap. Translates if desired.
        /// </summary>
        /// <param name="bmp">Bitmap to be resized.</param>
        /// <param name="width">New width</param>
        /// <param name="height">New height</param>
        /// <param name="shiftx">Pixels to shift horizontally if desired.</param>
        /// <param name="shifty">Pixels to shift vertically if desired.</param>
        /// <returns></returns>
        public static Bitmap ResizeBitmap(Bitmap bmp, int width, int height, int shiftx, int shifty)
        {
            Bitmap result = new Bitmap(32, 32);
            using (Graphics g = Graphics.FromImage(result))
            {
                Rectangle imageSize = new Rectangle(0, 0, width, height);
                g.FillRectangle(Brushes.Black, imageSize);
                g.DrawImage(bmp, (32 - width) / 2 + shiftx, (32 - height) / 2 + shifty, width, height);
            }

            return result;
        }

        /// <summary>
        /// Filter a bitmap to be monochrome
        /// </summary>
        /// <param name="bitmap">Bitmap to be filtered</param>
        /// <returns>Monochrome bitmap</returns>
        public static Bitmap filterBitmap(Bitmap bitmap)
        {
            Bitmap filtered = new Bitmap(bitmap.Width, bitmap.Height);

            float filterValue = 255 / 1;

            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    Color color = bitmap.GetPixel(x, y);
                    int red = (int)color.R;
                    int green = (int)color.G;
                    int blue = (int)color.B;
                    int total = red + blue + green;
                    total = total / 5;
                    total = (int)Math.Round(((float)total / filterValue));
                    Color newColor = Color.FromArgb((int)(total * filterValue), (int)(total * filterValue), (int)(total * filterValue));
                    filtered.SetPixel(x, y, newColor);
                }
            }
            return filtered;
        }

        /// <summary>
        /// Generates a black and white randomized bitmap
        /// </summary>
        /// <returns>The randomized bitmap.</returns>
        public static Bitmap generateRandom32Bitmap()
        {
            Bitmap bitmap = new Bitmap(32, 32);
            Random rand = new Random(DateTime.Now.Millisecond);
            int clear = rand.Next(0, 10);
            for (int y = 0; y < 32; y++)
            {
                for (int x = 0; x < 32; x++)
                {
                    if (rand.Next(0, 2) == 1)
                    {
                        bitmap.SetPixel(x, y, Color.White);
                    }
                    else
                    {
                        bitmap.SetPixel(x, y, Color.Black);
                    }
                    if (clear == 0)
                        bitmap.SetPixel(x, y, Color.White);
                    if (clear == 1)
                        bitmap.SetPixel(x, y, Color.Black);
                }
            }
            return bitmap;
        }

    }


}
