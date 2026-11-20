using System;
using System.Windows;
using System.Windows.Threading;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;

namespace ObjectRecognizerApp
{
    public partial class CameraWindow : System.Windows.Window
    {
        private static VideoCapture? sharedCapture;
        private static bool isInitialized = false;

        private DispatcherTimer? timer;
        private Mat? currentFrame;
        public byte[]? CapturedImageBytes { get; private set; }

        public CameraWindow()
        {
            InitializeComponent();
            StartCamera();
        }

        private void StartCamera()
        {
            try
            {
                if (sharedCapture == null || !sharedCapture.IsOpened())
                {
                    sharedCapture = new VideoCapture(0, VideoCaptureAPIs.ANY);
                }

                if (!sharedCapture.IsOpened())
                {
                    MessageBox.Show("Could not open camera.", "Camera Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    Close();
                    return;
                }

                currentFrame = new Mat();
                timer = new DispatcherTimer
                {
                    Interval = TimeSpan.FromMilliseconds(30)
                };
                timer.Tick += Timer_Tick;
                timer.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Camera initialization error:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Close();
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            if (sharedCapture != null && sharedCapture.IsOpened() && currentFrame != null)
            {
                sharedCapture.Read(currentFrame);
                if (!currentFrame.Empty())
                {
                    ImgPreview.Source = currentFrame.ToBitmapSource();
                }
            }
        }

        private void BtnCapture_Click(object sender, RoutedEventArgs e)
        {
            if (currentFrame != null && !currentFrame.Empty())
            {
                CapturedImageBytes = currentFrame.ImEncode(".png");
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("No frame captured yet.", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            timer?.Stop();
            currentFrame?.Dispose();
            // Note: We keep sharedCapture alive in the background so reopening is instant!
        }

        public static void ReleaseSharedCamera()
        {
            try
            {
                sharedCapture?.Release();
                sharedCapture?.Dispose();
                sharedCapture = null;
            }
            catch { }
        }
    }
}
