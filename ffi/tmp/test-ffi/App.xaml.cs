using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Threading;

namespace ObjectRecognizerApp
{
    public partial class App : Application
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        protected override void OnStartup(StartupEventArgs e)
        {
            try
            {
                // Test loading native library explicitly to catch DLL load / dependency failures early
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string dllPath = Path.Combine(baseDir, "object_recognizer.dll");
                if (File.Exists(dllPath))
                {
                    IntPtr ptr = LoadLibrary(dllPath);
                    if (ptr == IntPtr.Zero)
                    {
                        int err = Marshal.GetLastWin32Error();
                        MessageBox.Show($"Failed to load native 'object_recognizer.dll'. Win32 Error Code: {err}\n\nThis is often caused by missing Visual C++ Redistributable (vcruntime140.dll) or missing Windows Media / Media Foundation components on modified Windows 10 builds.", "Native Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
            catch (Exception ex)
            {
                File.WriteAllText("startup_error.log", ex.ToString());
            }

            base.OnStartup(e);
            this.DispatcherUnhandledException += App_DispatcherUnhandledException;
            AppDomain.CurrentDomain.UnhandledException += CurrentDomain_UnhandledException;
        }

        private void App_DispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
        {
            File.WriteAllText("error.log", e.Exception.ToString());
            MessageBox.Show($"Error:  {e.Exception.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            e.Handled = true;
            Shutdown();
        }

        private void CurrentDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            if (e.ExceptionObject is Exception ex)
            {
                File.WriteAllText("error.log", ex.ToString());
                MessageBox.Show($"Fatal Error: {ex.Message}", "Fatal Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}
