using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using ObjectRecognizer;

namespace ObjectRecognizerApp
{
    public partial class MainWindow : Window
    {
        private sealed record DebugSearchResult(
            byte[][] Subjects,
            HitDto[] Hits,
            double SegmentationMs,
            double EmbeddingMs,
            double SearchMs,
            int SegmentedSubjectCount,
            int ImageWidth,
            int ImageHeight,
            int ImageBytesLength,
            nuint VectorCount,
            OperationTiming[] CoreTimings);

        private RecognizerWrapper? recognizer;
        private byte[]? currentImageBytes;
        private string? currentImagePath;
        private string dbPath = string.Empty;
        private bool imageProcessing;

        public MainWindow()
        {
            InitializeComponent();
            Console.WriteLine("Debug stuff");
            _ = InitializeAsync();
        }

        private async Task InitializeAsync()
        {
            try
            {
                string appDataDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "ObjectRecognizer"
                );
                Directory.CreateDirectory(appDataDir);
                dbPath = Path.Combine(appDataDir, "vecdb.bin");

                TxtStatus.Text = "Loading embedded lite model...";
                recognizer = await Task.Run(() => RecognizerWrapper.NewEmbedded(dbPath));
                TxtStatus.Text = "Ready. Drag & drop image or click Open.";
                await RefreshTimingsAsync();

                string[] args = Environment.GetCommandLineArgs();
                if (args.Length > 1 && File.Exists(args[1]))
                {
                    await LoadImage(args[1]);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to initialize RecognizerWrapper:\n{ex.Message}", "Initialization Error", MessageBoxButton.OK, MessageBoxImage.Error);
                TxtStatus.Text = "Initialization failed.";
            }
        }

        private async void BtnBrowse_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.webp"
            };

            if (dlg.ShowDialog() == true)
            {
                await LoadImage(dlg.FileName);
            }
        }

        private async void BtnCamera_Click(object sender, RoutedEventArgs e)
        {
            if (recognizer == null)
            {
                MessageBox.Show("Recognizer is not initialized.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            CameraWindow camWin = new CameraWindow();
            camWin.Owner = this;
            if (camWin.ShowDialog() == true && camWin.CapturedImageBytes != null)
            {
                await LoadImageBytes(camWin.CapturedImageBytes, "CameraCapture.png");
            }
        }

        private async void BtnBulkAdd_Click(object sender, RoutedEventArgs e)
        {
            if (recognizer == null)
            {
                MessageBox.Show("Recognizer is not initialized.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            BulkAddDialog dialog = new BulkAddDialog();
            dialog.Owner = this;
            if (dialog.ShowDialog() == true)
            {
                try
                {
                    TxtStatus.Text = "Bulk adding training samples...";
                    var recognizerInstance = recognizer;
                    string[] filePaths = dialog.SelectedFilePaths.ToArray();
                    string itemLabel = dialog.ItemLabel;
                    string itemSku = dialog.ItemSku;
                    string databasePath = dbPath;
                    var result = await Task.Run(() =>
                    {
                        int successCount = 0;
                        Stopwatch sw = Stopwatch.StartNew();
                        foreach (var filePath in filePaths)
                        {
                            if (File.Exists(filePath))
                            {
                                byte[] imgBytes = File.ReadAllBytes(filePath);
                                string metadata = $"{{\"label\":\"{itemLabel}\",\"sku\":\"{itemSku}\",\"filename\":\"{Path.GetFileName(filePath)}\",\"timestamp\":\"{DateTime.UtcNow:o}\"}}";
                                recognizerInstance.AddTrainingSample(imgBytes, metadata);
                                successCount++;
                            }
                        }

                        recognizerInstance.VectorDbSave(databasePath);
                        sw.Stop();
                        return (SuccessCount: successCount, ElapsedMs: sw.ElapsedMilliseconds);
                    });
                    MessageBox.Show($"Successfully bulk added {result.SuccessCount} images in {result.ElapsedMs} ms.", "Bulk Add Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                    TxtStatus.Text = $"Bulk added {result.SuccessCount} items in {result.ElapsedMs} ms.";
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error during bulk add:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    TxtStatus.Text = "Error during bulk add.";
                }
                finally
                {
                    await RefreshTimingsAsync();
                }
            }
        }

        private async void BtnExport_Click(object sender, RoutedEventArgs e)
        {
            if (recognizer == null) return;

            SaveFileDialog dlg = new SaveFileDialog
            {
                Filter = "Vector DB Export (*.vecdb)|*.vecdb|All Files (*.*)|*.*",
                FileName = "object_recognizer_export.vecdb"
            };

            if (dlg.ShowDialog() == true)
            {
                try
                {
                    var recognizerInstance = recognizer;
                    string exportPath = dlg.FileName;
                    var result = await Task.Run(() =>
                    {
                        Stopwatch sw = Stopwatch.StartNew();
                        ulong count = recognizerInstance.VectorDbExport(exportPath);
                        sw.Stop();
                        return (Count: count, ElapsedMs: sw.ElapsedMilliseconds);
                    });
                    MessageBox.Show($"Successfully exported {result.Count} points in {result.ElapsedMs} ms to {exportPath}", "Export Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Export failed:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    await RefreshTimingsAsync();
                }
            }
        }

        private async void BtnImport_Click(object sender, RoutedEventArgs e)
        {
            if (recognizer == null) return;

            OpenFileDialog dlg = new OpenFileDialog
            {
                Filter = "Vector DB Export (*.vecdb)|*.vecdb|All Files (*.*)|*.*"
            };

            if (dlg.ShowDialog() == true)
            {
                MessageBoxResult confirmation = MessageBox.Show(
                    "Do you want replace existing database entries? Click 'Yes' Replace, 'No' Merge.",
                    "Import Mode", MessageBoxButton.YesNoCancel, MessageBoxImage.Question);

                if (confirmation == MessageBoxResult.Cancel) return;
                bool replace = (confirmation == MessageBoxResult.Yes);

                try
                {
                    var recognizerInstance = recognizer;
                    string importPath = dlg.FileName;
                    string databasePath = dbPath;
                    var result = await Task.Run(() =>
                    {
                        Stopwatch sw = Stopwatch.StartNew();
                        ulong count = recognizerInstance.VectorDbImport(importPath, replace);
                        recognizerInstance.VectorDbSave(databasePath);
                        sw.Stop();
                        return (Count: count, ElapsedMs: sw.ElapsedMilliseconds);
                    });
                    MessageBox.Show($"Successfully imported {result.Count} points in {result.ElapsedMs} ms!", "Import Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                    TxtStatus.Text = $"Successfully imported {result.Count} points in {result.ElapsedMs} ms.";
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Import failed:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    await RefreshTimingsAsync();
                }
            }
        }

        private async void BtnPurge_Click(object sender, RoutedEventArgs e)
        {
            if (recognizer == null) return;

            MessageBoxResult result = MessageBox.Show(
                "Are you sure you want to purge (clear) the entire vector database? This action cannot be undone.",
                "Confirm Purge", MessageBoxButton.YesNo, MessageBoxImage.Warning);

            if (result != MessageBoxResult.Yes) return;

            try
            {
                var recognizerInstance = recognizer;
                string databasePath = dbPath;
                long elapsedMs = await Task.Run(() =>
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    recognizerInstance.VectorDbClear();
                    recognizerInstance.VectorDbSave(databasePath);
                    sw.Stop();
                    return sw.ElapsedMilliseconds;
                });

                // Clear current results list
                GridResults.ItemsSource = null;
                MessageBox.Show($"Successfully purged vector database in {elapsedMs} ms.", "Purge Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                TxtStatus.Text = $"Vector database purged in {elapsedMs} ms.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Purge failed:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                TxtStatus.Text = "Error purging vector database.";
            }
            finally
            {
                await RefreshTimingsAsync();
            }
        }

        private async void Window_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files != null && files.Length > 0)
                {
                    await LoadImage(files[0]);
                }
            }
        }

        private async Task RefreshTimingsAsync()
        {
            if (recognizer == null) return;
            var recognizerInstance = recognizer;
            SetTimings(await Task.Run(() => recognizerInstance.GetTimings()));
        }

        private void SetTimings(OperationTiming[] timings)
        {
            TxtTimings.Text = string.Join(
                " | ",
                timings.Select(timing => $"{timing.Name}: {timing.ElapsedMs:F2} ms"));
        }

        private static Task<DebugSearchResult> SearchDebugAsync(
            RecognizerWrapper wrapper,
            byte[] imageBytes,
            int imageWidth,
            int imageHeight,
            int k,
            CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            return Task.Run(() =>
            {
                var cutoutTimer = Stopwatch.StartNew();
                var cutouts = wrapper.SegmentSubjectRaw(imageBytes);
                cutoutTimer.Stop();

                var vectorCount = wrapper.VectorDbLen();
                var embeddingTimer = Stopwatch.StartNew();
                var embedding = wrapper.GenerateEmbedding(imageBytes);
                embeddingTimer.Stop();

                var searchTimer = Stopwatch.StartNew();
                var hits = wrapper.VectorDbSearch(embedding, (nuint)Math.Max(1, k));
                searchTimer.Stop();

                return new DebugSearchResult(
                    cutouts,
                    hits.OrderByDescending(hit => hit.Score).Take(Math.Max(1, k)).ToArray(),
                    cutoutTimer.Elapsed.TotalMilliseconds,
                    embeddingTimer.Elapsed.TotalMilliseconds,
                    searchTimer.Elapsed.TotalMilliseconds,
                    cutouts.Length,
                    imageWidth,
                    imageHeight,
                    imageBytes.Length,
                    vectorCount,
                    wrapper.GetTimings());
            }, ct);
        }

        private async Task LoadImage(string path)
        {
            if (imageProcessing) return;
            currentImagePath = path;
            byte[] bytes = File.ReadAllBytes(path);
            await LoadImageBytes(bytes, path);
        }

        private async Task LoadImageBytes(byte[] bytes, string identifier)
        {
            if (recognizer == null || imageProcessing) return;
            imageProcessing = true;

            try
            {
                currentImageBytes = bytes;

                // Display original image
                int imageWidth;
                int imageHeight;
                using (MemoryStream ms = new MemoryStream(bytes))
                {
                    BitmapImage origBmp = new BitmapImage();
                    origBmp.BeginInit();
                    origBmp.StreamSource = ms;
                    origBmp.CacheOption = BitmapCacheOption.OnLoad;
                    origBmp.EndInit();
                    ImgOriginal.Source = origBmp;
                    imageWidth = origBmp.PixelWidth;
                    imageHeight = origBmp.PixelHeight;
                }

                TxtStatus.Text = "Processing image (segmentation & embedding)...";
                BtnTrain.IsEnabled = false;
                BtnSaveSideBySide.IsEnabled = false;

                var recognizerInstance = recognizer;
                byte[] imageBytes = bytes;
                var result = await SearchDebugAsync(
                    recognizerInstance,
                    imageBytes,
                    imageWidth,
                    imageHeight,
                    10);

                if (result.SegmentedSubjectCount > 0)
                {
                    // Create WPF BitmapSource directly raw RGB8 pixel buffer (224x224)
                    int width = 224;
                    int height = 224;
                    int stride = width * 3;
                    BitmapSource subjBmp = BitmapSource.Create(
                        width, height, 96, 96,
                        System.Windows.Media.PixelFormats.Rgb24,
                        null,
                        result.Subjects[0],
                        stride);
                    ImgSubject.Source = subjBmp;
                }

                GridResults.ItemsSource = result.Hits;

                BtnTrain.IsEnabled = true;
                BtnSaveSideBySide.IsEnabled = true;
                double totalMs = result.SegmentationMs + result.EmbeddingMs + result.SearchMs;
                TxtStatus.Text = $"Processed {Path.GetFileName(identifier)} | Total: {totalMs:F2}ms (Segment: {result.SegmentationMs:F2}ms, Embedding: {result.EmbeddingMs:F2}ms, DB Search: {result.SearchMs:F2}ms). {result.ImageWidth}x{result.ImageHeight}, {result.ImageBytesLength:N0} bytes, {result.VectorCount:N0} vectors, {result.SegmentedSubjectCount} cutouts, {result.Hits.Length} matches.";
                SetTimings(result.CoreTimings);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error processing image:\n{ex.Message}", "Processing Error", MessageBoxButton.OK, MessageBoxImage.Error);
                TxtStatus.Text = "Error processing image.";
            }
            finally { imageProcessing = false; }
        }

        private void BtnSaveSideBySide_Click(object sender, RoutedEventArgs e)
        {
            if (currentImageBytes == null || recognizer == null) return;

            SaveFileDialog dlg = new SaveFileDialog
            {
                Filter = "PNG Image (*.png)|*.png|JPEG Image (*.jpg)|*.jpg|All Files (*.*)|*.*",
                FileName = "side_by_side.png"
            };

            if (dlg.ShowDialog() == true)
            {
                try
                {
                    // 1. Load original image as BitmapSource / BitmapDecoder
                    BitmapImage origImg = new BitmapImage();
                    using (MemoryStream ms = new MemoryStream(currentImageBytes))
                    {
                        origImg.BeginInit();
                        origImg.CacheOption = BitmapCacheOption.OnLoad;
                        origImg.StreamSource = ms;
                        origImg.EndInit();
                    }

                    // 2. Get the segmented subject image source from ImgSubject
                    if (ImgSubject.Source is BitmapSource subjImg)
                    {
                        // Convert both to FormatConvertedBitmap (Pbgra32) for easy drawing
                        FormatConvertedBitmap origConverted = new FormatConvertedBitmap(origImg, System.Windows.Media.PixelFormats.Pbgra32, null, 0);
                        FormatConvertedBitmap subjConverted = new FormatConvertedBitmap(subjImg, System.Windows.Media.PixelFormats.Pbgra32, null, 0);

                        int targetHeight = Math.Max(origConverted.PixelHeight, subjConverted.PixelHeight);

                        // Scale original and subject to have the same height side by side
                        double origScale = (double)targetHeight / origConverted.PixelHeight;
                        double subjScale = (double)targetHeight / subjConverted.PixelHeight;

                        TransformedBitmap origScaled = new TransformedBitmap(origConverted, new System.Windows.Media.ScaleTransform(origScale, origScale));
                        TransformedBitmap subjScaled = new TransformedBitmap(subjConverted, new System.Windows.Media.ScaleTransform(subjScale, subjScale));

                        int totalWidth = origScaled.PixelWidth + subjScaled.PixelWidth;

                        DrawingVisual drawingVisual = new DrawingVisual();
                        using (DrawingContext dc = drawingVisual.RenderOpen())
                        {
                            dc.DrawImage(origScaled, new Rect(0, 0, origScaled.PixelWidth, targetHeight));
                            dc.DrawImage(subjScaled, new Rect(origScaled.PixelWidth, 0, subjScaled.PixelWidth, targetHeight));
                        }

                        RenderTargetBitmap resultBmp = new RenderTargetBitmap(totalWidth, targetHeight, 96, 96, System.Windows.Media.PixelFormats.Pbgra32);
                        resultBmp.Render(drawingVisual);

                        BitmapEncoder encoder;
                        string ext = Path.GetExtension(dlg.FileName).ToLower();
                        if (ext == ".jpg" || ext == ".jpeg")
                        {
                            encoder = new JpegBitmapEncoder();
                        }
                        else
                        {
                            encoder = new PngBitmapEncoder();
                        }

                        encoder.Frames.Add(BitmapFrame.Create(resultBmp));
                        using (FileStream fs = new FileStream(dlg.FileName, FileMode.Create))
                        {
                            encoder.Save(fs);
                        }

                        MessageBox.Show($"Successfully saved side-by-side image to:\n{dlg.FileName}", "Save Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                        TxtStatus.Text = $"Saved side-by-side image to {Path.GetFileName(dlg.FileName)}.";
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to save side-by-side image:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async void BtnTrain_Click(object sender, RoutedEventArgs e)
        {
            if (recognizer == null || currentImageBytes == null || currentImagePath == null) return;

            try
            {
                string fileName = Path.GetFileName(currentImagePath);
                string metadata = $"{{\"filename\":\"{fileName}\",\"timestamp\":\"{DateTime.UtcNow:o}\"}}";
                var recognizerInstance = recognizer;
                byte[] imageBytes = currentImageBytes;
                string databasePath = dbPath;
                var result = await Task.Run(() =>
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    var ids = recognizerInstance.AddTrainingSample(imageBytes, metadata);
                    ulong id = ids.Length > 0 ? ids[0] : 0;
                    recognizerInstance.VectorDbSave(databasePath);
                    sw.Stop();

                    // Refresh search
                    float[] embedding = recognizerInstance.GenerateEmbedding(imageBytes);
                    var hits = recognizerInstance.VectorDbSearch(embedding, 10);
                    return (Id: id, ElapsedMs: sw.ElapsedMilliseconds, Hits: hits);
                });

                MessageBox.Show($"Successfully added training sample with ID {result.Id} in {result.ElapsedMs} ms!", "Added", MessageBoxButton.OK, MessageBoxImage.Information);
                GridResults.ItemsSource = result.Hits;
                TxtStatus.Text = $"Training sample added in {result.ElapsedMs} ms.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error adding training sample:\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                await RefreshTimingsAsync();
            }
        }
    }
}
