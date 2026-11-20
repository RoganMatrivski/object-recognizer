using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace ObjectRecognizerApp
{
    public partial class BulkAddDialog : Window
    {
        public List<string> SelectedFilePaths { get; private set; } = new List<string>();
        public string ItemLabel => TxtLabel.Text.Trim();
        public string ItemSku => TxtSku.Text.Trim();

        public BulkAddDialog()
        {
            InitializeComponent();
        }

        private void BtnSelectImages_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog
            {
                Multiselect = true,
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.webp"
            };

            if (dlg.ShowDialog() == true)
            {
                SelectedFilePaths.Clear();
                ListImages.Items.Clear();

                foreach (var file in dlg.FileNames)
                {
                    SelectedFilePaths.Add(file);
                    ListImages.Items.Add(Path.GetFileName(file));
                }
            }
        }

        private void BtnProcess_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(ItemLabel))
            {
                MessageBox.Show("Please enter a label / object name.", "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (SelectedFilePaths.Count == 0)
            {
                MessageBox.Show("Please select at least one image.", "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            DialogResult = true;
            Close();
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
