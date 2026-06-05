using Avalonia.Controls;
using NeuralNetworkApp.ViewModels;

namespace NeuralNetworkApp.Views
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainWindowViewModel();
        }
    }
}