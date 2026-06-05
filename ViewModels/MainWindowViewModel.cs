using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Threading;
using NeuralNetworkApp.Core.Activation;
using NeuralNetworkApp.Core.Models;

namespace NeuralNetworkApp.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {
        private string _status = "Gotowy do pracy";
        private double _learningRate = 0.1;
        private int _maxEpochs = 1000;
        private double _maxError = 0.01;
        private string _hiddenLayersConfig = "4,4";
        private int _selectedActivationIndex = 0;
        private string _filePath = "data.tab";
        private int _inputCount = 2;
        private int _outputCount = 1;
        private string _currentEpochDisplay = "Epoka: 0/0";
        private Network? _activeNetwork;
        private double[][]? _inputs;
        private double[][]? _targets;
        private int _currentStepEpoch = 0;
        private double _maxObservedMse = 0.001;
        private string _loadedDataContent = string.Empty;
        private readonly List<double> _mseHistory = new List<double>();

        public string LoadedDataContent
        {
            get => _loadedDataContent;
            set
            {
                if (_loadedDataContent != value)
                {
                    _loadedDataContent = value;
                    OnPropertyChanged();
                    ResetTrainingState();
                }
            }
        }

        public ObservableCollection<string> LogEntries { get; } = new ObservableCollection<string>();

        private Avalonia.Points _chartPoints = new Avalonia.Points();
        public Avalonia.Points ChartPoints
        {
            get => _chartPoints;
            set
            {
                _chartPoints = value;
                OnPropertyChanged();
            }
        }

        public string Status
        {
            get => _status;
            set
            {
                if (_status != value)
                {
                    _status = value;
                    OnPropertyChanged();
                }
            }
        }

        public double LearningRate
        {
            get => _learningRate;
            set
            {
                if (_learningRate != value)
                {
                    _learningRate = value;
                    OnPropertyChanged();
                }
            }
        }

        public int MaxEpochs
        {
            get => _maxEpochs;
            set
            {
                if (_maxEpochs != value)
                {
                    _maxEpochs = value;
                    OnPropertyChanged();
                }
            }
        }

        public double MaxError
        {
            get => _maxError;
            set
            {
                if (_maxError != value)
                {
                    _maxError = value;
                    OnPropertyChanged();
                }
            }
        }

        public string HiddenLayersConfig
        {
            get => _hiddenLayersConfig;
            set
            {
                if (_hiddenLayersConfig != value)
                {
                    _hiddenLayersConfig = value;
                    OnPropertyChanged();
                    ResetTrainingState();
                }
            }
        }

        public int SelectedActivationIndex
        {
            get => _selectedActivationIndex;
            set
            {
                if (_selectedActivationIndex != value)
                {
                    _selectedActivationIndex = value;
                    OnPropertyChanged();
                    ResetTrainingState();
                }
            }
        }

        public string FilePath
        {
            get => _filePath;
            set
            {
                if (_filePath != value)
                {
                    _filePath = value;
                    OnPropertyChanged();
                    LoadedDataContent = string.Empty;
                    ResetTrainingState();
                }
            }
        }

        public int InputCount
        {
            get => _inputCount;
            set
            {
                if (_inputCount != value)
                {
                    _inputCount = value;
                    OnPropertyChanged();
                    ResetTrainingState();
                }
            }
        }

        public int OutputCount
        {
            get => _outputCount;
            set
            {
                if (_outputCount != value)
                {
                    _outputCount = value;
                    OnPropertyChanged();
                    ResetTrainingState();
                }
            }
        }

        public string CurrentEpochDisplay
        {
            get => _currentEpochDisplay;
            set
            {
                if (_currentEpochDisplay != value)
                {
                    _currentEpochDisplay = value;
                    OnPropertyChanged();
                }
            }
        }

        private void ResetTrainingState()
        {
            _activeNetwork = null;
            _inputs = null;
            _targets = null;
            _currentStepEpoch = 0;
            _mseHistory.Clear();
        }

        private bool InitializeTrainingData()
        {
            if (string.IsNullOrEmpty(FilePath) || !File.Exists(FilePath))
            {
                Status = "Błąd: Niepoprawna ścieżka pliku danych.";
                return false;
            }

            try
            {
                if (string.IsNullOrEmpty(LoadedDataContent))
                {
                    LoadedDataContent = File.ReadAllText(FilePath);
                }

                var inputsList = new List<double[]>();
                var targetsList = new List<double[]>();

                using (var reader = new StringReader(LoadedDataContent))
                {
                    string? line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;
                        string[] tokens = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                        if (tokens.Length < InputCount + OutputCount) continue;
                        double[] input = new double[InputCount];
                        double[] target = new double[OutputCount];
                        for (int i = 0; i < InputCount; i++)
                        {
                            input[i] = double.Parse(tokens[i], System.Globalization.CultureInfo.InvariantCulture);
                        }
                        for (int i = 0; i < OutputCount; i++)
                        {
                            target[i] = double.Parse(tokens[InputCount + i], System.Globalization.CultureInfo.InvariantCulture);
                        }

                        inputsList.Add(input);
                        targetsList.Add(target);
                    }
                }

                _inputs = inputsList.ToArray();
                _targets = targetsList.ToArray();

                if (_activeNetwork == null)
                {
                    string[] layerStrings = HiddenLayersConfig.Split(',', StringSplitOptions.RemoveEmptyEntries);
                    int[] hiddenLayers = Array.ConvertAll(layerStrings, int.Parse);
                    IActivationFunction activation = SelectedActivationIndex == 0 ? new UnipolarActivation() : new BipolarActivation();
                    _activeNetwork = new Network(InputCount, hiddenLayers, OutputCount, activation, 1.0);
                }
                return true;
            }
            catch (Exception ex)
            {
                Status = $"Błąd inicjalizacji: {ex.Message}";
                return false;
            }
        }

        public async Task StartTrainingAsync()
        {
            ResetTrainingState();
            if (!InitializeTrainingData()) return;

            Status = "Uczenie automatyczne w toku...";
            LogEntries.Clear();
            ChartPoints = new Avalonia.Points();
            _maxObservedMse = 0.001;
            await Task.Run(() =>
            {
                while (_currentStepEpoch < MaxEpochs)
                {
                    _currentStepEpoch++;
                    double mse = PerformSingleTrainingEpoch();

                    UpdateInterfaceProgress(_currentStepEpoch, MaxEpochs, mse);

                    if (mse <= MaxError)
                    {
                        Dispatcher.UIThread.Post(() => Status = "Uczenie zakończone sukcesem (osiągnięto próg błędu).");
                        return;
                    }
                }
                Dispatcher.UIThread.Post(() => Status = "Uczenie zakończone (osiągnięto limit epok).");
            });
        }

        public async Task TrainSingleStepAsync()
        {
            if (_inputs == null || _activeNetwork == null)
            {
                if (!InitializeTrainingData()) return;
                LogEntries.Clear();
                ChartPoints = new Avalonia.Points();
                _currentStepEpoch = 0;
                _maxObservedMse = 0.001;
            }

            Status = "Wykonuję krok uczenia...";
            await Task.Run(() =>
            {
                _currentStepEpoch++;
                double mse = PerformSingleTrainingEpoch();
                UpdateInterfaceProgress(_currentStepEpoch, MaxEpochs, mse);

                Dispatcher.UIThread.Post(() =>
                {
                    Status = $"Uczenie krokowe. Wykonano epokę {_currentStepEpoch}.";
                });
            });
        }

        private double PerformSingleTrainingEpoch()
        {
            var inputs = _inputs;
            var targets = _targets;
            var network = _activeNetwork;

            if (inputs == null || targets == null || network == null) return 0.0;
            double totalError = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                network.Backpropagate(inputs[i], targets[i], LearningRate);
                double[] output = network.Compute(inputs[i]);

                for (int j = 0; j < output.Length; j++)
                {
                    totalError += Math.Pow(targets[i][j] - output[j], 2);
                }
            }
            return totalError / (inputs.Length * OutputCount);
        }

        private void UpdateInterfaceProgress(int currentEpoch, int maxEpochs, double mse)
        {
            Dispatcher.UIThread.Post(() =>
            {
                CurrentEpochDisplay = $"Epoka: {currentEpoch}/{maxEpochs}";
                LogEntries.Insert(0, $"[Epoka {currentEpoch:D4}] Średni błąd kwadratowy (MSE): {mse:F6}");

                _mseHistory.Add(mse);
                if (mse > _maxObservedMse) _maxObservedMse = mse;

                double canvasWidth = 600;
                double canvasHeight = 160;
                double padding = 15;
                double usableHeight = canvasHeight - (2 * padding);

                var newPoints = new Avalonia.Points();
                for (int i = 0; i < _mseHistory.Count; i++)
                {
                    double currentMse = _mseHistory[i];
                    double x = ((double)i / Math.Max(1, maxEpochs - 1)) * canvasWidth;
                    double y = canvasHeight - padding - ((currentMse / _maxObservedMse) * usableHeight);

                    if (!double.IsNaN(x) && !double.IsNaN(y))
                    {
                        newPoints.Add(new Avalonia.Point(x, y));
                    }
                }
                ChartPoints = newPoints;
            });
        }

        public async Task SaveNetworkWeightsAsync()
        {
            if (_activeNetwork == null)
            {
                Status = "Błąd: Brak zainicjalowanej sieci do zapisu.";
                return;
            }

            try
            {
                var dumpData = new List<double[]>();
                foreach (var layer in _activeNetwork.Layers)
                {
                    foreach (var neuron in layer.Neurons)
                    {
                        dumpData.Add(neuron.Weights);
                    }
                }

                string jsonString = JsonSerializer.Serialize(dumpData);
                await File.WriteAllTextAsync("network_weights.json", jsonString);
                Status = "Stan sieci został pomyślnie zapisany do network_weights.json";
            }
            catch (Exception ex)
            {
                Status = $"Błąd zapisywania: {ex.Message}";
            }
        }
    }
}