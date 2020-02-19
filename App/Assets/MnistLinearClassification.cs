using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DefaultNamespace
{
    public class MnistLinearClassification : MonoBehaviour
    {
        [DllImport("lib.dll")]
        private static extern IntPtr linearCreateModel(int inDim);
    
        [DllImport("lib.dll")]
        private static extern int linearClassPredict(IntPtr model, int inDim, double[] paramsDim);

        [DllImport("lib.dll")]
        private static extern void linearClassTrain(IntPtr model, int inDim, int epoch, double trainingStep, 
            double[] trainingParams, int trainingParamsNumber, double[] trainingResults);

        [DllImport("lib.dll")]
        private static extern void linearClearModel(IntPtr model);
        
        private IntPtr? _model;
        
        public int epoch = 1000;
        
        private const int _numberOfTrainImages = 60000;
        private const int _numberOfParams = 784;
        
        public void Reinitialize()
        {
            
        }
        
        public void CreateModel()
        {
            if (_model != null)
            {
                Clear();
            }
        
            _model = linearCreateModel(_numberOfParams);
        }

        public void Train()
        {
            if (_model == null)
            {
                Debug.Log("Create model before");
                return;
            }

            using (StreamReader sr = new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), "Datasets\\mnist-in-csv\\mnist_test.csv")))
            {
                var trainingParams = new double[_numberOfTrainImages * _numberOfParams];
                var trainingResults = new double[_numberOfTrainImages];
                string currentLine;
                int currentLineIndex = -1; // First line of file = headers -> allows to begin data parsing at 0
                
                while((currentLine = sr.ReadLine()) != null) // CurrentLine will be null when the StreamReader reaches the end of file
                {
                    if (currentLineIndex == -1)
                        continue;
                        
                    var currentImageParams = currentLine.Split(',');
                    for (int paramIndex = 1; paramIndex < currentImageParams.Length; paramIndex++)
                    {
                        trainingParams[currentLineIndex * _numberOfParams + (paramIndex - 1)] = Convert.ToDouble(currentImageParams[paramIndex]);
                    }
                    
                    trainingResults[currentLineIndex] = Convert.ToDouble(currentImageParams[0]);
                    
                    linearClassTrain(_model.Value, _numberOfParams, epoch, 0.1, trainingParams, _numberOfTrainImages, trainingResults);
                    
                    currentLineIndex++;
                }
            }
        }

        public void Predict()
        { 
            if (_model == null)
            {
                Debug.Log("Create model before");
                return;
            }
            
            using (var streamReader = new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), "Datasets\\mnist-in-csv\\mnist_test.csv")))
            {
                string currentLine;
                int currentLineIndex = -1; // La première ligne du fichier est les headers 
                
                while((currentLine = streamReader.ReadLine()) != null) // currentLine will be null when the StreamReader reaches the end of file
                {
                    if (currentLineIndex == -1)
                        continue;
                    
                    var values = currentLine.Split(',');
                    
                    double[] paramsDim = Array.ConvertAll(values.Skip(1).ToArray(), Double.Parse);

                    var predicted = linearClassPredict(_model.Value, _numberOfParams, paramsDim);
                    // TODO: check if result (get 1) is equal to values[0] and store the result to analyze accuracy
                    
                    currentLineIndex++;
                }
            }
        }
        
        public void Clear()
        {
            // Call lib Free model
            if (_model == null) return;
        
            linearClearModel(_model.Value);
            _model = null;
        }
    }
}