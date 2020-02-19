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
        
        public void Reinitialize()
        {
            
        }
        
        public void CreateModel()
        {
            if (_model != null)
            {
                Clear();
            }
        
            _model = linearCreateModel(784);
        }

        public void Train()
        {
            if (_model == null)
            {
                Debug.Log("Create model before");
                return;
            }
            
            using (StreamReader sr = new StreamReader("C:\\Users\\casag\\Documents\\ESGI\\MACHINE LEARNING\\mnist-in-csv\\mnist_train.csv"))
            {
                var numberOfEntries = 60000;
                var trainingParams = new double[numberOfEntries * 784];
                var trainingResults = new double[numberOfEntries];
                string currentLine;
                int currentLineIndex = -1; // First line of file = headers -> allows to begin data parsing at 0
                
                while((currentLine = sr.ReadLine()) != null) // CurrentLine will be null when the StreamReader reaches the end of file
                {
                    
                    if (currentLineIndex == -1)
                        continue;
                        
                    var values = currentLine.Split(',');
                    
                    for (int i = 1; i < values.Length; i++)
                    {
                        trainingParams[currentLineIndex * 784 + (i - 1)] = Convert.ToDouble(values[i]);
                    }

                    trainingResults[currentLineIndex] = Convert.ToDouble(values[0]);
                    
                    linearClassTrain(_model.Value, 784, epoch, 0.1, trainingParams, numberOfEntries, trainingResults);
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
            using (StreamReader sr = new StreamReader("C:\\Users\\casag\\Documents\\ESGI\\MACHINE LEARNING\\mnist-in-csv\\mnist_test.csv"))
            {
                int numberOfEntries = 10000;
                string currentLine;
                int currentLineIndex = -1; // La première ligne du fichier est les headers 
                
                while((currentLine = sr.ReadLine()) != null) // currentLine will be null when the StreamReader reaches the end of file
                {
                    if (currentLineIndex == -1)
                        continue;
                    
                    var values = currentLine.Split(',');
                    
                    double[] paramsDim = Array.ConvertAll(values.Skip(1).ToArray(), Double.Parse);

                    var predicted = linearClassPredict(_model.Value, 784, paramsDim);
                    // check if result (get 1) is equal to values[0] and store the result to analyze accuracy
                    
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