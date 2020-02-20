using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

public class TitanicLinearClassification : MonoBehaviour
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
        
        private const int _numberOfTrainPassengers = 60000;
        private const int _numberOfTestPassengers = 418;
        private const int _numberOfParams = 6;
        
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
            Debug.Log("Model created");
        }

        public void Train()
        {
            if (_model == null)
            {
                Debug.Log("Create model before");
                return;
            }

            using (StreamReader sr = new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), "Datasets\\titanic\\train.csv")))
            {
                var trainingParams = new double[_numberOfTrainPassengers * _numberOfParams];
                var trainingResults = new double[_numberOfTrainPassengers];
                string currentLine;
                int currentLineIndex = -1; // First line of file = headers -> allows to begin data parsing at 0
                
                while((currentLine = sr.ReadLine()) != null) // CurrentLine will be null when the StreamReader reaches the end of file
                {
                    if (currentLineIndex == -1)
                    {
                        currentLineIndex++;
                        continue;
                    }

                    var currentImageParams = currentLine.Split(',');
                    for (int paramIndex = 2; paramIndex < currentImageParams.Length; paramIndex++)
                    {
                        double param = string.IsNullOrEmpty(currentImageParams[paramIndex])
                            ? 0
                            : Convert.ToDouble(currentImageParams[paramIndex]);
                        trainingParams[currentLineIndex * _numberOfParams + (paramIndex - 2)] = param;
                    }
                    
                    trainingResults[currentLineIndex] = Convert.ToDouble(currentImageParams[1]);
                    
                    linearClassTrain(_model.Value, _numberOfParams, epoch, 0.1, trainingParams, _numberOfTrainPassengers, trainingResults);
                    
                    currentLineIndex++;
                }
            }
            Debug.Log("Model trained");
        }

        private Dictionary<int, int> GetTrainResults()
        {
            Dictionary<int, int> trainResults = new Dictionary<int, int>();
            using (StreamReader sr = new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), "Datasets\\titanic\\gender_submission.csv")))
            {
                string currentLine;
                int currentLineIndex = -1;
                
                while((currentLine = sr.ReadLine()) != null) // CurrentLine will be null when the StreamReader reaches the end of file
                {
                    if (currentLineIndex == -1)
                    {
                        currentLineIndex++;
                        continue;
                    }

                    var array = currentLine.Split(',');
                    array.Select(str => { if (str.Length == 0) str = "0"; return str; }).ToArray();
                    var currentValues = Array.ConvertAll(array, int.Parse);
                    trainResults.Add(currentValues[0], currentValues[1]);

                    currentLineIndex++;
                }
            }

            return trainResults;
        }

        public void Predict()
        { 
            if (_model == null)
            {
                Debug.Log("Create model before");
                return;
            }
            
            Dictionary<int, int> trainResults = GetTrainResults();
            
            using (var streamReader = new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), "Datasets\\titanic\\test.csv")))
            {
                string currentLine;
                int currentLineIndex = -1; // La première ligne du fichier est les headers 
                int correctPredictions = 0;
                while((currentLine = streamReader.ReadLine()) != null) // currentLine will be null when the StreamReader reaches the end of file
                {
                    if (currentLineIndex == -1)
                    {
                        currentLineIndex++;
                        continue;
                    }

                    var values = currentLine.Split(',');
                    double[] paramsDim = Array.ConvertAll(values.Skip(1).ToArray(), Double.Parse);

                    var predicted = linearClassPredict(_model.Value, _numberOfParams, paramsDim);
                    
                    Debug.Log("predicted : " + (predicted == -1 ? 0 : predicted) + " expected : " + trainResults[Convert.ToInt32(values[0])]);
                    if ((predicted == -1 ? 0 : predicted).Equals(trainResults[Convert.ToInt32(values[0])]))
                        correctPredictions++;
                    currentLineIndex++;
                }
                var accuracy = (correctPredictions/(double)_numberOfTestPassengers) * 100;
                Debug.Log("Précision : " + accuracy + "%");
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
