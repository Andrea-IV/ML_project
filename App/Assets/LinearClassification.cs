using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class LinearClassification : MonoBehaviour
{
    [DllImport("lib.dll")]
    private static extern IntPtr linearCreateModel(int inDim, int outDim);
    
    [DllImport("lib.dll")]
    private static extern IntPtr linearClassPredict(IntPtr model, int inDim, int outDim, double[] paramsDim);

    [DllImport("lib.dll")]
    private static extern void linearClassTrain(IntPtr model, int inDim, int outDim, int epoch, double trainingStep, 
        double[] trainingParams, int trainingParamsNumber, double[] trainingResults);

    [DllImport("lib.dll")]
    private static extern void linearClearModel(IntPtr model);

    
    //Marshal

    public Transform[] testSpheres;
    public Transform[] trainingSpheres;
    public int epoch = 1000;

    private IntPtr? _model;
    
    public void Reinitialize()
    {
        // Put test spheres to 0
        foreach (var testSphere in testSpheres)
        {
            var position = testSphere.position;
            testSphere.position = new Vector3(position.x,0, position.z);
        }
    }

    public void CreateModel()
    {
        if (_model != null)
        {
            Clear();
        }
        
        _model = linearCreateModel(2, 1);
        Debug.Log("Model created !");
    }

    public void Train()
    {
        if (_model == null)
        {
            Debug.Log("Create model before");
            return;
        }
        
        // Call lib to train on the array
        var trainingSphereNumber = trainingSpheres.Length;
        var trainingParams = new double[trainingSphereNumber * 2];
        var trainingResults = new double[trainingSphereNumber];
        for (var i = 0; i < trainingSphereNumber; i++)
        {
            trainingParams[i * 2] = trainingSpheres[i].position.x;
            trainingParams[i * 2 + 1] = trainingSpheres[i].position.z;
            trainingResults[i] = trainingSpheres[i].position.y;
        }
        linearClassTrain(_model.Value, 2, 1, epoch, 0.1, trainingParams, trainingSphereNumber, trainingResults);
        Debug.Log("Model trained !");
    }

    public void Predict()
    {
        if (_model == null)
        {
            Debug.Log("Create model before");
            return;
        }
        
        // Call lib to predict test spheres
        foreach (var testSphere in testSpheres)
        {
            var position = testSphere.position;
            double[] paramsDim = {position.x, position.z};
            var predictedPtr = linearClassPredict(_model.Value, 2, 1, paramsDim);
            var predictedLength = Marshal.ReadInt32(predictedPtr);
            var predicted = new int[predictedLength];
            Marshal.Copy(IntPtr.Add(predictedPtr, 4), predicted, 0, predictedLength);
            
            position = new Vector3(
                position.x,
                predicted[0] * (float)0.5,
                position.z
            );
            testSphere.position = position;
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
