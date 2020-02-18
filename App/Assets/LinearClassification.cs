using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class LinearClassification : MonoBehaviour
{
    [DllImport("lib.dll")]
    private static extern IntPtr create_model(int inDim, int outDim);
    
    [DllImport("lib.dll")]
    private static extern int predict(IntPtr model, int inDim, double[] paramsDim);

    [DllImport("lib.dll")]
    private static extern void train(IntPtr model, int inDim, int epoch, double trainingStep, 
        double[] trainingParams, int trainingParamsNumber, double[] trainingResults);

    [DllImport("lib.dll")]
    private static extern void clear_model(IntPtr model);

    
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
        
        _model = create_model(2, 0);
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
        int trainingSphereNumber = trainingSpheres.Length;
        double[] trainingParams = new double[trainingSphereNumber * 2];
        double[] trainingResults = new double[trainingSphereNumber];
        for (int i = 0; i < trainingSphereNumber; i++)
        {
            trainingParams[i * 2] = trainingSpheres[i].position.x;
            trainingParams[i * 2 + 1] = trainingSpheres[i].position.z;
            trainingResults[i] = trainingSpheres[i].position.y;
        }
        train(_model.Value, 2, epoch, 0.1, trainingParams, trainingSphereNumber, trainingResults);
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
            var predicted = predict(_model.Value, 2, paramsDim);
            
            position = new Vector3(
                position.x,
                predicted * (float)0.5,
                position.z
            );
            testSphere.position = position;
        }
    }

    public void Clear()
    {
        // Call lib Free model
        if (_model != null)
        {
            clear_model(_model.Value);
            _model = null;
        }
    }
}
