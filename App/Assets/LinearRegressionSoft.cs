using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class LinearRegressionSoft : MonoBehaviour
{
    [DllImport("lib.dll")]
    private static extern IntPtr linearCreateModel(int inDim);
    
    [DllImport("lib.dll")]
    private static extern double linearRegPredict(IntPtr model, int inDim, double[] paramsDim);

    [DllImport("lib.dll")]
    private static extern void linearRegTrain(IntPtr model, int inDim, double trainingStep, 
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
        
        _model = linearCreateModel(2);
        Debug.Log("Model created !");
    }
    
    private Vector3 TransformPosition(Vector3 initial)
    {
        var x = initial.x;
        var z = initial.z;
        if (x <= 5 && x >= 4 && z >= 3 && z <= 4 ||
            x >= -3 && x <= -2 && z >= 9 && z <= 10)
        {
            x += 3;
            z += 3;
        }

        if (x <= 4 && x >= 3 && z >= 6 && z <= 7)
        {
            x -= 3;
            z -= 3;
        }
        
        return new Vector3(x, initial.y, z);
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
            var position = trainingSpheres[i].position;
            position = TransformPosition(position);
            trainingParams[i * 2] = position.x;
            trainingParams[i * 2 + 1] = position.z;
            trainingResults[i] = position.y;
        }
        linearRegTrain(_model.Value, 2, 0.1, trainingParams, trainingSphereNumber, trainingResults);
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
            var transformedPosition = TransformPosition(position);
            double[] paramsDim = {transformedPosition.x, transformedPosition.z};
            var predicted = linearRegPredict(_model.Value, 2, paramsDim);
            position = new Vector3(
                position.x,
                (float)predicted,
                position.z
            );
            testSphere.position = position;
        }
        
        Debug.Log("Model predicted !");
    }

    public void Clear()
    {
        // Call lib Free model
        if (_model == null) return;
        
        linearClearModel(_model.Value);
        _model = null;
    }
}
