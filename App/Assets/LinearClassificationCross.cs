using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class LinearClassificationCross : MonoBehaviour
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

    
    //Marshal

    public Transform[] testSpheres;
    public Transform[] trainingSpheres;
    public int epoch = 1000;

    private IntPtr? _model;
    
    public void Reinitialize()
    {
        // Put test spheres to y = 0
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
        var x = Math.Abs(initial.x);
        var z = Math.Abs(initial.z);
        
        if (x > 4.2 || z > 4.2 || z > 1.2 && x > 1.2)
        {
            x += 10;
            z += 10;
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
            var transformedPosition = TransformPosition(trainingSpheres[i].position);
            trainingParams[i] = transformedPosition.x;
            trainingParams[i + 1] = transformedPosition.z;
            trainingResults[i] = transformedPosition.y;
            
//            trainingParams[i] = position.x;
//            trainingParams[i + 1] = position.z;
//            trainingResults[i] = position.y;
            
//            trainingSpheres[i].position = new Vector3(transformedPosition.x, transformedPosition.y, transformedPosition.z);
        }
        linearClassTrain(_model.Value, 2, epoch, 0.01, trainingParams, trainingSphereNumber, trainingResults);
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
//            double[] paramsDim = {position.x, position.z};
            var predicted = linearClassPredict(_model.Value, 2, paramsDim);


            position = new Vector3(
                position.x,
                predicted * (float)0.5,
                position.z
            );

            position = new Vector3(transformedPosition.x, predicted * (float)0.5, transformedPosition.z);
            testSphere.position = position;
        }
    }

    private float applyModification(double initialValue)
    {
        var result = Math.Abs(initialValue);
        return (float) result;
    }
    
    public void Clear()
    {
        // Call lib Free model
        if (_model == null) return;
        
        linearClearModel(_model.Value);
        _model = null;

//        Debug.Log("Entering clear");
//
//        foreach (var testSphere in testSpheres)
//        {
//            var position = testSphere.position;
//            position.x = applyModification(position.x);
//            position.z = applyModification(position.z);
//            
//            if (position.x > 4 || position.z > 4 || position.z > 1 && position.x > 1)
//            {
//                position.x += 4;
//                position.z += 4;
//            }
//            testSphere.position = new Vector3(position.x, position.y, position.z);
//        }
//
//        foreach (var trainingSphere in trainingSpheres)
//        {
//            var position = trainingSphere.position;
//            position.x = applyModification(position.x);
//            position.z = applyModification(position.z);
//            
//            if (position.x > 4 || position.z > 4 || position.z > 1 && position.x > 1)
//            {
//                position.x += 4;
//                position.z += 4;
//            }
//            
//            trainingSphere.position = new Vector3(position.x, position.y, position.z);
//        }
    }
}
