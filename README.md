# DogBreed-Classification-With-Amazon-Sagemaker
This is an image classfication problem using an imagenet pretrained model called resnet-18 with sagemaker studio 
In the project, we profile the model's performance with respect to cpu, gpu, io and memory utilization. We first run the training with a higher hyperparameter ranges and then select the best hyperparameters to retrain our model. 


## Project Set Up and Installation
The project repository is cloned from the provided link to the udacity's github repo (deep-learning-topics-within-computer-vision-nlp-project-starter)


## Dataset
The dog breed dataset is used for the training 


### Access
The data is uploaded to the S3 bucket through the AWS Gateway so that SageMaker has access to the data, using sagemaker.Session().upload() api.
- s3://{bucket-name}/dog-image-data/

## Script Files used
1. `hpo.py` for hyperparameter tuning jobs where we train the model for multiple time with different hyperparameters and search for the best one based on loss metrics.
2. `train_model.py` for really training the model with the best parameters getting from the previous tuning jobs, and put debug and profiler hooks for debugging purpose.


## Hyperparameter Tuning
I used a ResNet-18 pretrained model because it performs best for image classification tasks. Resnet is also resilient to the vanishing gradient problem, and the number of stacked layers does not degrade the network performance on the test dataset.

Below are hyperparameter types and their respective ranges used in the training
- learning rate 
- batch size
- epochs

```python
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.1, 0.11),
    "batch-size": CategoricalParameter([16, 32, 128]),
    "epochs": IntegerParameter(1, 2)
}
```
The objective type is to maximize accuracy.

```python
objective_metric_name = "average test accuracy"
objective_type = "Maximize"
metric_definitions = [{"Name": "average test accuracy", "Regex": "Test set: Average accuracy: ([0-9\\.]+)"}]
```

Best hyperparameter values

```python
hyperparameters = {'batch-size': 32, 'lr': '0.1022061234548314', 'epochs': '1'}
```


**Training Jobs:**
The initial training without hyperparameter tuning took relatively longer than the one with hyperparameter set 
![Training Jobs](https://github.com/vanusquarm/Dog-breed-prediction/blob/main/screenshots/training%20jobs.PNG)

**Best Hyperparameters:**
![Hyperparameters](https://github.com/vanusquarm/Dog-breed-prediction/blob/main/screenshots/best-training%20jobs.PNG)


## Debugging and Profiling
I first configured a debugger rule object that accepts a list of rules against output tensors that I want to evaluate. SageMaker Debugger automatically runs the ProfilerReport rule by default. This rules autogenerates a profiling report
Secondly, I configured a debugger hook parameter to adjust save intervals of the output tensors in the different training phases.
Next, I constructed a PyTorch estimator object with the debugger rule object and hook parameters.
I finally started the training job by fitting the training data to the estimator object.

### Results
My training job was quite short. Observing the peaks in utilization of cpu, gpu, memory and IO helped to better select the right instance type for training for improved resource efficiency.
However, I experienced a higher bottleneck in cpu operation indicting that the gpu was waiting most of the time for data to arrive 

![Hyperparameters](https://github.com/vanusquarm/Dog-breed-prediction/blob/main/screenshots/Cloudwatch-logs.PNG)

## Model Deployment
The deployed model runs on 1 instance type of a standard compute resource ("ml.t2.medium"). The configuration of these parameters are set using the PyTorch deploy function. 
Upon performing the model deploy, an Endpoint is created. 
To query the endpoint with the test sample input, first perform a resize, crop, toTensor, and normalization transformation on the image, and then pass the transformed image to the predict function of the endpoint.

Use [Python Pillow](https://pypi.org/project/Pillow/) and io to transform the jpg to Tensor binary and serve it to the endpoint.
```python
#   Run a prediction on the endpoint

from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
PIL_image = Image.open("dogImages/test/001.Affenpinscher/Affenpinscher_00071.jpg") 
image = transform(PIL_image) #   Your code to load and preprocess image to send to endpoint for prediction
payload = image.unsqueeze(dim=0) #  Changes the shape of tensor from [224, 224, 3] to [1, 3, 224, 224].
response = predictor.predict(payload) # Make your prediction
response
```

**ACTIVE ENDPOINT**
- SAGEMAKER STUDIO UI
![Active Endpoint](https://github.com/vanusquarm/Dog-breed-prediction/blob/main/screenshots/Active%20Endpoint.PNG)


