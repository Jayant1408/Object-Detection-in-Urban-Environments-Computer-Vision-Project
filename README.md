# Self-Driving Car Engineer Nanodegree
## Course: Computer Vision
Udacity Self-Driving Car Engineer Nanodegree – Computer Vision Module

## Introduction
In this project, we applied the skills gained in this module to develop a convolutional neural network capable of detecting and classifying objects in urban environments using the `Waymo Open Dataset`. The dataset includes annotated images of real-world traffic scenes containing vehicles, pedestrians, and cyclists.
<!-- 
## TODO

<a href="https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/2022-10-16-Report.md" target="_blank" rel="noopener noreferrer">
<img src="out/figures/report/2022-10-16-Figure-8-Evaluation-1.gif" width="80%" height="80%" alt="Fig 1. Inference run in an urban residential neighbourhood of San Francisco on the final object detection model trained over the Waymo Open Dataset.">
</a>

$$
\begin{align}
\textrm{Fig. 1. Results from the final object detection model: inference in an urban residential neighbourhood in San Francisco, CA.} \\
\end{align}
$$
## TODO -->

We began by conducting a thorough exploratory data analysis — examining label distributions, visualizing sample images, and identifying patterns of object occlusion. This analysis informed our choice of data augmentation techniques to improve model robustness. We then trained an object detection model using a deep convolutional architecture, monitored its performance with TensorBoard, and applied early stopping based on validation metrics.

The project utilized the `TensorFlow Object Detection` API, which enabled streamlined training, evaluation, and deployment. We also generated visual outputs and videos to showcase the model's detection performance on unseen data.

## Core Goals
* Trained an object detection model on the Waymo dataset.
* Gained hands-on experience with the TensorFlow Object Detection API.
* Tuned key hyperparameters to optimize detection performance.
* Performed an in-depth error analysis to understand the model's limitations. 

### Learning Outcomes
* Developed core machine learing skills in the context of autonomous vehicle perception.
* Learn about a complete ML pipeline from problem framing and metric selection to model iteration and evaluation.
* Learned how to process raw image data followed by performing camera calibration and distortion correction.
* Built, trained, and deployed convolutional neural networks for object detection.


### Projects
* ✅ 1.1: [Object Detection in Urban Environments](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/tree/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments).


### Exercises
* ✅ 1.1.1: [Choosing Metrics](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-1-1-Choosing-Metrics/2022-07-25-Choosing-Metrics-IoU.ipynb);
* ✅ 1.1.2: [Data Acquisition and Visualisation](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-1-2-Data-Acquisition-Visualisation/2022-08-01-Data-Acquisition-Visualisation.ipynb);
* ✅ 1.1.3: [Creating TensorFlow TFRecords](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-1-3-Creating-TF-Records/2022-08-03-Creating-TF-Records.ipynb);
* ✅ 1.2.1: [Camera Calibration and Distortion Correction](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-2-1-Calibration-Distortion/2022-08-10-Calibration-Distortion-Correction.ipynb);
* ✅ 1.2.2: [Image Manipulation and Masking](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-2-2-Image-Manipulation/2022-08-17-Image-Manipulation-Masking.ipynb);
* ✅ 1.2.3: [Geometric Transformations](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-2-3-Geometric-Transformations/2022-08-23-Geometric-Transformations-Image-Augmentation.ipynb);
* ✅ 1.3.1: [Logistic Regression](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-3-1-Logistic-Regression/2022-08-27-Logistic-Regression.ipynb);
* ✅ 1.3.2: [Stochastic Gradient Descent](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-3-2-Stochastic-Gradient-Descent/2022-08-29-Stochastic-Gradient-Descent.ipynb);
* ✅ 1.3.3: [Image Classification with Feedforward Neural Networks (FNNs)](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-3-3-Image-Classification-FNNs/2022-09-05-Image-Classification-Feed-Forward-Neural-Networks.ipynb);
* ✅ 1.4.1: [Pooling Layers in Convolutional Neural Networks (CNNs)](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-4-1-Pooling-Layers-CNNs/2022-09-07-Pooling-Layers-Convolutional-Neural-Networks.ipynb);
* ✅ 1.4.2: [Building Custom Convolutional Neural Networks (CNNs)](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-4-2-Building-Custom-CNNs/2022-09-12-Building-Custom-Convolutional-Neural-Networks.ipynb);
* ✅ 1.4.3: [Image Augmentations for the Driving Domain](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-4-3-Image-Augmentations/2022-09-19-Image-Augmentations.ipynb);
* ✅ 1.5.1: [Non-Maximum Suppression (NMS) and Soft-NMS](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-5-1-Non-Maximum-Suppression/2022-09-21-Non-Maximum-Suppression.ipynb);
* ✅ 1.5.2: [Mean Average Precision (mAP)](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-5-2-Mean-Average-Precision/2022-09-25-Mean-Average-Precision.ipynb);
* ✅ 1.5.3: [Learning Rate Schedules and Adaptive Learning Rate methods](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-5-3-Learning-Rate-Schedules/2022-09-28-Learning-Rate-Schedules.ipynb).
* ✅ 1.6.1: [Fully Convolutional Networks](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/Exercises/1-6-1-Fully-Convolutional-Networks/2023-01-23-Fully-Convolutional-Networks.ipynb).


### Course Contents

The following topics are covered in course exercises:
* Image classification with Convolutional Neural Networks (CNNs)
* Object detection with TensorFlow API
* Precision/recall, AP, mAP metrics for object detection
* Bounding box prediction
* Intersection over Union (IoU)
* Non-maximum Suppression / Soft-NMS
* Machine Learning (ML) workflows with TensorFlow Sequential, Functional API
* Model subclassing with TensorFlow
* Camera calibration (DLT, Levenberg-Marquardt)
* Camera pinhole and perspective projection models
* Recovering intrinsic/extrinsic parameters
* Colour thresholding
* Colour models (HSV, HSL, RGB)
* Binary masks and image masking
* Geometric transformations (affine, euclidean, etc.)
* Transformation and rotation matrices
* Data augmentation (e.g., random cropping, re-scaling, data generators, selective blending, etc.)
* Data augmentation with Albumentations (simulating motion, occlusions, time-of-day, etc.)
* Automated data augmentation (e.g., proxy/target tasks, policies, Smart/RandAugment, P/PBA)
* ETL pipelines
* Serialising binary datatypes (`.tfrecords` and `TFRecordDataset`)
* Protocol buffers (Google `protobuf`)
* Stochastic gradient descent
* Custom learning rate schedules
* Pooling and convolutional layers
* Padding and stride hyperparameters
* Filters and feature maps in 1D/2D
* Exploratory Data Analysis (EDA)
* TensorFlow Model Callbacks (e.g., TensorBoard)
* Image classification on MNIST and GTSRB datasets
* Traditional CNN architectures (LeNet-5)
* Tuning CNNs (e.g., addressing gradient decay, BatchNorm/Dropout, hyperparameter tuning, etc.)
* Building lightweight CNN architectures for embedded hardware
* Using custom activation functions (e.g., LeCun scaled `tanh`)
* Custom layer subclassing (e.g., Sub-sampling layer in LeNet-5)
* Selecting optimizers / loss and objective functions
* Complex model architectures and components (SSDs, RetinaNet, FPNs, RCNNs, SPPs)
* Improving object detection models for the self-driving car domain
* Monitoring GPU utilisation during training (and large-scale training on TPUs!)
* Designing skip connections;
* Transposed convolution layers;
* Fully Convolutional Networks and their performance (e.g., FPN-8);
* And so much more...


Other topics covered in course lectures and reading material:
* Deep learning history
* Tradeoffs
* Framing ML problems
* Metrics and error analysis in ML
* Economic impact and broader consequences of SDCs
* Camera models and calibration / reconstruction
* Pixel-level and geometric image transformations
* Backpropagation (and calculations performed by hand!)
* Traditional CNN architectures (AlexNet, VGG, ResNet, Inception)
* Selective search algorithm
* Region-proposal networks and improvements (RCNN, Fast-RCNN, Faster-RCNN, SPPNet)
* One- and two-stage detectors (YOLO versus SSD and CenterNet)
* Optimising deep neural networks (fine-tuning strategies, dropout / inverted dropout, batch normalisation)
* Transfer learning and applications at Waymo

### Learning Outcomes
#### Lesson 1: The Machine Learning Workflow
* Identify key stakeholders in a ML problem;
* Frame the ML problem;
* Perform exploratory data analysis (EDA) on an image dataset;
* Pick the most adequate model for a particular ML task;
* Choose the correct metric(s);
* Select and visualise the data.

#### Lesson 2: Sensor and Camera Calibration
* Manipulate image data;
* Calibrate an image using checkerboard images;
* Perform geometric transformation of an image;
* Perform pixel-level transformations of an image.

#### Lesson 3: From Linear Regression to Feedforward Neural Networks
* Implement a logistic regression model in TensorFlow;
* Implement back propagation;
* Implement gradient descent;
* Build a custom neural network for a classification task.

#### Lesson 4: Image Classification with Convolutional Neural Networks
* Write custom classification architectures using TensorFlow;
* Choose the right augmentations to increase dataset variability;
* Use regularisation techniques to prevent overfitting
* Calculate the output shape of a convolutional layer;
* Count the number of parameters in a convolutional network.

#### Lesson 5: Object Detection in Images
* Use the TensorFlow Object Detection API;
* Choose the best object detection model for a given problem;
* Optimise training processes to maximise resource usage;
* Implement Non-Maximum Suppression (NMS);
* Calculate Mean Average Precision (mAP);
* Choose hyper parameters to optimise a neural network.

#### Lesson 6: Fully Convolutional Networks
* Converting fully-connected to 1x1 convolution layers;
* Using transposed convolutions to upsample feature maps;
* Designing skip connections to improve segmentation map granularity;
* Encoder / decoder network architectures;
* Comparing the performance of fully convolutional networks (e.g., FCN-8s) to traditional CNNs;
* Implementing fully convolutional networks in TensorFlow using Sequential and Function API design patterns.

### Material
Syllabus:
* [Program Syllabus | Udacity Nanodegree](https://d20vrrgs8k4bvw.cloudfront.net/documents/en-US/Self-Driving+Car+Engineer+Nanodegree+Syllabus+nd0013+.pdf).

Literature:
* See specific assignments for related literature.

Datasets:
* [German Traffic Sign Recognition Benchmark](https://doi.org/10.17894/ucph.358970eb-0474-4d8f-90b5-3f124d9f9bc6) (GTSRB);
* [Waymo Open Dataset: Perception](https://waymo.com/open/).

Lectures:
* Lecture materials (videos, slides) available offline. Course lecture notes available on request.

### Other resources
Companion code:
* [Object Detection in an Urban Environment | Starter code](https://github.com/udacity/nd013-c1-vision-starter).