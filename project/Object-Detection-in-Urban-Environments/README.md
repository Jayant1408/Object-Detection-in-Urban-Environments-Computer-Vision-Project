# Project: Object Detection in an Urban Environment
Udacity Self-Driving Car Engineer Nanodegree – Computer Vision Module

## Introduction
In this project, we applied the skills gained in this module to develop a convolutional neural network capable of detecting and classifying objects in urban environments using the `Waymo Open Dataset`. The dataset includes annotated images of real-world traffic scenes containing vehicles, pedestrians, and cyclists.

## TODO

<a href="https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/2022-10-16-Report.md" target="_blank" rel="noopener noreferrer">
<img src="out/figures/report/2022-10-16-Figure-8-Evaluation-1.gif" width="80%" height="80%" alt="Fig 1. Inference run in an urban residential neighbourhood of San Francisco on the final object detection model trained over the Waymo Open Dataset.">
</a>

$$
\begin{align}
\textrm{Fig. 1. Results from the final object detection model: inference in an urban residential neighbourhood in San Francisco, CA.} \\
\end{align}
$$
## TODO

We began by conducting a thorough exploratory data analysis — examining label distributions, visualizing sample images, and identifying patterns of object occlusion. This analysis informed our choice of data augmentation techniques to improve model robustness. We then trained an object detection model using a deep convolutional architecture, monitored its performance with TensorBoard, and applied early stopping based on validation metrics.

The project utilized the `TensorFlow Object Detection` API, which enabled streamlined training, evaluation, and deployment. We also generated visual outputs and videos to showcase the model's detection performance on unseen data.

## Core Goals
* Trained an object detection model on the Waymo dataset.
* Gained hands-on experience with the TensorFlow Object Detection API.
* Tuned key hyperparameters to optimize detection performance.
* Performed an in-depth error analysis to understand the model's limitations. 

## TensorFlow Object Detection API
The TensorFlow Object Detection API simplifies the development and training of state-of-the-art object detection models. A full list of pre-trained models can be viewed on the `TensorFlow Object Detection Model Zoo`.


## File Descriptions

Filename                                                                            | Description
------------------------------------------------------------------------------------|--------------
`2022-10-11-Setup-and-Training.ipynb`                                               | Jupyter notebook file installing project dependencies in Linux environment; executes the project script files.
`Exploratory-Data-Analysis/2022-09-29-Exploratory-Data-Analysis-Part-1.ipynb`       | Jupyter notebook file performing basic EDA on the processed Waymo perception dataset.
`Exploratory-Data-Analysis/2022-09-29-Exploratory-Data-Analysis-Part-2.ipynb`       | Jupyter notebook file looking at various class distributions, occlusions, etc.
`setup.py`                                                                          | Sets the project base directory; used with `setuptools` to install project modules on build path.
`scripts/setup/install.sh`                                                          | Set of shell commands to install project dependencies within Google Colab.
`scripts/preprocessing/download_extract.py`                                         | Downloads and extracts attribute / bounding box data from the remote GCS files in `filenames.txt`.
`scripts/preprocessing/download_process.py`                                         | Downloads and processes the remote GCS files in `filenames.txt` as a `tf.data.Dataset` instance.
`scripts/preprocessing/create_splits.py`                                            | Splits the Waymo data into training, validation and test sets.
`scripts/data_analysis/visualisation.py`                                            | Set of functions to visualise the Waymo data and export Frame objects to GIF files.
`scripts/training/edit_config.py`                                                   | Edits and saves a new `TrainEvalPipelineConfig` Google Protobuf file.
`scripts/inference/inference_video.py`                                              | Runs inference and records the bounding box predictions to a GIF file.
`configs/dataset/waymo_open_dataset.yaml`                                           | The configuration file for the Waymo Open Dataset; only modify if necessary.
`configs/model/ssd_resnet50.yaml`                                                   | The configuration file for the RetinaNet pre-trained model; modify num. train steps and TPU settings as needed.
`configs/hyperparameters/hyperparameters1.yaml`                                     | The configuration file for the model hyperparameters (epochs, batch size, etc.).
`experiments/testing_configs.py`                                                    | Tests and verifies the Hydra configurations (replaces `absl` for CLI argument overriding).
`experiments/model_main_tf2.py`                                                     | Creates and runs a TF-2 object detection model instance.
`experiments/exporter_main_v2.py`                                                   | Exports an object detection model for inference; configured with the `pipeline.config` file.
`experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/`            | Pre-trained model subdirectory containing all relevant files and checkpoints e.g., `pipeline_new.config`.
`data/waymo_open_dataset/label_map.pbtxt`                                           | StringIntLabelMap `proto` configuration file mapping class names (strings) to class IDs (integers).
`data/waymo_open_dataset/filenames.txt`                                             | List of remote paths to every Waymo Open Dataset v1.2 segment (file) hosted on the Google Cloud Storage bucket.
`addons/`                                                                           | Subdirectory to store dependencies e.g., Google Cloud SDK, COCO API, Waymo Open Dataset API, etc.
`build/`                                                                            | Dockerfile and setup instructions for local environment configuration (not used).
`out/`                                                                              | Report figures, TensorBoard training/validation scalar charts, output detection predictions, etc.


## Setup and Installation
To configure your workspace, run:
```
pip install --editable {BASE_DIR}
```
where `BASE_DIR` points to the top-level project directory. All project modules will be installed onto your PYTHONPATH automatically.

To configure a Google Colab instance, copy the shell commands from [`scripts/setup/install.sh`](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/scripts/setup/install.sh) into a cell and execute it. Note that with Google Colab instances there might be build issues when working with this project directory. It is only recommended to run individual scripts inside Google Colab.

If you are running this project inside the Linux Ubuntu LTS VM provided by Udacity, see build instructions in [`2022-10-16-Report.md`]().

## Data Acquisition

The following scripts will fetch and process the Waymo Open Dataset files from their Google Cloud Storage (GCS) hosted bucket, access to the dataset needs to be requested.

### Downloading and Processing

To download and process the `.tfrecord` files from Google Cloud Storage into `tf.data.TFRecordDataset` instances, run:

```python
python3 download_process.py
```

with none/any/all of the following parameters:
```
    DATA_DIR:        str         Path to the `data` directory to download files to.
    LABEL_MAP_PATH:  str         Path to the dataset `label_map.pbtxt` file.
    SIZE:            str         Number of `.tfrecord` files to download from GCS.
```

Overriding parameters globally is accomplished at runtime using the Basic Override syntax provided by Hydra:

```python
python3 download_process.py \
    dataset.data_dir=DATA_DIR \
    dataset.label_map_path=LABEL_MAP_PATH \
    dataset.size=SIZE
```
See `configs/dataset/`for additional details on preconfigured values if running without parameters.


### Creating Dataset Splits

To split the downloaded data into three subsets `train`, `val`, and `test`, run:

```python
python3 create_splits.py
```

with none/any/all of the following:
```
    DATA_DIR:           str         Path to the source `data` directory.
    TRAIN:              str         Path to the `train` data directory.
    TEST:               str         Path to the `test` data directory.
    VAL:                str         Path to the `val` data directory.
    TRAIN_TEST_SPLIT:   float       Percent as [0, 1] to split train/test.
    TRAIN_VAL_SPLIT:    float       Percent as [0, 1] to split train/val.
```

Overriding parameters globally is accomplished at runtime using the Basic Override syntax provided by Hydra:

```python
python3 create_splits.py \
    dataset.data_dir=DATA_DIR \
    dataset.train=TRAIN dataset.test=TEST dataset.val=VAL \
    dataset.train_test_split=TRAIN_TEST_SPLIT \
    dataset.train_val_split=TRAIN_VAL_SPLIT
```
See `configs/dataset/` for additional details on preconfigured values if running without parameters.


## Model configuration

In this project we will be using the RetinaNet (a SSD ResNet50 v1 with FPN) which has been pre-trained on the `COCO 2017` dataset. First, fetch the default configuration file and pre-trained weights from the TensorFlow Object Detection Model Zoo, or alternatively from the `TensorFlow Hub`. 

In order to use this model on the Waymo Open Dataset, we have to re-configure some of the proto definitions in the `pipeline.config` file (e.g., batch size, epochs, number of classes, etc.).


To configure the model for training, run:

```python
python3 edit_config.py
```

with none/any/all of the following parameters:
```
TRAIN:                                  str         Path to the `train` data directory.
TEST:                                   str         Path to the `test` data directory.
VAL:                                    str         Path to the `val` data directory.
BATCH_SIZE:                             int         Number of examples to process per iteration.
CHECKPOINT_DIR:                         str         Path to the pre-trained `checkpoint` folder.
LABEL_MAP_PATH:                         str         Path to the dataset `label_map.pbtxt` file.
PIPELINE_CONFIG_PATH:                   str         Path to the `pipeline.config` file to modify.
```

Overriding parameters globally is accomplished at runtime using the Basic Override syntax provided by Hydra:

```python
python3 edit_config.py \
    dataset.train=TRAIN dataset.test=TEST dataset.val=VAL \
    dataset.label_map_path=LABEL_MAP_PATH \
    hyperparameters.batch_size=BATCH_SIZE \
    model.checkpoint_dir=CHECKPOINT_DIR \
    model.pipeline_config_path=PIPELINE_CONFIG_PATH
```
See `configs/model/` for additional details on preconfigured values if running without parameters.


## Training and Evaluation

For local training/evaluation run:

```python
python3 model_main_tf2.py
```

with none/any/all of the following parameters:
```
DIR_BASE:                               str         Path to the current `model` subdirectory.
MODEL_OUT:                              str         Path to the `/tmp/model_outputs` folder.
CHECKPOINT_DIR:                         str         Path to the pre-trained weights/variables saved in `checkpoint` folder (see note below).
PIPELINE_CONFIG_PATH:                   str         Path to the `pipeline_new.config` file (output from `edit_config.py`).
NUM_TRAIN_STEPS:                        int         Number of training steps (batch iterations) to perform. 
EVAL_ON_TRAIN_DATA:                     bool        If True, will evaluate on training data (only supported in distributed training).
SAMPLE_1_OF_N_EVAL_EXAMPLES:            int         Number of evaluation samples to skip (will sample 1 of every n samples per batch).
SAMPLE_1_OF_N_EVAL_ON_TRAIN_EXAMPLES:   int         Number of training samples to skip (only used if `eval_on_train_data` is True).
EVAL_TIMEOUT:                           int         Number of seconds to wait for an evaluation checkpoint before exiting.
USE_TPU:                                bool        Whether or not the job is executing on a TPU.
TPU_NAME:                               str         Name of the Cloud TPU for Cluster Resolvers.
CHECKPOINT_EVERY_N:                     int         Defines how often to checkpoint (every n steps).
RECORD_SUMMARIES:                       bool        Whether or not to record summaries defined by the model or training pipeline.
NUM_WORKERS:                            int         When `num_workers` > 1, training uses 'MultiWorkerMirroredStrategy',
                                                    When `num_workers` = 1, training uses 'MirroredStrategy'.
```

Overriding parameters globally is accomplished at runtime using the Basic Override syntax provided by Hydra (see note below):

```python
python3 model_main_tf2.py \
    model.pipeline_config_path=PIPELINE_CONFIG_PATH \
    model.model_out=MODEL_OUT model.num_train_steps=NUM_TRAIN_STEPS \
    model.sample_1_of_n_eval_examples=SAMPLE_1_OF_N_EVAL_EXAMPLES \
    ...
```
See `configs/model/` for additional details on preconfigured values if running without parameters.

## TODOOOOO
**NOTE**: for now, Hydra configuration support has been replaced with `argparse` command line arguments. This is due to an issue mentioned [here](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/issues/22). Hydra will be replacing the temporary `argparse` functionality in the [`model_main_tf2.py`](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/experiments/model_main_tf2.py) and [`exporter_main_v2.py`](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/experiments/exporter_main_v2.py) scripts as soon as the bug as been resolved.


**NOTE**: the `CHECKPOINT_DIR` flag should only be included when running [`model_main_tf2.py`](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/experiments/model_main_tf2.py) for continuous evaluation alongside the training loop. In any case, the `CHECKPOINT_DIR` path included in the [`edit_config.py`](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/scripts/training/edit_config.py) script should point to the _pre-trained model weights_, i.e., the `ckpt-0.data` and `ckpt-0.index` files downloaded from the TensorFlow Object Detection Model Zoo. The path to these checkpoint files inside the `checkpoint` folder should be  `/path/to/checkpoint/ckpt-0`. This isn't obvious upon first glance, since the path appears to reference `ckpt-0` as a subfolder of `checkpoint` when in fact it is not.

### TODOOO REMOVE If REQUIRED

To export the trained model, run:

```python
python3 exporter_main_v2.py
```

with the following configuration parameters (see note on Hydra below):
```
input_type:                             object      Here we specify the `image_tensor` type.
PIPELINE_CONFIG_PATH:                   str         Path to the `pipeline_new.config` file (output from `edit_config.py`).
TRAINED_CHECKPOINT_DIR:                 str         Path to the training checkpoints, i.e., the `checkpoint` folder (see note below).
EXPORTED_MODEL_DIR:                     str         Path to the destination folder to store the saved model files.
```


```python 
python3 exporter_main_v2.py \
        --input_type image_tensor \
        --pipeline_config_path {PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_dir {TRAINED_CHECKPOINT_DIR} \
        --output_directory {EXPORTED_MODEL_DIR} \
```

**NOTE**: The `TRAINED_CHECKPOINT_DIR` in `exporter_main_v2.py` differs from the one used in `model_main_tf2.py` above. This path should instead point to the folder where the _training_ checkpoints are saved, which should be inside the `MODEL_OUT` path configured above.


### Inference

To run inference over a `.tfrecord` file, run:

```python
python3 inference_video.py
```

with the following parameters provided as arguments to the command above (see note on Hydra below):
```
EXPORTED_MODEL_PATH:                    str         Path to the exported and saved model, i.e., the `model/saved_model` folder.
LABEL_MAP_PATH:                         str         Path to the dataset `label_map.pbtxt` file.
PIPELINE_CONFIG_PATH:                   str         Path to the `pipeline.config` file.
TF_RECORD_PATH:                         str         Path to the `.tfrecord` file to perform inference (object detection) over.
INFERENCE_OUTPUT_PATH:                  str         Path to the output destination to save the inference video (a GIF file).
```

```python
python3 scripts/inference/inference_video.py \
	--labelmap_path {LABEL_MAP_PATH} \
	--model_path {EXPORTED_MODEL_PATH} \
	--config_path {PIPELINE_CONFIG_PATH}
	--tf_record_path {TF_RECORD_PATH} \
	--output_path {OUTPUT_PATH}
```

This script will export a video of the detection results on each frame of the `.tfrecord`. The video file is saved as a GIF.

**NOTE**: The `MODEL_PATH` should point to the folder where the _training_ checkpoints are saved. This path should be inside the `MODEL_OUT` folder and also be equivalent to `TRAINED_CHECKPOINT_DIR` configured above. 

**NOTE**: for now, Hydra configuration support has been replaced with `argparse` command line arguments. This is due to an issue mentioned [here](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/issues/22). Hydra will be replacing the temporary `argparse` functionality in the [`inference_video.py`](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/1-Computer-Vision/1-1-Object-Detection-in-Urban-Environments/scripts/inference/inference_video.py) script as soon as the bug as been resolved.







insights.
## Tasks
### Exploratory Data Analysis (EDA)
* (Completed) Analyzed and visualized sample data from the dataset.
* (Completed) Computed the distribution of object labels across the dataset.
* (Completed) Analyzed object occlusions and verified annotation completeness.
* (Completed) Utilized appropriate data augmentation techniques based on EDA 

### Model Training and Evaluation
* (Completed) Configured the training pipeline using the TensorFlow Object Detection API's `pipeline.config` file.
* (Completed) Trained a convolutional neural network for multi-class object detection.
* (Completed) Monitored TensorBoard to track training metrics and performance trends.
* (Completed) Evaluated the model’s performance on object classification tasks.

### Extension of Tasks and Future Work
* Experiment with different hyperparameter settings to improve results.
* Plan to revisit and refine augmentation strategies for further performance gains.


