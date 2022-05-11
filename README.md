[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<p>
  <a href="https://github.com/unsignedrant/yolo-tf2/">
  </a>

  <h3 align="center">Yolo (all versions) Real Time Object Detector in tensorflow 2.x</h3>
    .
    <a href="https://github.com/unsignedrant/yolo-tf2/issues">Report Bug</a>
    ·
    <a href="https://github.com/unsignedrant/yolo-tf2/issues">Request Feature</a>
  </p>
  
<!-- TODO -->
## **TODO**

* [ ] Transfer learning
* [x] YoloV4 configuration
* [x] YoloV4 training
* [ ] YoloV4 loss function adjustments.
* [ ] Live plot losses
* [x] Command line options
* [x] YoloV3 tiny
* [ ] Rasberry Pi support


<!-- TABLE OF CONTENTS -->
## **Table of Contents**

* [Getting Started](#getting-started)
  * [Installation](#installation)

* [Description](#description)

* [Features](#features)
  * [Command line options](#command-line-options)
    * [General](#general-options)
    * [Training](#training-options)
    * [Detection](#detection-options)
  * [Conversion from DarkNet .cfg files to keras models](#conversion-from-darknet-cfg-files-to-keras-models)
  * [Support for all yolo versions](#all-yolo-versions-are-supported)
  * [Tensorflow-2.x](#tensorflow-2x)
  * [Random weights and DarkNet weights support](#random-weights-and-darknet-weights-support)
  * [Multiple input options](#multiple-input-options)
    * [.csv](#csv-file)
    * [.xml](#xml-files-in-voc-format)
    * [.tfrecord](#tfrecord-files)
  * [Anchor generation](#anchor-generation)
  * [Visualization of training stages](#visualization-of-training-stages)
  * [mAP evaluation in pandas](#map-evaluation)
  * [labelpix support](#labelpix-support)
  * [Photo & video detection](#photo--video-detection)

* [Usage](#usage)
  * [Training](#training)
    * [Code example](#training-code-example)
    * [Command line example](#training-command-line-example)
  * [Detection](#detection)
    * [Code example](#detection-code-example)
    * [Command line example](#detection-command-line-example)
  * [Evaluation](#evaluation)

* [Contributing](#contributing)
* [Issues](#issue-policy)
  * [What will be addressed](#relevant-issues)
  * [What will not be adressed](#irrelevant-issues)
    
* [License](#license)
* [Show your support](#show-your-support)

![GitHub Logo](/assets/detections.png)

<!-- GETTING STARTED -->
## **Getting started**

### **Installation**

```sh
pip install git+https://github.com/unsignedrant/yolo-tf2
```
**Verify installation**

```sh
% yolotf2
Yolo-tf2 1.0

Usage:
	yolotf2 <command> [options] [args]

Available commands:
	train      Create new or use existing dataset and train a model
	detect     Detect a folder of images or a video

Use yolotf2 <command> -h to see more info about a command

Use yolotf2 -h to display all command line options
```

<!-- DESCRIPTION -->
### **Description**

yolo-tf2 was initially an implementation of [yolov3](https://pjreddie.com/darknet/yolo/) 
(you only look once)(training & inference) and support for all yolo versions was added in 
[db2f889](https://github.com/unsignedrant/yolo-tf2/commit/db2f8898233bc855929a9207996ea926568751fc). 
Yolo is a state-of-the-art, real-time object detection system that is extremely 
fast and accurate. The official repo is [here](https://github.com/AlexeyAB/darknet).
There are many implementations that support tensorflow, only a few that 
support tensorflow v2 and as I did not find versions that suit my needs so, 
I decided to create this version which is very flexible and customizable. 
It requires python 3.10+, is not platform specific and is MIT licensed.

<!-- FEATURES -->

## **Features**

### **Command line options**

#### **General options**

| flags             | help                                                      | required   | default       |
|:------------------|:----------------------------------------------------------|:-----------|:--------------|
| --anchors         | Path to anchors .txt file                                 | True       | -             |
| --batch-size      | Training/detection batch size                             | -          | 8             |
| --classes         | Path to classes .txt file                                 | True       | -             |
| --input-shape     | Input shape ex: (m, m, c)                                 | -          | (416, 416, 3) |
| --iou-threshold   | iou (intersection over union) threshold                   | -          | 0.5           |
| --masks           | Path to masks .txt file                                   | True       | -             |
| --max-boxes       | Maximum boxes per image                                   | -          | 100           |
| --model-cfg       | Yolo DarkNet configuration .cfg file                      | True       | -             |
| --quiet           | If specified, verbosity is set to False                   | -          | -             |
| --score-threshold | Confidence score threshold                                | -          | 0.5           |
| --v4              | If yolov4 configuration is used, this should be specified | -          | -             |

#### **Training options**

| flags                 | help                                                                           | default   |
|:----------------------|:-------------------------------------------------------------------------------|:----------|
| --dataset-name        | Checkpoint/dataset prefix                                                      | -         |
| --delete-images       | If specified, dataset images will be deleted upon being saved to tfrecord.     | -         |
| --epochs              | Number of training epochs                                                      | 100       |
| --es-patience         | Early stopping patience                                                        | -         |
| --image-dir           | Path to folder containing images referenced by .xml labels                     | -         |
| --labeled-examples    | Path to labels .csv file                                                       | -         |
| --learning-rate       | Training learning rate                                                         | 0.001     |
| --output-dir          | Path to folder where training dataset / checkpoints / other data will be saved | .         |
| --shuffle-buffer-size | Dataset shuffle buffer                                                         | 512       |
| --train-shards        | Total number of .tfrecord files to split training dataset into                 | 1         |
| --train-tfrecord      | Path to training .tfrecord file                                                | -         |
| --valid-frac          | Validation dataset fraction                                                    | 0.1       |
| --valid-shards        | Total number of .tfrecord files to split validation dataset into               | 1         |
| --valid-tfrecord      | Path to validation .tfrecord file                                              | -         |
| --weights             | Path to trained weights .tf or .weights file                                   | -         |
| --xml-dir             | Path to folder containing .xml labels in VOC format                            | -         |

#### **Detection options**

| flags                 | help                                                                                               | required   | default   |
|:----------------------|:---------------------------------------------------------------------------------------------------|:-----------|:----------|
| --codec               | Codec to use for predicting videos                                                                 | -          | mp4v      |
| --display-vid         | Display video during detection                                                                     | -          | -         |
| --evaluation-examples | Path to .csv file with ground truth for evaluation of the trained model and mAP score calculation. | -          | -         |
| --image-dir           | A directory that contains images to predict                                                        | -          | -         |
| --images              | Paths of images to detect                                                                          | -          | -         |
| --output-dir          | Path to directory for saving results                                                               | -          | -         |
| --video               | Path to video to predict                                                                           | -          | -         |
| --weights             | Path to trained weights .tf or .weights file                                                       | True       | -         |

### **Conversion from DarkNet .cfg files to keras models**
This feature was introduced to replace the old hard-coded model.
Models are loaded directly from DarkNet .cfg files for convenience.

### **All yolo versions are supported**
As of [db2f889](https://github.com/unsignedrant/yolo-tf2/commit/db2f8898233bc855929a9207996ea926568751fc)
DarkNet .cfg files are automatically converted to keras models.

### **tensorflow 2.x**

The current code leverages features that were introduced in tensorflow 2.x 
including keras models, tfrecord datasets, etc...

### **Random weights and DarkNet weights support**

Both options are available, and **Note** in case of using DarkNet [weights](https://pjreddie.com/media/files/yolov3.weights)
you must maintain the same number of [COCO classes](https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda) (80 classes)
as transfer learning to models with different classes is not currently supported.

### **Multiple input options**

There are 3 input options accepted by the api:

#### **.csv file**

A .csv file similar to the one below is supported. **Note** that `x0`, `y0`, `x1`, `y1`
are x and y coordinates relative to their corresponding image width and height. For
example: 

    image width = 1000
    image height = 500 
    x0, y0 = 100, 300
    x1, y1 = 120, 320
    x0, y0, x1, y1 = 0.1, 0.6, 0.12, 0.64 respectively. 

| image            | object_name | object_index |       x0 |       y0 |       x1 |       y1 |
|:-----------------|:------------|-------------:|---------:|---------:|---------:|---------:|
| /path/to/368.jpg | Car         |            0 | 0.478423 |  0.57672 | 0.558036 | 0.699735 |
| /path/to/368.jpg | Car         |            0 | 0.540923 | 0.583333 | 0.574405 | 0.626984 |
| /path/to/368.jpg | Car         |            0 | 0.389881 | 0.574074 | 0.470982 | 0.683862 |
| /path/to/368.jpg | Car         |            0 | 0.447173 | 0.555556 | 0.497024 | 0.638889 |
| /path/to/368.jpg | Street Sign |            1 | 0.946429 |  0.40873 | 0.991815 | 0.510582 |


#### **.xml files in VOC format**

```xml
<annotation>
	<folder>VOC2012</folder>
	<filename>2007_000033.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>500</width>
		<height>366</height>
		<depth>3</depth>
	</size>
	<segmented>1</segmented>
	<object>
		<name>aeroplane</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>9</xmin>
			<ymin>107</ymin>
			<xmax>499</xmax>
			<ymax>263</ymax>
		</bndbox>
	</object>
	<object>
		<name>aeroplane</name>
		<pose>Left</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>421</xmin>
			<ymin>200</ymin>
			<xmax>482</xmax>
			<ymax>226</ymax>
		</bndbox>
	</object>
	<object>
		<name>aeroplane</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>325</xmin>
			<ymin>188</ymin>
			<xmax>411</xmax>
			<ymax>223</ymax>
		</bndbox>
	</object>
</annotation>

```

#### **.tfrecord files**

.tfrecord files previously generated by the code can be reused.
A typical feature map looks like:

    {
        'image': tf.io.FixedLenFeature([], tf.string),
        'x0': tf.io.VarLenFeature(tf.float32),
        'y0': tf.io.VarLenFeature(tf.float32),
        'x1': tf.io.VarLenFeature(tf.float32),
        'y1': tf.io.VarLenFeature(tf.float32),
        'object_name': tf.io.VarLenFeature(tf.string),
        'object_index': tf.io.VarLenFeature(tf.int64),
    }

### **Anchor generation**

A [k-means](https://en.wikipedia.org/wiki/K-means_clustering) algorithm finds the optimal sizes and generates 
anchors with process visualization.

### **Visualization of training stages**

**Including:**

* **k-means visualization:**

![GitHub Logo](/assets/anchors.png)

* **Generated anchors:**

![GitHub Logo](/assets/anchors_sample.png)

* **Precision and recall curves:**

![GitHub Logo](/assets/pr.png)

* **Evaluation bar charts:**

![GitHub Logo](/assets/map.png)

* **Actual vs. detections:**

![GitHub Logo](/assets/true_false.png)

You can always visualize different stages of the program using my other repo 
[labelpix](https://github.com/unsignedrant/labelpix) which is a tool for drawing 
bounding boxes, but can also be used to visualize bounding boxes over images using 
csv files in the format mentioned [here](#csv-file).

### **mAP evaluation**

Evaluation is available through the detection api which supports mAP score calculation.
A typical evaluation result looks like:

|     | object_name    | average_precision | actual | detections | true_positives | false_positives | combined |
|----:|:---------------|------------------:|-------:|-----------:|---------------:|----------------:|---------:|
|   1 | Car            |          0.825907 |    298 |        338 |            275 |              63 |      338 |
|  12 | Bus            |          0.666667 |      3 |          2 |              2 |               0 |        2 |
|   6 | Palm Tree      |          0.627774 |    122 |         93 |             82 |              11 |       93 |
|   7 | Trash Can      |          0.555556 |      9 |          7 |              5 |               2 |        7 |
|   8 | Flag           |          0.480867 |     14 |          8 |              7 |               1 |        8 |
|   2 | Traffic Lights |          0.296155 |    122 |         87 |             58 |              29 |       87 |
|   5 | Street Lamp    |          0.289578 |     73 |         41 |             28 |              13 |       41 |
|   3 | Street Sign    |          0.287331 |     93 |         52 |             35 |              17 |       52 |
|   9 | Fire Hydrant   |          0.194444 |      6 |          3 |              2 |               1 |        3 |
|   4 | Pedestrian     |          0.183942 |    130 |         56 |             35 |              21 |       56 |
|   0 | Delivery Truck |                 0 |      1 |          0 |              0 |               0 |        0 |
|  10 | Road Block     |                 0 |      2 |          7 |              0 |               7 |        7 |
|  11 | Minivan        |                 0 |      3 |          0 |              0 |               0 |        0 |
|  13 | Bicycle        |                 0 |      4 |          1 |              0 |               1 |        1 |
|  14 | Pickup Truck   |                 0 |      2 |          0 |              0 |               0 |        0 |

### **labelpix support**

You can check my other repo [labelpix](https://github.com/unsignedrant/labelpix) which is a
labeling tool that you can use produce small datasets for experimentation. It supports .csv files
in the format mentioned [here](#csv-file) and/or .xml files as [here](#xml-files-in-voc-format)

### **Photo & video detection**

Detections can be performed on photos or videos using the detection api.

## **Usage**

### **Training**

The following files are expected:

* Object classes .txt file.

      person
      bicycle
      car
      motorbike
      aeroplane
      bus
      train
      truck
      boat
      traffic light
      fire hydrant
* DarkNet model .cfg [file](https://github.com/pjreddie/darknet/tree/master/cfg)
* Anchors .txt file

      10,13
      16,30
      33,23
      30,61
      62,45
      59,119
      116,90
      156,198
      373,326
* Masks .txt file

      6,7,8
      3,4,5
      0,1,2
* Labeled examples, **ONE** of:
  * .csv file as shown [here](#csv-file)
  * .xml labels + image folder as shown [here](#xml-files-in-voc-format)
  * Training + Validation .tfrecord files, having a feature map as shown [here](#tfrecord-files)


#### **Training code example**

Training is available through [yolo_tf2.train](/yolo_tf2/api.py) api. For more info about
other parameters, check the docstrings, available through `help()`

    import yolo_tf2
    
    yolo_tf2.train(
        input_shape=(608, 608, 3),
        classes='/path/to/classes.txt',
        model_cfg='/path/to/darknet/file.cfg',
        anchors='/path/to/anchors.txt',
        masks='/path/to/masks.txt',
        labeled_examples='/path/to/labeled_examples.csv',
        output_dir='/path/to/training-output-dir'
    )

#### **Training command line example**

    yolotf2 train --input-shape 608 608 3 --classes /path/to/classes.txt --model-cfg /path/to/darknet/file.cfg --anchors /path/to/anchors.txt --masks /path/to/masks.txt --labeled-examples /path/to/labeled_examples.csv --output-dir /path/to/training-output-dir  

### **Detection**

The following files are expected:

* Object classes .txt file.

      person
      bicycle
      car
      motorbike
      aeroplane
      bus
      train
      truck
      boat
      traffic light
      fire hydrant
* DarkNet model .cfg [file](https://github.com/pjreddie/darknet/tree/master/cfg)
* Anchors .txt file

      10,13
      16,30
      33,23
      30,61
      62,45
      59,119
      116,90
      156,198
      373,326
* Masks .txt file

      6,7,8
      3,4,5
      0,1,2
* Trained .tf or .weights file
* Whatever is to detect: any of:
  * A list of image paths
  * Image dir
  * Video

**Note: For yolov4 configuration, `v4=True` or `--v4` should be specified**

#### **Detection code example**

Detection is available through [yolo_tf2.detect](/yolo_tf2/api.py) api. For more info about
other parameters, check the docstrings, available through `help()`

    import yolo_tf2
    
    yolo_tf2.detect(
        input_shape=(608, 608, 3),
        classes='/path/to/classes.txt',
        anchors='/path/to/anchors.txt',
        masks='/path/to/masks.txt',
        model_cfg='/path/to/darknet/file.cfg',
        weights='/path/to/trained_weights.tf',
        images=['/path/to/image1', '/path/to/image2', ...],
        output_dir='detection-output'
    )

#### **Detection command line example**

    yolotf2 detect --input-shape 608 608 3 --classes /path/to/classes.txt --model-cfg /path/to/darknet/file.cfg --anchors /path/to/anchors.txt --masks /path/to/masks.txt --weights /path/to/trained_weights.tf --images /path/to/image1 /path/to/image2 --output-dir /path/to/detection-output-dir 

**Notes:** 

* To detect video, `video` or `--video` needs to be passed instead
* For yolov4 configuration, `v4=True` or `--v4` should be specified**

### **Evaluation**
 
Evaluation is available through the very same detection api described in the previous section.
The only difference is an additional parameter `evaluation_examples` or `--evaluation-examples`
for command line which is a .csv file containing the actual labels of the images being detected. 
The names of the images passed will be looked for in the actual labels, and if any of the
filenames were not found, an error is raised, which means:

if you do:

    import yolo_tf2
    
    yolo_tf2.detect(
        input_shape=(608, 608, 3),
        classes='/path/to/classes.txt',
        anchors='/path/to/anchors.txt',
        masks='/path/to/masks.txt',
        model_cfg='/path/to/darknet/file.cfg',
        weights='/path/to/trained_weights.tf',
        images=['/path/to/image1', '/path/to/image2', ...],
        output_dir='detection-output',
        evaluation_examples='/path/to/actual/examples'
    )

`evaluation_examples` .csv file should look like:

| image           | object_name   |   object_index |       x0 |       y0 |       x1 |       y1 |
|:----------------|:--------------|---------------:|---------:|---------:|---------:|---------:|
| /path/to/image1 | Car           |              0 | 0.478423 |  0.57672 | 0.558036 | 0.699735 |
| /path/to/image1 | Car           |              0 | 0.540923 | 0.583333 | 0.574405 | 0.626984 |
| /path/to/image1 | Car           |              0 | 0.389881 | 0.574074 | 0.470982 | 0.683862 |
| /path/to/image2 | Car           |              0 | 0.447173 | 0.555556 | 0.497024 | 0.638889 |
| /path/to/image2 | Street Sign   |              1 | 0.946429 |  0.40873 | 0.991815 | 0.510582 |

Because `images=['/path/to/image1', '/path/to/image2', ...]` were passed,
their actual labels must be provided. Same thing applies to the images contained in a directory
if `image_dir` was passed instead.

## **Contributing**

Contributions are what make the open source community such an amazing place to  
learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## **Issue policy**
There are relevant cases in which the issues will be addressed and irrelevant ones that will be closed.

### Relevant issues
The following issues will be addressed.
- Bugs.
- Performance issues.
- Installation issues.
- Documentation issues.
- Feature requests.
- Dependency issues that can be solved.

### Irrelevant issues
The following issues will not be addressed and will be closed.
- Issues without context / clear and concise explanation.
- Issues without standalone code (minimum reproducible example), or a jupyter notebook link to reproduce errors.
- Issues that are improperly formatted.
- Issues that are dataset / label specific without a dataset sample link.
- Issues that are the result of doing something that is unsupported by the existing features.
- Issues that are not considered as improvement / useful.

## **License**

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## **Show your support**

Give a ⭐️ if this project helped you!


[contributors-shield]: https://img.shields.io/github/contributors/unsignedrant/yolo-tf2?style=flat-square
[contributors-url]: https://github.com/unsignedrant/yolo-tf2/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/unsignedrant/yolo-tf2?style=flat-square
[forks-url]: https://github.com/unsignedrant/yolo-tf2/network/members
[stars-shield]: https://img.shields.io/github/stars/unsignedrant/yolo-tf2?style=flat-square
[stars-url]: https://github.com/unsignedrant/yolo-tf2/stargazers
[issues-shield]: https://img.shields.io/github/issues/unsignedrant/yolo-tf2?style=flat-square
[issues-url]: https://github.com/unsignedrant/yolo-tf2/issues
[license-shield]: https://img.shields.io/github/license/unsignedrant/yolo-tf2
[license-url]: https://github.com/unsignedrant/yolo-tf2/blob/master/LICENSE
