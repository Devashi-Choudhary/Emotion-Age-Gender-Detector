# Emotion-Age-Gender-Detector

Automatic emotion, age and gender classification has become relevant to an increasing amount of applications, particularly since the rise of social platforms and social media. In this project, face images of persons were trained using deep learning architecture, and we try to detect emotion, age and gender with high success rate. 

![image](https://github.com/Devashi-Choudhary/Emotion-Age-Gender-Detector/blob/master/ReadMe_Images/10.JPG)

The sample image from video.

[You-Tube Video](https://www.youtube.com/watch?v=OcFS5-RDlL0)

# High Level Overview of Emotion-Age-Gender-Detector

![overview](https://github.com/Devashi-Choudhary/Emotion-Age-Gender-Detector/blob/master/ReadMe_Images/o.png)

As shown above, the [Wide-Resnet Architecture](https://medium.com/@SeoJaeDuk/wide-residual-networks-with-interactive-code-5e190f8f25ec) is used for training of age-gender detection using faces and the [Convolutional Neural Network (CNN)](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148) is used for traing of emotion detection from facial expression. For testing, first the face is detected using [dlib](https://medium.com/data-science-blog/face-detection-with-python-and-dlib-ae599e73421c) and passed to trained model for prediction.


# Dataset Used

1. For training of emotion model, [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset is used. Fer2013 datset that contains 30,000 images of facial expressions grouped in seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.

![fer2013](https://github.com/Devashi-Choudhary/Emotion-Age-Gender-Detector/blob/master/ReadMe_Images/fer2013_sample.png)

2. For training of age-gender model [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset is used. This is the largest publicly available dataset of face images with gender and age labels. In total, it contains 460,723 face images from 20,284 celebrities.

![imdb](https://github.com/Devashi-Choudhary/Emotion-Age-Gender-Detector/blob/master/ReadMe_Images/Some-samples-in-the-IMDB-and-Wiki-datasets.png)

# Dependencies

Deep Learning based Emotion-Age-Gender-Detector architecture uses [OpenCV](https://opencv.org/) (opencv==4.2.0) and [Python](https://www.python.org/downloads/) (python==3.7). The deep learning model uses [Keras](https://keras.io/) (keras==2.3.1) on [Tensorflow](https://www.tensorflow.org/) (tensorflow>=1.15.2). For face detection in images it uses [dlib](https://pypi.org/project/dlib/). Also, imutils==0.5.3, numpy==1.18.2, argparse==1.1, pandas==0.23.4, and scipy==1.1.0 are also used.

# How to execute code:

1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, run the following command in your Terminal/Command Prompt to install the libraries required

> `pip install requirements.txt`

4. Get the dataset as mentioned below. 

For fer2013 dataset, you only need `fer2013.csv` file. 

For [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) or [wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) dataset you need `.mat` file. After getting .mat file, run the command to filter out noise data and serialize images and labels for training into .mat file.

> `python3 mat_file.py --output data/imdb_db.mat --db imdb --img_size 64`

5. **Training of Model :**

For training of emotion detector model using facial expressions, you need to run the following command on terminal:

> `python train_emotion_model.py --path fer2013.csv` 

where --path is path to fer2013 csv file.

For training of age-gender detector model using face, you need to run the following command on terminal:

> `python3 train_age_gender_model.py --input data/imdb_db.mat`

where --input is path to .mat file.

**Note :** Some part of code is taken from [here](https://github.com/yu4u/age-gender-estimation) for age-gender prediction. Please refer to it for more implementation details.

6. Now you need to have the data, use the image/video/webcam for evaluating the performance of trained models. Open terminal type the following commands:

For getting output in images, run the command:

> `python image_demo.py -i images/1.jpg -e models/emotion.hdf5 -ag models/age-gender.hdf5`

where -i is path to input image.

For getting output in video, run the command:

> `python video_demo.py -i videos/video.mp4 -e models/emotion.hdf5 -ag models/age-gender.hdf5`

where -v is path to input video.

For getting output in real time video stream, run the command:

> `python real_time_demo.py -e models/emotion.hdf5 -ag models/age-gender.hdf5`

where -e is path to trained emotiona model and -ag is path to trained age-gender model.

**Note :** For inference, you can download pretrained models from [here](https://drive.google.com/drive/u/0/folders/1tkGB-yaBjrdW2tSiKgo397L3FDnwJdAG) and place it inside `models/` directory. "For age-gender model it automatically downloaded for TensorFlow backend." from [original repo](https://github.com/yu4u/age-gender-estimation). The default values of emotion model and age-gender model are set. No need to give (-e, -ag) arguement while running the python file.

# Results

![results](https://github.com/Devashi-Choudhary/Emotion-Age-Gender-Detector/blob/master/ReadMe_Images/results.png)

# References 
1. [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. [IMDB or wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
3. [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
4. [age-gender-estimation](https://github.com/yu4u/age-gender-estimation)
5. [facial-expression-recognition-using-cnn](https://github.com/amineHorseman/facial-expression-recognition-using-cnn)
6. [dlib for face detection](https://medium.com/data-science-blog/face-detection-with-python-and-dlib-ae599e73421c)
