# Face Detection and Recognition

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-contrib-green?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-array%20ops-013243?logo=numpy&logoColor=white)
![Haar Cascade](https://img.shields.io/badge/Haar%20Cascade-face%20detection-orange)
![LBPH](https://img.shields.io/badge/LBPH-face%20recognition-purple)

## Introduction

This repository demonstrates a complete, beginner-friendly pipeline for **face detection** and **face recognition** using OpenCV. It's split into two stages:

1. **Face Detection** — using Haar Cascade classifiers to locate faces within an image.
2. **Face Recognition** — using the LBPH (Local Binary Patterns Histogram) algorithm to identify *who* the detected face belongs to.

As a working example, the model is trained on images of three cricketers — **Jasprit Bumrah**, **Rohit Sharma**, and **Virat Kohli** — and learns to correctly label new, unseen photos of them.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Face Detection](#face-detection)
- [Face Recognition](#face-recognition)
  - [Training](#1-training-the-model)
  - [Predicting](#2-running-the-recognizer)
- [Results](#results)
- [Notes](#notes)
- [Project Structure](#project-structure)

## Setup

1. Clone this repository.
2. Download the Haar Cascade XML file for frontal face detection from OpenCV's official repo and place it in the project root as `haar_face.xml`:
   https://github.com/opencv/opencv/tree/4.x/data/haarcascades
3. Install the required dependency:
```bash
   pip install opencv-contrib-python numpy
```
   > **Note:** Use `opencv-contrib-python` (not plain `opencv-python`), since the LBPH face recognizer (`cv.face.LBPHFaceRecognizer_create()`) lives in the `contrib` module.

## Face Detection

Run `Face_detection.py` to detect faces in a sample image using the Haar Cascade classifier.

**Input image:**

<img width="401" height="256" alt="input image" src="https://github.com/user-attachments/assets/216df8b8-3474-452e-a2bd-05ba10d190f0" />

**Steps performed:**
1. Load the Haar Cascade classifier:
```python
   haar_cascade = cv.CascadeClassifier('haar_face.xml')
```
2. Detect faces using `detectMultiScale()`:
```python
   haar_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
```

**Output:**

<img width="401" height="255" alt="detected faces" src="https://github.com/user-attachments/assets/162e5fd6-76e4-43ac-ab4a-0d44f91d9023" />

As shown above, Haar Cascades don't always detect every face correctly — this is a known limitation of the algorithm on complex or crowded images. For more reliable recognition, this project moves on to **LBPH (Local Binary Pattern Histogram)** face recognition below.

## Face Recognition

### 1. Training the Model

Run `faces_train.py` to train the recognizer on the `faces/train/` dataset:

1. Read each image and convert it to grayscale.
2. Detect faces using the Haar Cascade `detectMultiScale()` method.
3. Crop out the detected face region (Region of Interest / ROI).
4. Store each face ROI in a `features` array and its corresponding person label in a `labels` array.
5. Train an LBPH recognizer:
```python
   face_recognizer = cv.face.LBPHFaceRecognizer_create()
   face_recognizer.train(features, labels)
```
6. Save the trained model, features, and labels to disk (`face_trained.yml`, `features.npy`, `labels.npy`).

### 2. Running the Recognizer

Run `face_recognition.py` to test the trained model on a new, unseen image:

1. Load the trained LBPH model:
```python
   face_recognizer = cv.face.LBPHFaceRecognizer_create()
   face_recognizer.read('face_trained.yml')
```
2. Detect the face in the new image using the Haar Cascade.
3. Predict the identity using:
```python
   label, confidence = face_recognizer.predict(face_roi)
```
4. Draw the predicted name and bounding box on the image.

## Results

**Test image:**

<img width="500" height="500" alt="test image" src="https://github.com/user-attachments/assets/2d05b534-ceaa-4856-a928-aca13b98bad0" />

**Prediction result:**

<img width="500" height="500" alt="recognized face" src="https://github.com/user-attachments/assets/b2726a92-1ff2-4e4d-ab72-2952a0295c6f" />

The model correctly identifies the person as **Virat Kohli** 

## Notes

- Always make sure image paths include the file **extension** (e.g., `.jpg`, `.png`). `cv.imread()` fails silently (returns `None`) if the extension is missing or the path is wrong, which leads to errors like `!_src.empty()` in `cvtColor`.
- If your project folder is inside a cloud-synced directory (e.g., OneDrive), ensure files are fully downloaded locally — cloud-only files can fail to load.
- Recognition accuracy depends heavily on dataset quality (lighting, angle, number of training images per person).

## Project Structure

| Path | Description |
|---|---|
| `faces/train/jasprith_bumrah/` | Training images of Jasprit Bumrah |
| `faces/train/rohit_sharma/` | Training images of Rohit Sharma |
| `faces/train/virat_kohli/` | Training images of Virat Kohli |
| `faces/val/jasprith_bumrah/` | Validation images of Jasprit Bumrah |
| `faces/val/rohit_sharma/` | Validation images of Rohit Sharma |
| `faces/val/virat_kohli/` | Validation images of Virat Kohli |
| `photos/` | Sample images used for basic face detection |
| `Face_detection.py` | Basic Haar cascade face detection script |
| `faces_train.py` | Trains the LBPH recognizer on the dataset |
| `face_recognition.py` | Runs the trained recognizer on a new image |
| `haar_face.xml` | Haar cascade classifier for face detection |
| `features.npy` | Extracted face ROIs *(generated after training)* |
| `labels.npy` | Corresponding labels *(generated after training)* |
| `face_trained.yml` | Saved/trained LBPH model *(generated after training)* |

**Folder overview:**

- Face_Detection_and_recognition
  - faces
    - train
      - jasprith_bumrah
      - rohit_sharma
      - virat_kohli
    - val
      - jasprith_bumrah
      - rohit_sharma
      - virat_kohli
  - photos
  - Face_detection.py
  - faces_train.py
  - face_recognition.py
  - haar_face.xml
  - features.npy
  - labels.npy
  - face_trained.yml

> `train/` contains images used to train the recognizer.
>  `val/` contains unseen images used to test the trained model.
> `features.npy`, `labels.npy`, and `face_trained.yml` are generated automatically after running `faces_train.py` — you don't need to create these manually.
