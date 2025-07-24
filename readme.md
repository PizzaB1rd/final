# Mushroom Identifier
 
 This is a AI model to classify mushrooms show what kind of mushroom they are and how certain it is that variety

![add image descrition here](direct image link here)

## The Algorithm

The AI model relies on a bunch of images of mushrooms to validate if the requested image is a type of mushroom in its training data and will use resnet18 to access the data and to run the program on the image and test it with the imagenet program to return what type of mushroom it is and how sure it is that type of mushroom in percentage

## Running this project

1.#!/bin/bash
curl -L -o ~/Downloads/mushrooms-images-classification-215.zip\
  https://www.kaggle.com/api/v1/datasets/download/daniilonishchenko/mushrooms-images-classification-215
2. cd jetson-inference/python/training/classification/data
3. unzip ~/Downloads/mushrooms-images-classification.zip -d wildcats
4. cd ~/jetson-inference/
./docker/run.sh
5. cd python/training/classification
6. python3 train.py --model-dir=models/mushroom_data data/mushroom_data
7. python3 onnx_export.py --model-dir=models/mushroom_data
8. NET=models/mushroom_data
9. DATASET=data/mushroom_data
10. imagenet.py \
  --model=$NET/resnet18.onnx \
  --labels=$DATASET/labels.txt \
  --input_blob=input_0 \
  --output_blob=output_0 \
  $DATASET/test/almond_mushroom/am1.png $DATASET/output.png

[View a video explanation here](video link)
