#!/bin/bash

set -e

mkdir weights -p
wget -O weights/shape_predictor_5_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
wget -O weights/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d weights/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d weights/shape_predictor_5_face_landmarks.dat.bz2
rm weights/weights/shape_predictor_5_face_landmarks.dat.bz2
rm weights/shape_predictor_68_face_landmarks.dat.bz2

