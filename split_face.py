#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#      300 faces In-the-wild challenge: Database and results.
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy
from pathlib import Path
import cv2
import sys
import os
import dlib
import glob
import numpy as np

landmarks = {
'eyes' : [36,48,30],
'eyebrow'  : [17,27,10],
'nose' : [27,36,20],
'mouse' : [48,68,30],
'face': [0,17,10]
}



def get_rectangle(shape,range_left,range_right,margin):
    eyes_range = list(range(range_left, range_right))
    eyes_shape = [[1000, 1000], [-1, -1]]
    eyes_cords = []

    for k in eyes_range:
        eyes_cords.append([shape.part(k).x, shape.part(k).y])
    eyes_cords = np.array(eyes_cords)
    eyes_shape[0][0] = eyes_cords[:, 0].min() - margin
    eyes_shape[0][1] = eyes_cords[:, 1].min() - margin
    eyes_shape[1][0] = eyes_cords[:, 0].max() + margin
    eyes_shape[1][1] = eyes_cords[:, 1].max() + margin


    return eyes_shape


if __name__ == "__main__":
    predictor_path = './shape_predictor_68_face_landmarks.dat'
    faces_folder_path = './neutral_front'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    win = dlib.image_window()


    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))

        # dlib (col, row, RGB)
        # cv2 (row, col, BGR)
        img = dlib.load_rgb_image(f)
        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            for landmark,range_ in landmarks.items():
            # landmark = 'face'
            # range_ = landmarks[landmark]
                #############
                landmark_shape = get_rectangle(shape,range_[0],range_[1],range_[2])
                landmark_shape_rectangle = dlib.rectangle(left=landmark_shape[0][0],top = landmark_shape[0][1],right=landmark_shape[1][0],bottom=landmark_shape[1][1])
                win.add_overlay(landmark_shape_rectangle)
                landmark_img = img[landmark_shape_rectangle.top():landmark_shape_rectangle.bottom(),landmark_shape_rectangle.left():landmark_shape_rectangle.right()]
                # cv2.rectangle(img,(eyes_shape_rectangle.left(),eyes_shape_rectangle.top()),(eyes_shape_rectangle.right(),eyes_shape_rectangle.bottom()),(0,0,255),2)
                #
                # print(eyes_img.shape)
                landmark_img = cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB)
                landmark_path = Path('./neutral_front_split/{}/'.format(landmark))
                if(not os.path.exists(landmark_path)):
                    os.makedirs(landmark_path)
                landmark_path = landmark_path / os.path.split(f)[1]
                cv2.imwrite(landmark_path.__str__(),landmark_img)
                print(landmark_path)
            #######################

            # #############
            # # # For face
            # face_cord = []
            # range_ = range(landmarks['face'][0],landmarks['face'][1])
            # for i in range_:
            #     face_cord.append(shape.part(i))
            # print(face_cord)

            # Draw the face landmarks on the screen.
            # win.add_overlay(shape)

        win.add_overlay(dets)
        # dlib.hit_enter_to_continue()
        # break
