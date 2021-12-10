
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
