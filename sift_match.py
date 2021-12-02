from pathlib import Path
import cv2
import sys
import os
import dlib
import argparse
import glob
import numpy as np


def get_rectangle(shape, range_left, range_right, margin):
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


parser = argparse.ArgumentParser()
parser.add_argument('--target_landmark',default='eye',type=str)
args  = parser.parse_args()

if __name__ == "__main__":


    ## 需要匹配的
    target_landmark = args.target_landmark

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    faces_folder_path = ''


    base_dir = './FYM-Images/{}'.format(target_landmark)
    face_dir = './neutral_front_split/{}'.format(target_landmark)
    # print(img.shape)
    SIFT = cv2.SIFT_create()

    for rooot, diiirs, face_files in os.walk(face_dir):
        for face_file in face_files:
            if(not face_file.endswith('jpg')):
                continue

            ## take SIFT feature from real landmark

            face_path = os.path.join(face_dir,face_file)
            img_face = cv2.imread(face_path)
            gray = cv2.cvtColor(img_face,cv2.COLOR_BGR2GRAY)
            kp_real, des_real = SIFT.detectAndCompute(img_face,None)
            img_face_keypoint = cv2.drawKeypoints(gray, kp_real, None)
            cv2.imshow('key_real', img_face_keypoint)
            # # Print
            # img2 = cv2.drawKeypoints(gray,kp,img_face)
            # cv2.imshow('feature',img2)
            # cv2.waitKey(0)
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if(file.endswith('.png')):

                        ##  take SIFT feature of manga landmark
                        file_path = os.path.join(base_dir,file)
                        # img_landmark = dlib.load_rgb_image(file_path)
                        img_manga = cv2.imread(os.path.join(base_dir,file))
                        img_manga = img_manga[300:500,300:600,:]
                        gray_mangga = cv2.cvtColor(img_manga,cv2.COLOR_BGR2GRAY)

                        # cv2.imshow('rgb_manga',img_manga)
                        kp_manga,des_manga = SIFT.detectAndCompute(img_face,None)

                        img = cv2.drawKeypoints(gray_mangga, kp_manga, None)
                        cv2.imshow('key_manga', img)
                        cv2.waitKey(0)

                        ##  do match
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
                        search_params = dict(checks=50)


                        flann = cv2.FlannBasedMatcher(index_params,search_params)
                        matches = flann.knnMatch(des_real,des_manga,k=2)
                        goodMatch = []
                        for m,n in matches:
                            if(m.distance < 0.5*n.distance):
                                goodMatch.append(m)
                        goodMatch = np.expand_dims(goodMatch,1)

                        img_out = cv2.drawMatchesKnn(img_face,kp_real,img_manga,kp_manga,goodMatch[:15],None,flags=2)

                        cv2.imshow('match', img_out)
                        cv2.waitKey(0)


            # break
        # dlib.load_rgb_image(f)
