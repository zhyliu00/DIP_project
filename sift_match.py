import copy
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

def match_img(img_face,img_manga,descriptor='SIFT'):
    if(descriptor == 'ORB'):
        des1 = cv2.ORB_create(nfeatures=2000)
        des2 = cv2.ORB_create(nfeatures=300)
    elif(descriptor =='SIFT'):
        des1 = cv2.SIFT_create(nfeatures=2000)
        des2 = cv2.SIFT_create(nfeatures=300)
    else:
        raise NotImplementedError("not implemented descriptor {}".format(descriptor))
    height = img_face.shape[0]
    width = img_face.shape[1]
    face_color = img_face[5, 5, :]
    gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    kp_real, des_real = des1.detectAndCompute(img_face, None)
    img_face_keypoint = cv2.drawKeypoints(gray, kp_real, None)
    if (show):
        cv2.imshow('key_real', img_face_keypoint)
    img_manga = img_manga[350:350 + height, 300:300 + width, :]
    for x in range(height):
        for y in range(width):
            if (img_manga[x, y, :] == [102, 102, 102]).all():
                img_manga[x, y, :] = face_color

    gray_mangga = cv2.cvtColor(img_manga, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('rgb_manga',img_manga)
    kp_manga, des_manga = des2.detectAndCompute(gray_mangga, None)

    img = cv2.drawKeypoints(img_manga, kp_manga, None)
    if (show):
        cv2.imshow('key_manga', img)
        cv2.waitKey(0)

    ##  do match
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(np.asarray(des_real, np.float32), np.asarray(des_manga, np.float32), k=2)
    except Exception as e:
        print(e)
        return 0
    # print(len(matches))
    goodMatch = []

    ## distance很重要
    for m, n in matches:
        if (m.distance < 0.75 * n.distance):
            goodMatch.append(m)
    goodMatch = np.expand_dims(goodMatch, 1)
    res = len(goodMatch)
    # print(res)

    img_out = cv2.drawMatchesKnn(img_face, kp_real, img_manga, kp_manga, goodMatch[:], None, flags=2)
    if (show):
        cv2.imshow('match', img_out)
        cv2.waitKey(0)
    return res



if __name__ == "__main__":

    show = False
    ## 需要匹配的
    target_landmark = args.target_landmark

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    faces_folder_path = ''


    base_dir = './FYM-Images/{}'.format(target_landmark)
    face_dir = './neutral_front_split/{}'.format(target_landmark)
    match_dir = './match_{}/'.format(target_landmark)
    # print(img.shape)
    # SIFT = cv2.ORB_create(nfeatures=2000)

    for rooot, diiirs, face_files in os.walk(face_dir):
        for face_file in face_files:
            if(not face_file.endswith('jpg')):
                continue

            ## take SIFT feature from real landmark

            face_path = os.path.join(face_dir,face_file)
            img_face = cv2.imread(face_path)
            img_face_base = img_face

            # # Print
            # img2 = cv2.drawKeypoints(gray,kp,img_face)
            res_best = 0.0
            manga_best = None
            match_name = None
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if(not file.endswith('.png')):
                        continue

                    ##  take SIFT feature of manga landmark
                    file_path = os.path.join(base_dir,file)
                    # img_landmark = dlib.load_rgb_image(file_path)
                    img_manga = cv2.imread(os.path.join(base_dir,file))
                    img_manga_base = copy.deepcopy(img_manga)
                    descriptors=['SIFT','ORB']
                    res = 0
                    for des in descriptors:
                        res += match_img(img_face,img_manga,des)

                    if(res_best<res):
                        res_best=res
                        manga_best=img_manga_base
                        match_name = file
            if(not os.path.exists(match_dir)):
                os.makedirs(match_dir)
            # print(match_dir)
            print(res_best)
            if(res_best == 0):
                continue
            real_path = os.path.join(match_dir,face_file)
            manga_path = os.path.join(match_dir,face_file.rstrip('.jpg') + "_" + match_name)
            cv2.imwrite(real_path,img_face_base)
            cv2.imwrite(manga_path,manga_best)


            # break
        # dlib.load_rgb_image(f)
