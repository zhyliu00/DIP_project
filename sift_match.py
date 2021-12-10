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
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == "__main__":


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
    SIFT = cv2.SIFT_create()

    for rooot, diiirs, face_files in os.walk(face_dir):
        for face_file in face_files:
            if(not face_file.endswith('jpg')):
                continue

            ## take SIFT feature from real landmark

            face_path = os.path.join(face_dir,face_file)
            img_face = cv2.imread(face_path)
            img_face_base = img_face
            print(img_face.shape)
            height = img_face.shape[0]
            width = img_face.shape[1]
            face_color = img_face[5,5,:]
            gray = cv2.cvtColor(img_face,cv2.COLOR_BGR2GRAY)
            kp_real, des_real = SIFT.detectAndCompute(img_face,None)
            img_face_keypoint = cv2.drawKeypoints(gray, kp_real, None)
            # cv2.imshow('key_real', img_face_keypoint)
            # # Print
            # img2 = cv2.drawKeypoints(gray,kp,img_face)
            # cv2.imshow('feature',img2)
            # cv2.waitKey(0)
            res_best = 0.0
            manga_best = None
            match_name = None
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if(file.endswith('.png')):

                        ##  take SIFT feature of manga landmark
                        file_path = os.path.join(base_dir,file)
                        # img_landmark = dlib.load_rgb_image(file_path)
                        img_manga = cv2.imread(os.path.join(base_dir,file))
                        img_manga_base = copy.deepcopy(img_manga)
                        img_manga = img_manga[350:350+height,300:300+width,:]
                        for x in range(height):
                            for y in range(width):
                                if(img_manga[x,y,:]==[102, 102, 102]).all():
                                    img_manga[x,y,:]=face_color

                        res = calculate_ssim(img_face,img_manga)


                        gray_mangga = cv2.cvtColor(img_manga,cv2.COLOR_BGR2GRAY)

                        # cv2.imshow('rgb_manga',img_manga)
                        kp_manga,des_manga = SIFT.detectAndCompute(img_face,None)

                        img = cv2.drawKeypoints(img_manga, kp_manga, None)
                        # cv2.imshow('key_manga', img)
                        # cv2.waitKey(0)

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

                        # cv2.imshow('match', img_out)
                        # print(res)
                        # cv2.waitKey(0)
                        if(res_best<res):
                            res_best=res
                            manga_best=img_manga_base
                            match_name = file
            if(not os.path.exists(match_dir)):
                os.makedirs(match_dir)
            print(match_dir)
            real_path = os.path.join(match_dir,face_file)
            manga_path = os.path.join(match_dir,face_file + "_" + match_name)
            cv2.imwrite(real_path,img_face_base)
            cv2.imwrite(manga_path,manga_best)


            # break
        # dlib.load_rgb_image(f)
