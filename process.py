import copy
import dlib
import cv2
import numpy as np
from get_hair import get_hair_rgb
from pathlib import Path
import os

show = False
def get_des(descriptor):
    if(descriptor == 'ORB'):
        return cv2.ORB_create(nfeatures=500,scaleFactor=1.1)
    elif(descriptor =='SIFT'):
        return cv2.SIFT_create(nfeatures=500)
    elif (descriptor == 'KAZE'):
        return cv2.KAZE_create()
    elif (descriptor == 'BRISK'):
        return cv2.BRISK_create()
    elif (descriptor == 'AKAZE'):
        return cv2.AKAZE_create()
    else:
        raise NotImplementedError("not implemented descriptor {}".format(descriptor))


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

def match_img(img_face,kp_real, des_real,img_manga,kp_manga, des_manga,descriptor='SIFT'):
    des2 = get_des(descriptor)
    face_color = img_face[5, 5, :]

    gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    # kp_real, des_real = des1.detectAndCompute(gray, None)
    img_face_keypoint = cv2.drawKeypoints(gray, kp_real, None)
    if (show):
        cv2.imshow('key_real_{}'.format(descriptor), img_face_keypoint)


    for x in range(img_manga.shape[0]):
        for y in range(img_manga.shape[1]):
            if (img_manga[x, y, :] == [102, 102, 102]).all():
                img_manga[x, y, :] = face_color

    # gray_mangga = cv2.cvtColor(img_manga, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('rgb_manga',img_manga)
    # kp_manga, des_manga = des2.detectAndCompute(gray_mangga, None)
    # print(len(kp_real),len(kp_manga))
    img = cv2.drawKeypoints(img_manga, kp_manga, None)
    if (show):
        cv2.imshow('{}'.format(descriptor), img)
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
        if (m.distance < 0.8 * n.distance):
            goodMatch.append(m)
    goodMatch = np.expand_dims(goodMatch, 1)
    res = len(goodMatch)
    # print(res)

    img_out = cv2.drawMatchesKnn(img_face, kp_real, img_manga, kp_manga, goodMatch[:], None, flags=2)
    if (show):
        cv2.imshow('match', img_out)
        cv2.waitKey(0)
    return res

def get_masked_skin(img):
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    # Get pointer to video frames from primary device
    image = img
    imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    skinYCrCb = cv2.bitwise_and(image, image, mask=skinRegionYCrCb)
    if(show):
        cv2.imshow('skin_detection',skinYCrCb)
        cv2.waitKey(0)

    skinRegionYCrCb = skinRegionYCrCb != 0
    masked_rgb = skinYCrCb[skinRegionYCrCb,:]
    res = np.mean(masked_rgb,axis=0)

    return res
if __name__ == "__main__":
    landmarks = {
        'eye' : [36,48,30],
        'eyebrow'  : [17,27,10],
        'nose' : [27,36,20],
        'mouse' : [48,68,30],
        'face': [0,17,10],
        'hair': None
    }
    ## get input
    img = cv2.imread('./neutral_front/002_03.jpg')
    results = {}
    target_dir = './good_match'
    if (not os.path.exists(target_dir)):
        os.makedirs(target_dir)


    predictor_path = './shape_predictor_68_face_landmarks.dat'
    faces_folder_path = './neutral_front'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(img, 1)

    skin_rgb = get_masked_skin(img)
    results['skin_rgb'] = skin_rgb





    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        for landmark,range_ in landmarks.items():
        # landmark = 'face'
        # range_ = landmarks[landmark]
            if(landmark != 'hair'):
                landmark_shape = get_rectangle(shape,range_[0],range_[1],range_[2])
                landmark_shape_rectangle = dlib.rectangle(left=landmark_shape[0][0],top = landmark_shape[0][1],right=landmark_shape[1][0],bottom=landmark_shape[1][1])
                landmark_img = img[landmark_shape_rectangle.top():landmark_shape_rectangle.bottom(),landmark_shape_rectangle.left():landmark_shape_rectangle.right()]
            #############
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hair_masked = get_hair_rgb(img_rgb)
                landmark_img = hair_masked
                landmark_img = cv2.cvtColor(landmark_img,cv2.COLOR_BGR2RGB)
        # cv2.rectangle(img,(eyes_shape_rectangle.left(),eyes_shape_rectangle.top()),(eyes_shape_rectangle.right(),eyes_shape_rectangle.bottom()),(0,0,255),2)
            #
            # print(eyes_img.shape)
            # landmark_img = cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB)
            # cv2.waitKey(0)

            img_face = landmark_img
            img_face_base = img_face
            gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            # descriptors=['SIFT','ORB','KAZE','BRISK','AKAZE']
            des_dict = {}
            # # Print
            # img2 = cv2.drawKeypoints(gray,kp,img_face)
            res_best = 0.0
            manga_best = None
            match_name = None
            material_dir = './FYM-Images/{}'.format(landmark)
            for root, dirs, files in os.walk(material_dir):
                for file in files:
                    if(not file.endswith('.png')):
                        continue
                    manga_id = landmark + '_{}'.format(file.rstrip('.png'))
                    ##  take SIFT feature of manga landmark
                    file_path = os.path.join(material_dir,file)
                    # img_landmark = dlib.load_rgb_image(file_path)
                    img_manga = cv2.imread(os.path.join(material_dir,file))
                    img_manga_base = copy.deepcopy(img_manga)
                    if(landmark != 'hair'):
                        img_manga = img_manga[300:800,300:600,:]


                    res = 0
                    descriptors = ['SIFT', 'ORB']

                    for des in descriptors:
                        if(des not in des_dict.keys()):
                            des1 = get_des(des)
                            kp_real, des_real = des1.detectAndCompute(gray, None)
                            des_dict[des]= {
                                'kp':kp_real,
                                'des':des_real
                            }
                        else:
                            kp_real, des_real = des_dict[des]['kp'] , des_dict[des]['des']
                        gray_mangga = cv2.cvtColor(img_manga, cv2.COLOR_BGR2GRAY)
                        des2 = get_des(des)
                        # cv2.imshow('rgb_manga',img_manga)
                        kp_manga, des_manga = des2.detectAndCompute(gray_mangga, None)

                        res += match_img(img_face,kp_real, des_real,img_manga,kp_manga, des_manga,des)

                    if(res_best<res):
                        res_best=res
                        manga_best=img_manga_base
                        match_name = file
            target_file_name = os.path.join(target_dir,'./{}_{}.jpg'.format(landmark,match_name))
            results[landmark] = match_name
            cv2.imwrite(target_file_name,manga_best)

    print("results:",results)
    # return results