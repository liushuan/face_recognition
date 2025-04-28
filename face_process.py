
import cv2
import numpy as np
from skimage import transform as trans
import glob
import  os

def get_crop_img_landmark(img_raw, box, landmark, zoom=1.5):
    img_height, img_width , _ = img_raw.shape
    width = box[2] - box[0]
    height = box[3] - box[1]
    offset_x = int(width*(zoom-1.0)/2)
    offset_y = int(height*(zoom-1.0)/2)
    xmin = int(max(box[0] - offset_x, 0))
    ymin = int(max(box[1] - offset_y, 0))
    xmax = int(min(box[2] + offset_x, img_width))
    ymax = int(min(box[3] + offset_y, img_height))
    crop_face = img_raw[ymin:ymax, xmin:xmax]
    for i in range(5):
        landmark[2*i] = landmark[2*i] - xmin
        landmark[2 * i+1] = landmark[2 * i+1] - ymin
    return crop_face, landmark

def process_face_warp(img_raw, box, landmark):
    mean_landmark = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    mean_landmark[:, 0] += 8.0
    crop_face, landmark = get_crop_img_landmark(img_raw, box, landmark)
    dst = np.array(landmark).astype(np.float32).reshape(-1, 2)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, mean_landmark)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(crop_face, M, (112, 112), borderValue=0.0)
    return warped

def get_cos(feature1, feature2):
    score = np.dot(feature1, feature2.transpose())
    return score