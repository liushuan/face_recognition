from yolox_infer import YoloxWrapper
from face_process import process_face_warp
import cv2
import time
import glob
import numpy as np
import os
import shutil
from tqdm import tqdm
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

import glob 
from multiprocessing import Pool

def crop_face_item(file_list, result_file, show_bar):
    model = r'models/face_det_sim.onnx'
    yolo_det =  YoloxWrapper(model, 640, 384, class_names=["face",], with_keypoints=True, device="gpu")
    pbar = file_list
    if show_bar:
        pbar = tqdm(file_list, total=len(file_list), bar_format=TQDM_BAR_FORMAT)
    for file in pbar:
        img = cv2.imread(file)
        # print(img.shape)
        padded_img, r = yolo_det.image_preprocess(img)
        outputs = yolo_det.forward(padded_img)
        dets = yolo_det.postprocess(outputs, r)
        if dets is not None:
            # print(dets)
            name = file.split("/")[-1]
            
            boxes, landmarkes =  yolo_det.get_det_box_landmark(dets)
            for i in range(len(boxes)):
                warp_face = process_face_warp(img, boxes[i], landmarkes[i])
                
                save_name = os.path.join(result_file, os.path.splitext(name)[0]+"_"+str(i)+".jpg")
                cv2.imwrite(save_name, warp_face)

def crop_face_align():
    
    input_file = r'/home/liushuan/dataset/face/test/test/image'
    result_file = r'/home/liushuan/dataset/face/test/test/crop_face'
    
    model = r'models/yolo_nano_nas075fpl_epoch_200.onnx'
    yolo_det =  YoloxWrapper(model, 640, 384, class_names=["face",], with_keypoints=True, device="gpu")
    
    file_list = glob.glob(input_file+"/*.JPG")+glob.glob(input_file+"/*.png")
    print("file len:", len(file_list))
    pbar = tqdm(file_list, total=len(file_list), bar_format=TQDM_BAR_FORMAT)
    for file in pbar:
        img = cv2.imread(file)
        # print(img.shape)
        padded_img, r = yolo_det.image_preprocess(img)
        outputs = yolo_det.forward(padded_img)
        dets = yolo_det.postprocess(outputs, r)
        if dets is not None:
            # print(dets)
            name = file.split("/")[-1]
            
            boxes, landmarkes =  yolo_det.get_det_box_landmark(dets)
            for i in range(len(boxes)):
                warp_face = process_face_warp(img, boxes[i], landmarkes[i])
                
                save_name = os.path.join(result_file, os.path.splitext(name)[0]+"_"+str(i)+".jpg")
                cv2.imwrite(save_name, warp_face)


def crop_face_multi():
    input_file = r'/home/liushuan/dataset/face/test/test/image'
    result_file = r'/home/liushuan/dataset/face/test/test/crop_face2'
    file_list = glob.glob(input_file+"/*.JPG")+glob.glob(input_file+"/*.png")
    print("file len:", len(file_list))
    thread_number = 8
    process = Pool(thread_number)
    length = len(file_list)
    one_length = round(length / thread_number + 0.5)
    print("length :", length )
    show_bar = False
    for i in range(thread_number):
        device_id = i
        if i == 0:
            show_bar = False
        else:
            show_bar = False
        if (i + 1)*one_length < length:
            new_crop_files =  file_list[i*one_length:(i+1)*one_length]
            process.apply_async(crop_face_item, args=(new_crop_files, result_file, show_bar))
        elif i*one_length < length:
            new_crop_files = file_list[i*one_length:]
            process.apply_async(crop_face_item, args=(new_crop_files, result_file, show_bar))
    print("start crop face")
    process.close()
    process.join()
    print("crop face  finished.")


if __name__ == '__main__':
    crop_face_multi()
    # crop_face_align()