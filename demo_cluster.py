from yolox_infer import YoloxWrapper
from face_process import process_face_warp
from face_extract import FaceExtract
from dbscan_cluster import dbscan_cluster
from my_cluster import *
import cv2
import time
import glob
import numpy as np
import os
import shutil


def test_image():
    input_file = r'imgs/1736938718.809096_0.png'
    output_file = r'outputs/44_result.jpg'
    model = r'models/yolo_nano_nas075fpl_epoch_200.onnx'
    yolo_det =  YoloxWrapper(model, 640, 384, class_names=["face",], with_keypoints=True, device="cpu")
    reg_model = r'models/model_r28.onnx'
    face_extract = FaceExtract(reg_model, 112, 112, device='cpu')
    img = cv2.imread(input_file)
    print(img.shape)
    time1 = time.time()
    padded_img, r = yolo_det.image_preprocess(img)
    outputs = yolo_det.forward(padded_img)
    dets = yolo_det.postprocess(outputs, r)
    time2 = time.time()
    print('infer time: ', (time2-time1)*1000, " ms")
    if dets is not None:
        print(dets)
        boxes, landmarkes =  get_det_box_landmark(dets)
        print(boxes, landmarkes)
        for i in range(len(boxes)):
            time1 = time.time()
            warp_face = process_face_warp(img, boxes[i], landmarkes[i])
            feature, norm =  face_extract.forward(warp_face)
            time2 = time.time()
            print('extract time: ', (time2-time1)*1000, " ms")
            print(feature.shape, "norm:", norm)
            cv2.imwrite("warp_face.jpg", warp_face)

def test_feature():
    input_file = r'/home/liushuan/dataset/face/test_data/menjin2/norm_2list/157590_1.jpg'
    reg_model = r'models/model_r28.onnx'
    face_extract = FaceExtract(reg_model, 112, 112, device='cpu')
    img = cv2.imread(input_file)
    feature, norm =  face_extract.forward(img)
    
    print(norm, "feature:", feature)

def get_forder_features(image_path):

    file_list = glob.glob(image_path+"/**/*.png", recursive=True) + glob.glob(image_path+"/**/*.jpg", recursive=True)
    reg_model = r'models/model_r28.onnx'
    face_extract = FaceExtract(reg_model, 112, 112, device='cpu')
    feature_list = []
    norm_list = []
    print("face len:", len(file_list))
    for file_name in file_list:
        img = cv2.imread(file_name)
        if img is None:
            continue
        # warp_face = process_face_warp(img, boxes[i], landmarkes[i])
        feature, norm =  face_extract.forward(img)
        feature_list.append(feature)
        norm_list.append(norm)
    return file_list, feature_list, norm_list

def copy_file_cluster(dst_path, file_list, labels):
    for i in range(len(labels)):
        src_name = file_list[i]
        name = src_name.split("/")[-1]
        dst_dir = os.path.join(dst_path, str(labels[i]))
        if os.path.exists(dst_dir) is False:
            os.makedirs(dst_dir)
        dst_name = os.path.join(dst_dir, name)
        shutil.copy(src_name, dst_name)
        
        
from collections import Counter

def delete_tree_file(dst_path):
    # 遍历当前目录下的所有文件和文件夹
    for item in os.listdir(dst_path):
        item_path = os.path.join(dst_path, item)  # 获取完整路径
        # 如果是文件，删除
        if os.path.isfile(item_path):
            os.remove(item_path)
        # 如果是目录，删除整个目录
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

'''
1.挑选满足可以当底库的图(正面, 大人脸，高清晰度， 无遮挡， norm值比较大) ;
2.质量top排序后,按顺序合并其他的底库图像到当前图像 ;
3.选择下一个还未合并的底库继续 2 ;
4.合并其他图到底库；

'''

def test_cluster():
    image_path = r'/home/liushuan/dataset/face/test_data/menjin2/norm_2list'
    dst_path = r'/home/liushuan/dataset/face/test_cluster'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)
    else:
        delete_tree_file(dst_path)
    
    file_list, feature_list, norm_list = get_forder_features(image_path)
    feats = np.array(feature_list)
    print(feats.shape)
    time1 = time.time()
    labels = dbscan_cluster(features=feats, eps=0.3, min_samples=4)
    time2 = time.time()
    print('cluster time: ', (time2-time1)*1000, " ms")
    print("labels:",labels) 
    # import pdb; pdb.set_trace()
    count_dict = {}
    for item in labels:
        count_dict[item] = count_dict.get(item, 0) + 1
    print(count_dict)
    copy_file_cluster(dst_path, file_list, labels)

def test_cluster_ls():
    image_path = r'/home/liushuan/dataset/face/test/test/crop_face'
    dst_path = r'/home/liushuan/dataset/face/test/test/crop_face_cluster'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)
    else:
        delete_tree_file(dst_path)
    
    print("start extract feature")
    file_list, feature_list, norm_list = get_forder_features(image_path)
    features = np.array(feature_list)
    # np.save("features.npy", features)
    # np.save("norms.npy", np.array(norm_list))
    print("extract feature finished.")
    
    print("start cluster.")
    sorted_indices = get_quality_sort(norm_list, dst_path, file_list)
    cluster_id, not_cluster_id, lower_quality_id = cluster_features(sorted_indices, norm_list, features, file_list)
    not_cluster_id = iter_cluster(cluster_id, not_cluster_id, lower_quality_id, features)
    copy_file_cluster_result(cluster_id, not_cluster_id, lower_quality_id, dst_path, file_list)
    # import pdb; pdb.set_trace()
    
    print("cluster len is:", len(cluster_id.keys()))
    
    size_all = 0
    for key, dict_a in cluster_id.items():
        size_all += len(dict_a["A"])+len(dict_a["B"])+len(dict_a["C"])+len(dict_a["D"])+len(dict_a["E"])+len(dict_a["F"])
    
    size_all += len(not_cluster_id)
    size_all += len(lower_quality_id)
    
    print("size:", size_all)
    print("cluster finished")
    
    
if __name__ == '__main__':
    
    # test_feature()
    # test_image()
    test_cluster_ls()