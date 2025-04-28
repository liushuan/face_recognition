import sys
import os
from time import time
from pathlib import Path
import numpy as np
import glob
import cv2
import json
import shutil
import copy
import onnxruntime
import math
import time

def sigmoid(x):
    return 1/(1+np.exp(-x))


class YoloxWrapper:
    def __init__(self, onnx_file, in_width, in_height, conf_thre=0.2, nms_thre=0.5, class_names=["head",], min_size=[(5,5),], colors=[(255, 0, 0), ], show_indexs=[True,], with_keypoints=False, device="cpu"):
        self.onnx_file = onnx_file
        self.in_width = in_width
        self.in_height = in_height
        self.class_number = len(class_names)
        self.show_indexs=show_indexs
        self.class_names   = class_names
        self.input_size    = (in_height, in_width)      # (heigth, width)
        
        self.mean          = (0.0, 0.0, 0.0)
        self.std           = (1.0, 1.0, 1.0)
        self.norm          = False
        self.pad           = True
        self.resize        = True
        self.swapRGB       = False
        self.p6            = False
        self.with_keypoints= with_keypoints
        self.keypoint_number = 5
        
        self.colors = colors
        self.threshold =conf_thre
        self.nms_thre = nms_thre
        self.headnum       = 3
        self.strides = [8, 16, 32]
        self.keep_ratio = False
        self.crop = True
        if self.keep_ratio == False and self.resize == True:
            self.pad  = False

        self.max_score =0.0
        self.json_det = []
        self.category_ids={0:1,}
        self.device = device
        self.session = self.model_init(self.device)
        self.grids, self.expanded_strides = self.grid_init()
    def model_init(self, device="cpu"):
        if device == "gpu":
            session = onnxruntime.InferenceSession(self.onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            session = onnxruntime.InferenceSession(self.onnx_file, providers=['CPUExecutionProvider'])
            
        return session
    
    def grid_init(self,):
        grids = []
        expanded_strides = []
        if not self.p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [self.in_height // stride for stride in strides]
        wsizes = [self.in_width // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        # import pdb; pdb.set_trace()
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        return grids, expanded_strides
    
    def image_preprocess(self, img, swap=(2, 0, 1)):
        
        if len(img.shape) == 3:
            padded_img = np.ones((self.in_height, self.in_width , 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones((self.in_height, self.in_width), dtype=np.uint8) * 114

        r = min(self.in_height / img.shape[0], self.in_width  / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        if self.swapRGB:
            padded_img = padded_img[:, :, ::-1]
        padded_img = (padded_img - self.mean )/self.std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def forward(self, padded_img):
        ort_inputs = {self.session.get_inputs()[0].name: padded_img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        # print(output[0].shape)
        # import pdb; pdb.set_trace()
        ### cls
        shape1 = output[0].shape
        output[0] = output[0].reshape(shape1[0], shape1[1], -1)
        output[1] = output[1].reshape(shape1[0], shape1[1], -1)
        output[2] = output[2].reshape(shape1[0], shape1[1], -1)
        output1 = np.concatenate((output[0], output[1], output[2]), axis=-1)
        
        ### box
        shape2 = output[3].shape
        output[3] = output[3].reshape(shape2[0], shape2[1], -1)
        output[4] = output[4].reshape(shape2[0], shape2[1], -1)
        output[5] = output[5].reshape(shape2[0], shape2[1], -1)
        output2 = np.concatenate((output[3], output[4], output[5]), axis=-1)
        
        ### obj
        shape3 = output[6].shape
        output[6] = output[6].reshape(shape3[0], shape3[1], -1)
        output[7] = output[7].reshape(shape3[0], shape3[1], -1)
        output[8] = output[8].reshape(shape3[0], shape3[1], -1)
        output3 = np.concatenate((output[6], output[7], output[8]), axis=-1)
        # import pdb; pdb.set_trace()
        if self.with_keypoints:
            shape4 = output[9].shape
            output[9] = output[9].reshape(shape4[0], shape4[1], -1)
            output[10] = output[10].reshape(shape4[0], shape4[1], -1)
            output[11] = output[11].reshape(shape4[0], shape4[1], -1)
            output4 = np.concatenate((output[9], output[10], output[11]), axis=-1)

            # shape5 = output[12].shape
            # output[12] = output[12].reshape(shape5[0], shape5[1], -1)
            # output[13] = output[13].reshape(shape5[0], shape5[1], -1)
            # output[14] = output[14].reshape(shape5[0], shape5[1], -1)
            # output5 = np.concatenate((output[12], output[13], output[14]), axis=-1)
            
            outputs = np.concatenate((output2, output1, output3, output4), axis=1)
        else:
            outputs = np.concatenate((output2, output1, output3), axis=1)
        outputs = np.transpose(outputs, axes=(0, 2, 1))
        return outputs
    
    def postprocess(self, outputs, ratio):
        outputs[..., :2] = (outputs[..., :2] + self.grids) * self.expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * self.expanded_strides
        outputs[..., 4:4+self.class_number+1] = 1 / (1 + np.exp(-outputs[..., 4:4+self.class_number+1]))
        ###keypoint todo
        # import pdb; pdb.set_trace()
        if self.with_keypoints:
            # visible_start = 4+self.class_number+1 + self.keypoint_number*2
            # visible_end = visible_start + self.keypoint_number
            # outputs[..., visible_start:visible_end] = 1 / (1 + np.exp(-outputs[..., visible_start:visible_end]))
            start_idx = 4+self.class_number+1
            end_idx = start_idx + self.keypoint_number*2
            step = 2
            for i in range(start_idx, end_idx, step):
                outputs[..., i:i+2]   = (outputs[..., i:i+2]   + self.grids) * self.expanded_strides
        # import pdb; pdb.set_trace()
        predictions = outputs[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:4+self.class_number] * predictions[:, 4+self.class_number:4+self.class_number+1]
        
        if self.with_keypoints:
            keypoints = predictions[:, 4+self.class_number+1:4+self.class_number+1+self.keypoint_number*2]
            keypoints[:, 0:self.keypoint_number*2] /= ratio
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        if self.with_keypoints:
            dets = self.multiclass_nms_with_keypoint(boxes_xyxy, scores, keypoints, confThreshold=self.threshold, nmsThreshold=self.nms_thre)
        else:
            dets = self.multiclass_nms(boxes_xyxy, scores, confThreshold=self.threshold, nmsThreshold=self.nms_thre)
        return dets
    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep
    def multiclass_nms(self, boxes, scores, confThreshold, nmsThreshold):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores >= confThreshold
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nmsThreshold)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def multiclass_nms_with_keypoint(self, boxes, scores, keypoints, confThreshold, nmsThreshold):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores >= confThreshold
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                valid_keypoints = keypoints[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nmsThreshold)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds, valid_keypoints[keep]], 1)
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def put_text(self, img, label, p1, bg_color=(100, 100, 100), txt_color=(255, 255, 255)):
        lw = 2
        offset=5
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - offset >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - offset if outside else p1[1] + h + offset
        cv2.rectangle(img, p1, p2, bg_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - offset-1 if outside else p1[1] + h + offset-1), 0, lw / 3, txt_color,
            thickness=tf, lineType=cv2.LINE_AA)

    def vis(self, img, dets):
        vis_objs =0
        boxes, scores, cls_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            condition_good = score > self.threshold and self.show_indexs[cls_id]
            
            if condition_good:
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                text = '{}:{:.2f}'.format(self.class_names[cls_id], score)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), self.colors[cls_id], 4)
                # height, width, _ = img.shape
                # scale = max(height/352, width/640)
                # w_h_scale = " wh:"+str(int((x1-x0)/scale))+" "+str(int((y1-y0)/scale))
                # cv2.putText(img, str(round(score, 2)) + "w-h:"+str((x1-x0)//scale)+" "+str((y1-y0)//scale), (x0, y0 + txt_size[1]), font, 1.0, (0, 255, 0), thickness=2)
                # text += w_h_scale
                self.put_text(img, text, (x0, y0))
                vis_objs+=1
        return img, vis_objs

    def vis_keypoints(self, img, dets, keypoint_index=0):
        boxes, scores, cls_ids, landmarkes = dets[:, :4], dets[:, 4], dets[:, 5], dets[:, 6:]
        vis_objs = 0
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            landmark = landmarkes[i]
            condition_good = score > self.threshold and self.show_indexs[cls_id]
            if condition_good:
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                text = '{}:{:.2f}'.format(self.class_names[cls_id], score)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), self.colors[cls_id], 4)
                # height, width, _ = img.shape
                # scale = max(height/352, width/640)
                # w_h_scale = " wh:"+str(int((x1-x0)/scale))+" "+str(int((y1-y0)/scale))
                # cv2.putText(img, str(round(score, 2)) + "w-h:"+str((x1-x0)//scale)+" "+str((y1-y0)//scale), (x0, y0 + txt_size[1]), font, 1.0, (0, 255, 0), thickness=2)
                # text += w_h_scale
                
                if cls_id == keypoint_index:
                    for k in range(self.keypoint_number):
                        cv2.circle(img, (int(landmark[2*k]), int(landmark[2*k+1])), 2, (0, 0, 255), -1)
                self.put_text(img, text, (x0, y0))   
                vis_objs+=1
        return img, vis_objs
    
    def generate_xy(self, dets):
        center_xys = []
        boxes, scores, cls_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            condition_good = score > self.threshold and self.show_indexs[cls_id]
            
            if condition_good:
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                center_xys.append([(x0+x1)//2, (y0+y1)//2])
        return center_xys

    def get_det_box_landmark(self, dets):
        boxes, scores, cls_ids, landmarkes = dets[:, :4], dets[:, 4], dets[:, 5], dets[:, 6:]
        return boxes, landmarkes

if __name__ == '__main__':

    input_file = r'imgs/1736938718.809096_0.png'
    output_file = r'outputs/44_result.jpg'
    model = r'models/yolo_nano_nas075fpl_epoch_200.onnx'
    yolo_det =  YoloxWrapper(model, 640, 384, class_names=["face",], with_keypoints=True, device="cpu")
    
    img = cv2.imread(input_file)
    print(img.shape)
    time1 = time.time()
    padded_img, r = yolo_det.image_preprocess(img)
    outputs = yolo_det.forward(padded_img)
    dets = yolo_det.postprocess(outputs, r)
    time2 = time.time()
    print('infer time: ', (time2-time1)*1000, " ms")
    if dets is not None:
        img, vis_objs = yolo_det.vis_keypoints(img, dets)
    print(dets.shape, vis_objs)
    cv2.imwrite(output_file, img)
    