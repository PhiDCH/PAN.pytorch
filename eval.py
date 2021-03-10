# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import cv2
import torch
import shutil
import numpy as np
from tqdm.auto import tqdm

from utils import cal_recall_precison_f1, draw_bbox

torch.backends.cudnn.benchmark = True

import argparse
from predict import Pytorch_model

def run(model_path, img_folder, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
    model = Pytorch_model(model_path, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        _, boxes_list, t = model.predict(img_path)
        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder

def main():
    parse = argparse.ArgumentParser(description="Evaluate ICDAR2015")
    parse.add_argument('--trained_model', default='./pretrain/pan-resnet18-ic15.pth', type=str, help='pretrained model')
    parse.add_argument('--img_path', default='../icdar2015/test_img', type=str, help='folder path to test images')
    parse.add_argument('--gt_path', default='../icdar2015/test_gt', type=str, help='folder path to ground truth')
    parse.add_argument('--result_folder', default='./result', type=str, help='folder path to result images')

    args = parse.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    # model_path = r'output/PAN_shufflenetv2_FPEM_FFM.pth'
    # img_path = r'/mnt/e/zj/dataset/icdar2015/test/img'
    # gt_path = r'/mnt/e/zj/dataset/icdar2015/test/gt'
    # save_path = './output/result'#model_path.replace('checkpoint/best_model.pth', 'result/')
    
    model_path = args.trained_model
    img_path = args.img_path
    gt_path = args.gt_path
    save_path = args.result_folder
    gpu_id = 0

    save_path = run(model_path, img_path, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)

if __name__ == '__main__':
    main()
