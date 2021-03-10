# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import cv2
import torch
import shutil
import numpy as np
from tqdm.auto import tqdm

from models import get_model
from post_processing import decode
import torch 
from torchvision import transforms
import os
import cv2
import time

from utils import cal_recall_precison_f1, draw_bbox

torch.backends.cudnn.benchmark = True

import argparse

parse = argparse.ArgumentParser(description="Evaluate ICDAR2015")
parse.add_argument('--trained_model', default='./pretrain/pan-resnet18-ic15.pth', type=str, help='pretrained model')
parse.add_argument('--img_path', default='../icdar2015/test_img', type=str, help='folder path to test images')
parse.add_argument('--gt_path', default='../icdar2015/test_gt', type=str, help='folder path to ground truth')
parse.add_argument('--result_folder', default='./result', type=str, help='folder path to result images')

args = parse.parse_args()

class Pytorch_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        self.net = get_model(config)

        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img: str, short_size: int = 736):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        img = cv2.imread(img)
        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds, boxes_list = decode(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start
        return preds, boxes_list, t

def main(model_path, img_folder, save_path, gpu_id):
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


if __name__ == '__main__':
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

    save_path = main(model_path, img_path, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)
