from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import logging
import os
import os.path as osp
from opts import opts
import argparse
import time
from pathlib import Path
import torch
import cv2
import motmetrics as mm
import numpy as np
import torch
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.timer import Timer
from datasets.dataset.jde import LoadVideo
from ReID.model import ft_net
from ReID.reid import *
from my_utils import *

logger.setLevel(logging.INFO)

if __name__ == '__main__':

    # 初始化各项参数
    ######################################################################
        
    opt = opts().init() # 初始化配置
  
    source = "cam.txt" # 视频流文本
    show_image = True # 使用opencv实时显示图像
    use_cuda = True # 使用cuda
    save_dir = "../results/" # 每帧图像储存得到位置
    
    init_1 = True # 摄像头1初始化重叠区域
    init_0 = True # 摄像头0初始化重叠区域
    use_cam_pos = False # 如果提前知道两相机的位置关系则可令其为True，并在cam_pos变量处填写角度单位为弧度
    cam_pos = None 

    if save_dir:
        mkdir_if_missing(save_dir) # 创建输出所在文件夹

    stream_num = 2 # 视频流数目
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择在GPU活在CPU上运行
    
    dataloader = LoadVideo(source, opt.img_size) # 加载视频流

    logger.info('Starting tracking...')

    tracker = [] # 存放供不同摄像头使用的跟踪器
    online_im = [None] * stream_num
    online_targets = [None] * stream_num 
    det = [None] * stream_num
    tlwh = [None] * stream_num
    tid = [None] * stream_num
    vertical = [None] * stream_num


    for i in range(stream_num):
        tracker.append(JDETracker(opt)) # 生成不同摄像头的跟踪器

    timer = Timer() # 用于计时以计算帧数
    # results = []
    frame_id = 0

    ######################################################################

    # Re-ID模块
    ######################################################################
    
    data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    reid_backbone = 'ft_ResNet50'

    reid_ms = "1"
    print('We use the scale: %s'%reid_ms)
    str_ms = reid_ms.split(',')
    ms = [] 
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    reid_stride = 2

    reid_nclasses = 751 
    
    reid_model_structure = ft_net(reid_nclasses, stride = reid_stride)
    reid_model = load_network(reid_backbone, reid_model_structure) # 加载reID模型
    reid_model.classifier.classifier = nn.Sequential() # 移除最后的全连接层与分类器
    reid_model = reid_model.eval().to(device) # 转为测试模型
    ######################################################################  

    # 跨摄像头跟踪主模块
    ######################################################################
    for path, img, img0 in dataloader:

        online_persons_det = [[]] * stream_num # 用于存储检测到的人物图像，作为Re-ID模块的输入
        cnt = [0] * stream_num
        inds = []
        dv_est = 0
        online_tlwhs = []
        online_ids = []

        for _ in range(stream_num):
            online_tlwhs.append([])
            online_ids.append([])
            inds.append([])

        if init_1:
            raw_1 = cv2.resize(img0[1].copy(), (640, 480))
            roi_1 = cv2.selectROI('online_im1',raw_1, False, False)
            tl_1, dr_1 = c640to1920(roi_1) # 左上角与右下角
            init_1 = False

        if init_0:
            raw_0 = cv2.resize(img0[0].copy(), (640, 480))
            roi_0 = cv2.selectROI('online_im0',raw_0, False, False)
            tl_0, dr_0 = c640to1920(roi_0)
            init_0 = False

        grid_pos_1 = grid_clip(tl_1, dr_1) # 得到区域内各网格的坐标信息
        grid_pos_0 = grid_clip(tl_0, dr_0)

        grid_rst_1 = [0] * 6 # 用于记录当前区域各网格下的人数
        grid_rst_0 = [0] * 6

        for i, cam_id in enumerate(path):

            timer.tic() # 开始计时
            if use_cuda:
                blob = torch.from_numpy(img[i]).cuda().unsqueeze(0)
            else:
                blob = torch.from_numpy(img[i]).unsqueeze(0)
            online_targets[i] = tracker[i].update(blob, img0[i]) # 更新当前帧下的目标

            for c, t in enumerate(online_targets[i]):
                tlwh[i] = t.tlwh
                x1, y1, x2, y2 = int(t.tlwh[0]), int(t.tlwh[1]), int(t.tlwh[0])+int(t.tlwh[2]), int(t.tlwh[1])+int(t.tlwh[3])                
                tid[i] = t.track_id
                vertical[i] = tlwh[i][2] / tlwh[i][3] > 1.6 # 用于排除宽高比大于1.6的误检
                x1, y1, x2, y2 = pos_clip(x1, y1, x2, y2) # 进行坐标截断

                if tlwh[i][2] * tlwh[i][3] > opt.min_box_area and not vertical[i]: # 过滤边界框特别小和宽高比奇怪的目标
                    online_tlwhs[i].append(tlwh[i]) # 将边框信息(top_left_x, top_left_y, width, height)添加进去
                    online_ids[i].append(tid[i])

                    if i != 0:
                        if x1 >= tl_1[0] and x2 <= dr_1[0] and y1 >= tl_1[1] and y2 <= dr_1[1]:
                            center_pos_x_1 = (x1 + x2) // 2
                            grid_rst_1 = grid_cluster(grid_rst_1, center_pos_x_1, grid_pos_1)
                            im0 = img0[i][y1:y2,x1:x2,:]
                            im0 = data_transforms(im0) # 将人像转换为Re-ID的输入形式
                            im0 = im0.unsqueeze(0)
                            inds[i].append(c) # 对应位strack中的第c个轨迹

                            if cnt[i] == 0:
                                online_persons_det[i] = im0
                                cnt[i] = -1
                            else:
                                online_persons_det[i] = torch.cat((online_persons_det[i], im0), dim=0)

                    else:
                        if x1 >= tl_0[0] and x2 <= dr_0[0] and y1 >= tl_0[1] and y2 <= dr_0[1]:
                            center_pos_x_0 = (x1 + x2) // 2
                            grid_rst_0 = grid_cluster(grid_rst_0, center_pos_x_0, grid_pos_0)
                            im0 = img0[i][y1:y2,x1:x2,:]
                            im0 = data_transforms(im0)
                            im0 = im0.unsqueeze(0)
                            inds[i].append(c)

                            if cnt[i] == 0:
                                online_persons_det[i] = im0
                                cnt[i] = -1
                            else:
                                online_persons_det[i] = torch.cat((online_persons_det[i], im0), dim=0)

            reid_flag = 1
            for det_i in range(len(online_persons_det)):
                reid_flag = reid_flag * len(online_persons_det[det_i]) # 当两个区域内都有行人时才进行特征匹配

            # 特征匹配
            if reid_flag:    
                reid_feat = extract_feature(reid_model, online_persons_det, ms) # 使用Re-ID对人像提取特征
                dists = feat_dist(reid_feat) # 使用余弦距离进行匹配
                mini_dist_arg = np.argmin(dists, axis=1)
                for c0, c1 in enumerate(mini_dist_arg): # c0与c1为要匹配的不同摄像头下的两个目标
                    if dists[c0][c1] <= 0.35: # 只有当距离小于阈值时才进行匹配
                            
                            pos_1 = online_targets[1][inds[1][c1]].tlwh
                            cent_x_1 = (2 * pos_1[0] + pos_1[2]) // 2 # 中心点x坐标
                            g_num_1 = grid_rst_1[int(judge_num(tl_1, dr_1, cent_x_1))] # 得到当前网格下的人数
                
                            pos_0 = online_targets[0][inds[0][c0]].tlwh
                            cent_x_0 = (2 * pos_0[0] + pos_0[2]) // 2
                            g_num_0 = grid_rst_0[int(judge_num(tl_0, dr_0, cent_x_0))]
                            
                            if use_cam_pos == True:
                                dv_1 = online_targets[1][inds[1][c1]].dv
                                dv_0 = online_targets[0][inds[0][c0]].dv
                                dv_est = dv_estimation(dv_0, dv_1, cam_pos) # 计算方向向量的夹角
                                if dv_1[0] != 1 and dv_0[0] != 1:
                                    if g_num_0 <= 1 and g_num_1 <= 1 and np.abs(dv_est) <= 60:
                                        online_targets[1][inds[1][c1]].track_id = online_targets[0][inds[0][c0]].track_id 
                                        online_ids[1][inds[1][c1]] = online_ids[0][inds[0][c0]]
                            else:
                                if g_num_0 <= 1 and g_num_1 <= 1:
                                        online_targets[1][inds[1][c1]].track_id = online_targets[0][inds[0][c0]].track_id 
                                        online_ids[1][inds[1][c1]] = online_ids[0][inds[0][c0]]

            timer.toc()
            
            if show_image or save_dir is not None:
                online_im[i] = vis.plot_tracking(img0[i], online_tlwhs[i], online_ids[i], frame_id=frame_id,
                                            fps=1. / timer.average_time)
            
            if show_image:

                if i == 1:
                    cv2.rectangle(online_im[1], tl_1, dr_1, (0, 255, 0), 5)
                    online_im[1] = plot_grid(online_im[1], grid_pos_1)
                if i == 0:
                    cv2.rectangle(online_im[0], tl_0, dr_0, (0, 255, 0), 5)
                    online_im[0] = plot_grid(online_im[0], grid_pos_0)

                show = cv2.resize(online_im[i], (640, 480)) # (1920, 1080) 1920/640 = 3 1080/480 = 2.25
                cv2.imshow('online_im%d'%i, show)
                cv2.waitKey(1)
            
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{}/{:05d}.jpg'.format(i,frame_id)), online_im[i])

        frame_id += 1
    ######################################################################