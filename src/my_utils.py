import numpy as np
from scipy.spatial.distance import cdist
import cv2

"""
函数名：get_stream_num
函数功能：获取接入的视频流数目
参数：
    source：写有视频文件名的txt文件名
"""
def get_stream_num(source):

    return len(open(source).readlines())

"""
函数名：feat_dist
函数功能：输入各个目标的特征以计算它们之间的余弦距离，并返回这个距离
参数：
    def_feat：包含有不同摄像头下目标的特征
"""
def feat_dist(det_feat):

    query = det_feat[1] # n x 512
    gallery_feature = det_feat[0] # n x 512 
    cost_matrix = np.maximum(0.0, cdist(gallery_feature, query, "cosine"))  # Nomalized features

    return cost_matrix 

"""
函数名：c640to1920
函数功能：将640分辨率转换到1920
参数：
    img：输入图像
"""
def c640to1920(img):

    r = list(img)
    x1 = r[0] * 3
    x2 = x1 + r[2] * 3
    y1 = round(r[1] * 2.25)
    y2 = y1 + round(r[3] * 2.25)

    return (x1, y1), (x2, y2)

"""
函数名：grid_clip
函数功能：进行网格划分
参数：
    p1：划分区域左上角点坐标
    p2：划分区域右下角点坐标
"""
def grid_clip(p1, p2):

    grid_num = 6 # 令网格数量为6个
    width = p2[0] - p1[0]
    grid_width = width // grid_num
    gp = []
    for i in range(grid_num):
        tmp = []
        tmp.append((p1[0] + grid_width * (i+1), p1[1]))
        tmp.append((p1[0] + grid_width * (i+1), p2[1]))
        gp.append(tmp)
    
    return gp

"""
函数名：grid_cluster
函数功能：将目标划分达到对应网格下
参数：
    rst：网格状态
    pos_x：目标的中心点x轴坐标
    gp：网格坐标
"""
def grid_cluster(rst, pos_x, gp):
    # rst = [0] * 6 # grid_num = 6
    gp_x = []
    for i in range(len(rst)):
        gp_x.append(gp[i][0][0])

    if pos_x < gp_x[0]:
        rst[0] += 1
    elif pos_x >= gp_x[0] and pos_x < gp_x[1]:
        rst[1] += 1
    elif pos_x >= gp_x[1] and pos_x < gp_x[2]:
        rst[2] += 1
    elif pos_x >= gp_x[2] and pos_x < gp_x[3]:
        rst[3] += 1
    elif pos_x >= gp_x[3] and pos_x < gp_x[4]:
        rst[4] += 1
    elif pos_x >= gp_x[5]:
        rst[5] += 1
    
    return rst

"""
函数名：judge_num
函数功能：判断目标所属网格
参数：
    p1：划分区域左上角点坐标
    p2：划分区域右下角点坐标
    pos_x：目标的中心点x轴坐标
"""
def judge_num(p1, p2, pos_x):

    grid_num = 6
    width = p2[0] - p1[0]
    grid_width = width // grid_num 
    pos = (pos_x - p1[0]) // grid_width

    return pos


"""
函数名：pos_clip
函数功能：坐标截断，将出现小于0的值截断为0
参数：
    x1,y1,x2,y2：目标位置
"""
def pos_clip(x1, y1, x2, y2):

    if x1 < 0:
        x1 = 0
    if x2 < 0:
        x2 = 0
    if y1 < 0:
        y1 = 0
    if y2 < 0:
        y2 = 0

    return x1, y1, x2, y2

"""
函数名：plot_grid
函数功能：画网格，用于辅助示意
参数：
    img：输入图像
    grid_pos：网格坐标
"""
def plot_grid(img, grid_pos):

    cv2.line(img, grid_pos[0][0], grid_pos[0][1], (0, 0, 255), 3)
    cv2.line(img, grid_pos[1][0], grid_pos[1][1], (0, 0, 255), 3)
    cv2.line(img, grid_pos[2][0], grid_pos[2][1], (0, 0, 255), 3)
    cv2.line(img, grid_pos[3][0], grid_pos[3][1], (0, 0, 255), 3)
    cv2.line(img, grid_pos[4][0], grid_pos[4][1], (0, 0, 255), 3)

    return img

"""
函数名：dv_est
函数功能：用于方向向量的估计并判断夹角
参数：
    v0：摄像头0下目标的移动方向向量
    v1：摄像头1下目标的移动方向向量
    cam_agl：两摄像头的位置关系
"""
def dv_estimation(v0, v1, cam_agl):
    unit_v0 = v0 / np.linalg.norm(v0)
    unit_v1 = v1 / np.linalg.norm(v1)
    rotate_matrix = np.asarray([[np.cos(cam_agl),-np.sin(cam_agl)],[np.sin(cam_agl),np.cos(cam_agl)]])
    unit_v1_est = np.dot(unit_v0, rotate_matrix)
    dot_product = np.dot(unit_v1, unit_v1_est)
    agl = np.arccos(dot_product) / np.pi * 180

    return agl
