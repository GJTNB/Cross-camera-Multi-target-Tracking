import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F

from models.model import create_model, load_model
from models.decode import mot_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process
from utils.image import get_affine_transform
from models.utils import _tranpose_and_gather_feat
from math import ceil


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float) # 获得目标边界框坐标，用于左上点与宽高表示
        
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.old_xy = [0] * 4
        self.dv = [0.01, 0.01]

    def update_features(self, feat):
        feat /= np.linalg.norm(feat) # 自身特征除以特征的范数
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat # 用指数加权平均（滑动平均）来估计特征
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance) # 卡尔曼滤波预测均值与方差

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks]) # Nx8(N为轨迹数，也可以说是目标数)
            multi_covariance = np.asarray([st.covariance for st in stracks]) # Nx8x8
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            # 更新均值与方差
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    # 激活一个新的轨迹
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    # 更新一个已匹配的轨迹
    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.old_xy = self.tlwh_to_xyah(self.tlwh)
        new_tlwh = new_track.tlwh
        tmp_xy = self.tlwh_to_xyah(new_tlwh)
        tmp_xy[0] = ceil(tmp_xy[0] - 960)
        tmp_xy[1] = ceil(tmp_xy[1] - 540)
        self.old_xy[0] = ceil(self.old_xy[0] - 960)
        self.old_xy[1] = ceil(self.old_xy[1] - 540)
        if self.old_xy[0] != None:
            self.dv = tmp_xy[:2] - self.old_xy[:2]
            self.dv[1] = - self.dv[1] # 图像中的坐标中y轴向下为正方向，故实际中要取反
            if np.abs(self.dv[0]) < 1 and np.abs(self.dv[1]) < 1:
                self.dv = [0.01, 0.01]
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret


    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    # @jit(nopython=True)        
    def xy(self):

        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]

        return ret


    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')

        # 创建模型并使用预测模式
        """
            opt.arch：表示所要使用的模型结构，这里为dla_34；
            opt.head_conv：表示输出头的卷积通道模式；
            opt.heads：{'hm': opt.num_classes,
                        'wh': 2 if not opt.ltrb else 4,
                        'id': opt.reid_dim}；
            opt.reid_dim：表示输出reid的特征维度
        """
        self.model = create_model(opt.arch, opt.heads, opt.head_conv) 
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []

        self.frame_id = 0
        self.det_thresh = opt.conf_thres # 追踪阈值，设置为0.4
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = 800 # self.buffer_size # 猜测是轨迹存在的最大帧数
        self.max_per_image = opt.K # 一帧内可以存在的最大的目标数，设置为500
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter() # 创建卡尔曼滤波


    # 将坐标转换为(x, 5)的形式，5代表坐标与置信度
    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy() # 从计算图中剥离，不在参与反向传播
        dets = dets.reshape(1, -1, dets.shape[2]) # (1, 500, 6)
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes) # num_classes值为1
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    # 融合输出（多类别时有用，注释掉后简单修改程序仍可运行成功）
    def merge_outputs(self, detections):
        results = {}

        # 根据所要检测的类别数目在axis=0上进行连接，在这里我们只追踪人像故只有类别1
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)]) # 将置信度得分按行堆叠
        
        # 只要一个类别所以只有500，不会进入条件语句
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1 # 帧数+1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1] # 1920
        height = img0.shape[0] # 1080
        inp_height = im_blob.shape[2] # 608
        inp_width = im_blob.shape[3] # 1088
        c = np.array([width / 2., height / 2.], dtype=np.float32) # (960, 540)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0 # 1932.63...
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio, # 152, opt.down_ratio称为输出步长
                'out_width': inp_width // self.opt.down_ratio} # 272

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_() # hm.shape=(1, 1, 152, 272)
            wh = output['wh'] # wh.shape=(1, 4, 152, 272) # width & height 
            id_feature = output['id'] # id_feature.shape=(1, 128, 152, 272)
            id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg'] if self.opt.reg_offset else None # reg.shape=(1, 2, 152, 272)
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K) # dets中包含有坐标信息，置信度得分和类别得分 dets.shape=(1, 500, 6), inds.shape=(1, 500) opt.K为输出的最大目标数量
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy() # id_feature.shape=[500, 128]

        dets = self.post_process(dets, meta) # 字典类型，(1, 500, 6) -> (500, 5) 只有坐标信息与置信度，类别得分被单独提取出去
        dets = self.merge_outputs([dets])[1] # (500, 5) # 这里只有人单个类别所以此行代码可有可无

        remain_inds = dets[:, 4] > self.opt.conf_thres # 只有置信度大于阈值的才留下
        dets = dets[remain_inds] # (number of people, 5)，应该表示左上右下点坐标与置信度
        id_feature = id_feature[remain_inds] # (number of people, 128)

        if len(dets) > 0:
            '''Detections'''
            """
            tlbrs:表示所检测到的目标边界框的坐标，用左上与右下两点表示；
            f:表示目标的特征
            """
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
            
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            # 如果处于"activate"状态则将其添加到tracked_stracks中，否则添加到"unconfirmed"轨迹列表中
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks) # 轨迹池
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool) # 预测均值与方差
        dists = matching.embedding_distance(strack_pool, detections) # 计算特征距离
        # print(dists)
        # dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections) # 融合运动特征

        """
        matches:行数表示轨迹id，列数表示检测id。（猜测：两者相等表示匹配成功）
        u_tracks:未匹配的轨迹
        u_detection:未匹配的检测
        """
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track) # 往已激活轨迹列表中添加匹配成功的轨迹
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections) # 计算IOU距离
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5) # 第二次匹配, u_track有可能表示帧目标检测丢失的轨迹

        for itracked, idet in matches:

            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost: # 当目标消失超过设定的最大帧数即将该目标的轨迹删除
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked] # 留下标记为tracked的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks) # 增加新被重新激活的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks) # 增加重新找到的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks) # 删除掉已跟踪到轨迹的列表中已经跟丢的轨迹
        self.lost_stracks.extend(lost_stracks) # 增加新跟丢的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks) # 删除掉跟丢轨迹的列表中已被删除的轨迹
        self.removed_stracks.extend(removed_stracks) # 增加新删除的轨迹
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks) # 删除重复的轨迹
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):

    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb