from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing


def output(opt):
    # result_root = opt.. if opt.output_root != '' else '.'
    # mkdir_if_missing(result_root)

    output_video_path = osp.join("../results/", 'xxx.mp4') # "xxx"为你的输出视频名称
    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join("../results", 'x'), output_video_path) # "x"为你的摄像头编号
    os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    output(opt)
