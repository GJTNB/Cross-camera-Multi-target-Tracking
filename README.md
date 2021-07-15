# 跨摄像头多目标跟踪（Cross-cam Multi-target Tracking）

本项目将多目标跟踪与行人重识别相结合，设计了一个跨摄像头多目标跟踪系统，能够完成两个摄像头下目标的跟踪与识别。

所使用的深度学习框架为Pytorch，版本为1.4.0+cu92。

其中多目标跟踪与行人重识别的部分分别参考了以下两个优秀开源项目：
- https://github.com/ifzhang/FairMOT
- https://github.com/layumi/Person_reID_baseline_pytorch

使用方法：
1. 将所要处理的视频文件放入videos文件夹中（若处理的是实时的视频流可不用）；
2. 打开cam.txt文本，将刚才放入videos中的视频文件相对路径写入（若实时的视频流可根据接入电脑的端口写入，一般默认为0和1）；
3. 运行demo.py文件；
4. 在弹出的视频框中框选高度重叠的匹配区域，后按回车即可开始运行跨摄像头多目标跟踪系统。

