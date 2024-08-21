"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


#将bbox由[x1,y1,x2,y2]形式转为 [框中心点x,框中心点y,框面积s,宽高比例r]^T
def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1)) #将数组转为4行一列形式，即[x,y,s,r]^T


#将[x,y,s,r]形式的bbox，转为[x1,y1,x2,y2]形式
def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  此类表示作为bbox观察到的单个跟踪对象的内部状态。This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    使用初始边界框初始化跟踪器。
    Initialises a tracker using initial bounding box.
    """
    # define constant velocity model
    # 定义恒速模型

    # 状态变量是7维向量， 观测值是4维向量，按照需要的维度构建目标
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    # 状态转移矩阵 7*7
    self.kf.F = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],  
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
    # 观测矩阵 4*7
    self.kf.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

    # 测量噪声的协方差
    self.kf.R[2:,2:] *= 10.
    # 状态协方差矩阵，变化率不可观测所以设置一个较大值表示其较大的不确定性
    self.kf.P[4:,4:] *= 1000. #对未观测到的初始速度给出高的不确定性give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.           # 默认定义的协方差矩阵是np.eye(dim_x)，将P中的数值与10， 1000相乘，赋值不确定性
    # 过程噪声协方差矩阵
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    #状态向量前面四个值用bbox初始化，变化率设置为0
    self.kf.x[:4] = convert_bbox_to_z(bbox)#将bbox转为 [x,y,s,r]^T形式，赋给状态变量X的前4位
    
    # 滤波器生命周期的管理是通过几个变量来实现的，KalmanBoxTracker创建的时候会初始化几个变量：
    self.time_since_update = 0 # 表示现在和上一次更新的时间间隔
    self.id = KalmanBoxTracker.count # 0~based, 得到结果时候会+1, MOT评测时是1~based
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0 # hit_streak表示Tracker连续匹配成功并更新的次数，一旦调用predict()函数对当前帧做了预测，time_since_update就加一，表示其已经对当前帧做过一次预测了。
    self.age = 0

  # 更新:
  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  # 预测:
  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

'''
brief: 将检测的结果与跟踪的结果进行关联
param {*} detections
param {*} trackers
param {*} iou_threshold
return {*}
'''
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  # 如果跟踪器为空,直接返回。例如第一帧
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  # 检测器检测到的框与跟踪器预测的框之间的IOU矩阵
  iou_matrix = iou_batch(detections, trackers) # m*n

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32) # 将m*n的IOU矩阵改为值为01矩阵，元素大于iou_threshold的为1，小于iou_threshold的为0
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        # 如果已经满足每一行相加或每一列相加等于1的约束条件了（一对一匹配）
        # 取出来a矩阵里面等于一的元素的坐标（注意where(a)返回的是两个array，分别是出现1的行，出现1的列。所以需要stack堆叠一下）
        matched_indices = np.stack(np.where(a), axis=1) 
    else:
      #如果不满足约束条件，那么就使用匈牙利算法进行任务分配
      #加上负号是因为linear_assignment求的是最小代价组合，而我们需要的是IOU最大的组合方式，所以取负号
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  # print(matched_indices)
  """matched_indices形状类似于
      [[0 0]
      [1 2]
      [2 3]
      [3 5]
      [4 6]
      [5 1]
      [6 7]]
  """

  unmatched_detections = [] #未匹配上的检测器
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):  #如果检测器中第d个检测结果不在匹配结果索引中，则d未匹配上
      unmatched_detections.append(d)
  unmatched_trackers = [] #未匹配上的跟踪器
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):  #如果跟踪器中第t个跟踪结果不在匹配结果索引中，则t未匹配上
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  # 过滤掉那些IOU较小的匹配对
  matches = []  #存放过滤后的匹配结果
  #遍历粗匹配结果
  for m in matched_indices:
    # m[0]是检测器ID， m[1]是跟踪器ID，如它们的IOU小于阈值则将它们视为未匹配成功
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      # 未匹配成功的追踪器和检测器
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      #将过滤后的匹配对维度变形成1x2形式
      matches.append(m.reshape(1,2))
  # print(matches)
  """matches
  [array([[0, 0]], dtype=int32), array([[1, 3]], dtype=int32), array([[2, 2]], dtype=int32), array([[3, 8]], dtype=int32), array([[4, 4]], dtype=int32), array([[5, 6]], dtype=int32), array([[6, 7]], dtype=int32), array([[7, 1]], dtype=int32), array([[8, 5]], dtype=int32)]
  """
  if(len(matches)==0):
    # 如果过滤后匹配结果为空，那么返回空的匹配结果
    matches = np.empty((0,2),dtype=int)
  else: 
    # 如果过滤后匹配结果非空，则按0轴方向继续添加匹配对
    matches = np.concatenate(matches,axis=0)
  print(matches)
  """matches
    [[ 0  1]
    [ 1  0]
    [ 2  3]
    [ 3  2]
    [ 4  5]
    [ 5 11]
    [ 6  6]
    [ 7  8]
    [ 8  4]]
  """
  #其中跟踪器数组是5列的（最后一列是ID）
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age        # 最大年龄值（未被检测更新的跟踪器随帧数增加），超过之后会被删除
    self.min_hits = min_hits      # 目标命中的最小次数，小于该次数不返回
    self.iou_threshold = iou_threshold
    self.trackers = []  # 列表，存放KalmanBoxTracker类型，后面会将每个
    self.frame_count = 0

  '''
  brief: 追踪器更新
  param {*} self
  param {*} dets 检测结果,形式为 m*5 m为该帧检测到的物体数量,每一行为该物体的[x1,y1,x2,y2,score] 
  param {*} 5
  return {*}
  '''
  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    每一帧都得调用一次，即便检测结果为空
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    返回相似的数组,最后一列是目标ID
    Returns the a similar array, where the last column is the object ID.

    返回的目标数量可能与提供的检测数量不同
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1 # 帧计数
    # get predicted locations from existing trackers.
    # 根据当前所有卡尔曼跟踪器的个数创建零矩阵，维度为：卡尔曼跟踪器ID个数x 5 (这5列内容为bbox与ID)
    trks = np.zeros((len(self.trackers), 5))
    to_del = []   # 存放待删除
    ret = []      # 存放最后返回的结果
    # 循环遍历卡尔曼跟踪器列表
    for t, trk in enumerate(trks):
      # 1.预测（对上一帧的物体在这一帧中的位置做预测）
      pos = self.trackers[t].predict()[0]
      # 用卡尔曼跟踪器t预测上一帧中的物体在当前帧中可能的bbox位置
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)): 
        # 如果预测的bbox为空，那么将第t个卡尔曼跟踪器删除
        to_del.append(t)
    # 将预测为空的卡尔曼跟踪器所在行删除，最后trks中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空bbox
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    # 对to_del数组进行倒序遍历
    for t in reversed(to_del):
      # 从跟踪器中删除 to_del中的上一帧跟踪器ID
      self.trackers.pop(t)

    # 2.关联
    # 对传入的检测结果 与 上一帧跟踪物体在当前帧中预测的结果做关联
    # 返回匹配matched, 未匹配到的检测器unmatched_dets（数组，每一个元素是检测器的索引）,未匹配到的追踪器unmatched_trks
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # 3.更新
    # update matched trackers with assigned detections
    for m in matched:
      """m
      [idx_detector,idx_tracker]
      """
      self.trackers[m[1]].update(dets[m[0], :]) #用检测器更新其对应的追踪器

    # create and initialise new trackers for unmatched detections
    # 对于新增的未匹配的检测结果，创建新的跟踪器并初始化
    # 新增目标
    for i in unmatched_dets:
        #将新增的未匹配的检测结果dets[i,:]传入KalmanBoxTracker
        trk = KalmanBoxTracker(dets[i,:])
        #将新创建和初始化的跟踪器trk 传入trackers
        self.trackers.append(trk)
    i = len(self.trackers)
    
    # 对更新之后的卡尔曼跟踪器集进行倒序遍历
    for trk in reversed(self.trackers):
        #获取trk跟踪器的状态 [x1,y1,x2,y2]
        d = trk.get_state()[0]
        # Tracker必须是更新过的，只有update了之后，time_since_update才是0.
        # 同时，一个匹配成功的Tracker，需要判断其是否还在“试用期”，只有连续几帧都匹配成功才能使用它的跟踪信息，当然如果是刚开始的几帧是例外：
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        # 如果Tracker在max_age这么多帧中一直都没有更新过，就说明一直未匹配成功，该Tracker就会被删除：
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

'''
brief: 加载参数
return {*}
'''
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
#----------处理结果可视化---------------------
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()#交互模式，可以画动态图
    fig = plt.figure()#创建画布
    ax1 = fig.add_subplot(111, aspect='equal')# 子图，第一个数字指定要创建具有几行子图的规格，第二个数字指列数的规格，第三个数字指对全部子图的行列从左往右，从上往下的第几个子图
# ----------------------------------

  # 创建output文件夹
  if not os.path.exists('output'):
    os.makedirs('output')
  # data/train/*/det/det.txt
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  #遍历pattern路径下的所有的det.txt文件，glob返回所有匹配的文件路径构成的列表
  for seq_dets_fn in glob.glob(pattern):
    # 对每一个数据集都创建SORT实例
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    # 读取txt文件
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    # 获得train下面的文件名（e.g. ADL-Rundle-6...）
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    # 相对路径是 data/train/ * /det/det.txt
    # os.path.sep 自动获取系统对应的路径分隔符，对于ubuntu系统是 "/"
    # seq_dets_fn[pattern.find('*'):]------->  */det/det.txt  
    # split后取"*"对应的部分-------> *（e.g. ADL-Rundle-6...）


    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:  # 顺便把output/ADL-Rundle-6.txt等文件创建了
      print("Processing %s."%(seq)) # 打印一下正在处理的序列名称 例如 "ADL-Rundle-6"
      # 第一列中最大的那个数字 对应一共有几帧seq_dets[:,0].max()
      for frame in range(int(seq_dets[:,0].max())):
        # 先加1 检测和帧数都从1开始计数。因为数据集的det.txt文件里面 帧计数是从1开始的，没有第0帧
        frame += 1 # detection and frame numbers begin at 1
        # 获取frame帧的检测的那几行，但是只截取3～7列，分别是左上坐标x,y,以及宽高wh,目标检测表示的置信度得分
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]转化坐标格式为[左上角坐标，右下角坐标]（右下=左上＋长宽） [x1,y1,x2,y2] = [x1,y1,x1+w,y1+h]
        total_frames += 1

        if(display):
          # 这帧图片的路径
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im = io.imread(fn)
          ax1.imshow(im) # 显示图片
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        
        # * sort跟踪器更新
        trackers = mot_tracker.update(dets)
        
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
