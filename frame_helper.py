import cv2
import numpy as np
from itertools import count
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

w = 1164
h = 874
F = 910

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

K = np.array([
    [F, 0, w//2],
    [0, F, h//2],
    [0, 0, 1]])

K_inv = np.linalg.inv(K)

W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)

def normalize(x):
  x = add_ones_dimension(x)
  return (K_inv @ x.T).T[:, 0:2]

def denormalize(x):
  return (K @ x)[:-1]

def add_ones_dimension(x):
  return np.pad(x, pad_width = ((0,0), (0,1)) ,constant_values=1)

def remove_ones_dimension(x):
  return x[:,[0,1]]

def generate_frames(vid_path):
    video = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    _, prev_frame = video.read()
    for t in count():
      ret, curr_frame = video.read()
      if not ret:
        break
      yield prev_frame, curr_frame
      prev_frame = curr_frame
    video.release()
    cv2.destroyAllWindows()

def extractFeatures(frame):
  orb = cv2.ORB_create()
  kps = orb.detect(frame, None)
  kps, des = orb.compute(frame, kps)

  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), np.array(des)

def bfmatcher(kps, dess, threshold = 0.01, max_trials = 200):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(dess[0], dess[1], k=2)
  res = []

  for m,n in matches:
    if m.distance < 0.70 * n.distance:
        kp1 = kps[0][m.queryIdx]
        kp2 = kps[1][m.trainIdx]
        res.append((kp1, kp2))
  res = np.array(res)

  assert len(res)>=8, 'not enough points'

  model, inliers = ransac((normalize(res[:,0]), normalize(res[:,1])),
                          #FundamentalMatrixTransform, 
                          EssentialMatrixTransform,
                          min_samples=8,
                          residual_threshold=threshold, 
                          max_trials=max_trials)
  
  U, D, Vt = np.linalg.svd(model.params)

  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R1 = U @ W @ Vt
  R2 = U @ W.T @ Vt
  if abs(np.sum(R1.diagonal())-3) > abs(np.sum(R2.diagonal())-3):
    R = R2
  else: 
    R = R1
  t = U[:, 2]
  if t[2] < 0:
    t *= -1
  #pose = poseRt(R, t)
  #print (t)
  return res[inliers,0], res[inliers,1]