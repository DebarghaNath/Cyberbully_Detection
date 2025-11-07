import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
##==========================POSE===========================================
POSE_IDX = {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
        'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
        'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
        'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
        'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }

def normalize_landmarks(landmarks,eps=1e-4, rotate=True):
    
    lm = landmarks.copy().astype(np.float32)
    
    shoulder_exist = True
    hip_exist = True

    try:
      lh = lm[POSE_IDX['left_hip']][:2]
      rh = lm[POSE_IDX['right_hip']][:2]
    except Exception:
      hip_exist = False

    try: 
      ls = lm[POSE_IDX['left_shoulder']][:2]
      rs = lm[POSE_IDX['right_shoulder']][:2]
    except Exception:
      shoulder_exist = False
    
    if not hip_exist and not shoulder_exist:
        return lm

    if hip_exist and shoulder_exist:
        mid_hips = (lh + rh) / 2.0
        mid_shoulders = (ls + rs) / 2.0
        torso_len = np.linalg.norm(mid_shoulders - mid_hips)
        shoulder_width = np.linalg.norm(ls - rs)
        if torso_len > eps:
            scale = torso_len
        elif shoulder_width > eps:
            scale = shoulder_width
        else:
            scale = 1.0
    elif shoulder_exist:
        shoulder_width = np.linalg.norm(ls - rs)
        scale = shoulder_width if shoulder_width > eps else 1.0
    else: 
        hip_width = np.linalg.norm(lh - rh)
        scale = hip_width if hip_width > eps else 1.0

    lm[:, :2] = lm[:, :2] / (scale + eps)
    return lm

def pose_feature_calculator(pose_result):
    if not pose_result or not pose_result.pose_landmarks:
        return np.zeros(24, dtype=np.float32)
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark])

    nlm = normalize_landmarks(landmarks)

    def dist_n(a, b):
        return np.linalg.norm(nlm[a] - nlm[b])

    def angle_n(a, b, c):
        v1 = nlm[a] - nlm[b]
        v2 = nlm[c] - nlm[b]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cosang = np.dot(v1, v2) / denom
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    shoulder_width = dist_n(POSE_IDX['left_shoulder'], POSE_IDX['right_shoulder'])
    hip_width = dist_n(POSE_IDX['left_hip'], POSE_IDX['right_hip'])
    left_forearm = dist_n(POSE_IDX['left_elbow'], POSE_IDX['left_wrist'])
    right_forearm = dist_n(POSE_IDX['right_elbow'], POSE_IDX['right_wrist'])
    left_upper_arm = dist_n(POSE_IDX['left_shoulder'], POSE_IDX['left_elbow'])
    right_upper_arm = dist_n(POSE_IDX['right_shoulder'], POSE_IDX['right_elbow'])
    left_thigh = dist_n(POSE_IDX['left_hip'], POSE_IDX['left_knee'])
    right_thigh = dist_n(POSE_IDX['right_hip'], POSE_IDX['right_knee'])
    left_leg = dist_n(POSE_IDX['left_knee'], POSE_IDX['left_ankle'])
    right_leg = dist_n(POSE_IDX['right_knee'], POSE_IDX['right_ankle'])
    left_torso_height = dist_n(POSE_IDX['left_shoulder'], POSE_IDX['left_hip'])
    right_torso_height = dist_n(POSE_IDX['right_shoulder'], POSE_IDX['right_hip'])
    left_foot_ai = dist_n(POSE_IDX['left_ankle'], POSE_IDX['left_index'])
    left_foot_ah = dist_n(POSE_IDX['left_ankle'], POSE_IDX['left_heel'])
    left_foot_ih = dist_n(POSE_IDX['left_heel'], POSE_IDX['left_index'])
    right_foot_ai = dist_n(POSE_IDX['right_ankle'], POSE_IDX['right_index'])
    right_foot_ah = dist_n(POSE_IDX['right_ankle'], POSE_IDX['right_heel'])
    right_foot_ih = dist_n(POSE_IDX['right_heel'], POSE_IDX['right_index'])

    left_elbow_angle = angle_n(POSE_IDX['left_shoulder'], POSE_IDX['left_elbow'], POSE_IDX['left_wrist'])
    right_elbow_angle = angle_n(POSE_IDX['right_shoulder'], POSE_IDX['right_elbow'], POSE_IDX['right_wrist'])
    left_knee_angle = angle_n(POSE_IDX['left_hip'], POSE_IDX['left_knee'], POSE_IDX['left_ankle'])
    right_knee_angle = angle_n(POSE_IDX['right_hip'], POSE_IDX['right_knee'], POSE_IDX['right_ankle'])
    torso_inclination = angle_n(POSE_IDX['left_shoulder'], POSE_IDX['left_hip'], POSE_IDX['right_hip'])
    neck_angle = angle_n(POSE_IDX['left_shoulder'], POSE_IDX['nose'], POSE_IDX['right_shoulder'])

    feature_vec = [
        shoulder_width, hip_width,
        left_forearm, right_forearm, left_upper_arm, right_upper_arm,
        left_thigh, right_thigh, left_leg, right_leg,
        left_torso_height, right_torso_height,
        left_foot_ai, left_foot_ah, left_foot_ih,
        right_foot_ai, right_foot_ah, right_foot_ih,
        left_elbow_angle, right_elbow_angle,
        left_knee_angle, right_knee_angle,
        torso_inclination, neck_angle
    ]

    return np.array(feature_vec, dtype=np.float32)

def pose_feature_extractor(result_dict):
    features = []
    pose_result = result_dict.get("Pose")
    if pose_result and pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0] * 33 * 4)
    return np.array(features, dtype=np.float32)


def get_pose_feature(detailed, result_dict):
    pose_result = result_dict.get("Pose")
    basic_features = pose_feature_extractor(result_dict)
    detailed_features = pose_feature_calculator(pose_result)

    if detailed:
        return np.concatenate([basic_features, detailed_features])
    else:
        return basic_features

##==========================HAND===========================================
HAND_IDX = {
    'wrist': 0,

    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,

    'index_mcp': 5,
    'index_pip': 6,
    'index_dip': 7,
    'index_tip': 8,

    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,

    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,

    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20
}

def joint_angle_deg(a_idx, b_idx, c_idx, lm):
    a = lm[a_idx][:3]
    b = lm[b_idx][:3]
    c = lm[c_idx][:3]
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cosang = np.dot(v1, v2) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def finger_angle_calculator(hand_landmarks):
    pts = getattr(hand_landmarks, "landmark", hand_landmarks)
    lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)

    mcp_angles = []
    mcp_angles.append(joint_angle_deg(HAND_IDX['index_pip'], HAND_IDX['index_mcp'], HAND_IDX['wrist'], lm))
    mcp_angles.append(joint_angle_deg(HAND_IDX['middle_pip'], HAND_IDX['middle_mcp'], HAND_IDX['wrist'], lm))
    mcp_angles.append(joint_angle_deg(HAND_IDX['ring_pip'], HAND_IDX['ring_mcp'], HAND_IDX['wrist'], lm))
    mcp_angles.append(joint_angle_deg(HAND_IDX['pinky_pip'], HAND_IDX['pinky_mcp'], HAND_IDX['wrist'], lm))
    mcp_angles.append(joint_angle_deg(HAND_IDX['thumb_cmc'], HAND_IDX['thumb_mcp'], HAND_IDX['thumb_ip'], lm))
    mcp_angles = np.array(mcp_angles, dtype=np.float32) / 180.0

    pip_angles = []
    pip_angles.append(joint_angle_deg(HAND_IDX['index_mcp'], HAND_IDX['index_pip'], HAND_IDX['index_dip'], lm))
    pip_angles.append(joint_angle_deg(HAND_IDX['middle_mcp'], HAND_IDX['middle_pip'], HAND_IDX['middle_dip'], lm))
    pip_angles.append(joint_angle_deg(HAND_IDX['ring_mcp'], HAND_IDX['ring_pip'], HAND_IDX['ring_dip'], lm))
    pip_angles.append(joint_angle_deg(HAND_IDX['pinky_mcp'], HAND_IDX['pinky_pip'], HAND_IDX['pinky_dip'], lm))
    pip_angles = np.array(pip_angles, dtype=np.float32) / 180.0

    return np.concatenate([mcp_angles, pip_angles])

def finger_length_calculator(hand_landmarks):
  pts = getattr(hand_landmarks, "landmark", hand_landmarks)
  lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
  wrist_pos = lm[HAND_IDX['wrist']][:3]

  tips_idx = [
      HAND_IDX['thumb_tip'],
      HAND_IDX['index_tip'],
      HAND_IDX['middle_tip'],
      HAND_IDX['ring_tip'],
      HAND_IDX['pinky_tip']
  ]

  finger_lengths = np.array([np.linalg.norm(lm[t][:3] - wrist_pos) for t in tips_idx], dtype=np.float32)

  tip_points = lm[tips_idx, :3]
  max_tip_tip = 0.0
  for i in range(len(tip_points)):
      for j in range(i + 1, len(tip_points)):
          d = np.linalg.norm(tip_points[i] - tip_points[j])
          if d > max_tip_tip:
              max_tip_tip = d

  max_wrist_tip = finger_lengths.max() if finger_lengths.size > 0 else 0.0
  scale = max(max_wrist_tip, max_tip_tip, 1e-6)

  return finger_lengths / scale 

def hand_feature_calculator(hand_result):
  if not hand_result or not getattr(hand_result, "multi_hand_landmarks", None):
      return np.zeros((2, 14), dtype=np.float32)

  all_hand_features = []
  for hand_landmarks in hand_result.multi_hand_landmarks:
      finger_lengths = finger_length_calculator(hand_landmarks)
      finger_angles = finger_angle_calculator(hand_landmarks)
      hand_feat = np.concatenate([finger_lengths, finger_angles]).astype(np.float32)
      all_hand_features.append(hand_feat)

  while len(all_hand_features) < 2:
      all_hand_features.append(np.zeros(14, dtype=np.float32))

  return np.array(all_hand_features, dtype=np.float32)

def hand_feature_extractor(result_dict):
  hand_result = result_dict.get("Hands")
  all_hand_features = []

  if hand_result and getattr(hand_result, "multi_hand_landmarks", None):
      for hand_landmarks in hand_result.multi_hand_landmarks:
          hand_features = []
          for lm in hand_landmarks.landmark:
              hand_features.extend([lm.x, lm.y, lm.z])
          all_hand_features.append(hand_features)

  while len(all_hand_features) < 2:
      all_hand_features.append([0.0] * 21 * 3)

  return np.array(all_hand_features, dtype=np.float32)

def get_hand_feature(detailed, result_dict):
  hand_result = result_dict.get("Hands")
  basic_features = hand_feature_extractor(result_dict)  
  detailed_features = hand_feature_calculator(hand_result) 

  if detailed:
      return np.concatenate([
          basic_features[0], detailed_features[0],
          basic_features[1], detailed_features[1]
      ]).astype(np.float32)
  else:
      return basic_features.flatten().astype(np.float32)
FACE_IDX = {
    'nose_tip': 1,
    'chin': 199,
    'left_eye_outer': 33,
    'right_eye_outer': 263,
    'left_eye_inner': 133,
    'right_eye_inner': 362,
    'left_mouth_corner': 61,
    'right_mouth_corner': 291,
    'left_eyebrow_outer': 70,
    'right_eyebrow_outer': 300
}


##==========================FACE===========================================
def _pt(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

def norm_dist(landmarks, idx_left, idx_right):
    l = _pt(landmarks[idx_left])
    r = _pt(landmarks[idx_right])
    return float(np.linalg.norm(l[:2] - r[:2]))

def face_feature_calculator(face_result):
  if not face_result or not getattr(face_result, "multi_face_landmarks", None):
    return np.zeros(11+30, dtype=np.float32)

  lm = face_result.multi_face_landmarks[0].landmark

  d_interocular = norm_dist(lm, FACE_IDX['left_eye_outer'], FACE_IDX['right_eye_outer'])
  d_eye_corners = norm_dist(lm, FACE_IDX['left_eye_outer'], FACE_IDX['right_eye_outer'])
  d_nose_chin = norm_dist(lm, FACE_IDX['nose_tip'], FACE_IDX['chin'])
  d_mouth_width = norm_dist(lm, FACE_IDX['left_mouth_corner'], FACE_IDX['right_mouth_corner'])
  d_eye_to_mouth = norm_dist(lm, FACE_IDX['nose_tip'], FACE_IDX['left_mouth_corner'])
  d_eyebrow_span = norm_dist(lm, FACE_IDX['left_eyebrow_outer'], FACE_IDX['right_eyebrow_outer'])

  scale = d_interocular if d_interocular > 1e-8 else 1.0
  r_nose_chin = d_nose_chin / scale
  r_mouth_width = d_mouth_width / scale
  r_eye_to_mouth = d_eye_to_mouth / scale
  r_eyebrow_span = d_eyebrow_span / scale
  r_mouth_eye = d_mouth_width / (d_eye_corners + 1e-8)

  left_eye_outer = _pt(lm[FACE_IDX['left_eye_outer']])[:2]
  right_eye_outer = _pt(lm[FACE_IDX['right_eye_outer']])[:2]
  eye_mid = (left_eye_outer + right_eye_outer) / 2.0
  nose_xy = _pt(lm[FACE_IDX['nose_tip']])[:2]
  asym_nose = float(nose_xy[0] - eye_mid[0]) / (scale + 1e-8)

  features = np.array([
      d_eye_corners,
      d_nose_chin,
      d_mouth_width,
      d_eye_to_mouth,
      d_eyebrow_span,
      r_nose_chin,
      r_mouth_width,
      r_eye_to_mouth,
      r_eyebrow_span,
      r_mouth_eye,
      asym_nose
  ], dtype=np.float32)

  return features


def face_basic_extractor(result_dict):
  face_res = result_dict.get("Face")
  if not face_res or not getattr(face_res, "multi_face_landmarks", None):
      return np.zeros(468 * 3, dtype=np.float32)
  landmarks = face_res.multi_face_landmarks[0].landmark
  flat = []
  for lm in landmarks:
      flat.extend([lm.x, lm.y, lm.z])
  return np.array(flat, dtype=np.float32)

def get_face_feature(detailed, result_dict):
  face_res = result_dict.get("Face")
  basic = face_basic_extractor(result_dict)
  detailed_vec = face_feature_calculator(face_res)

  if detailed:
      return np.concatenate([basic, detailed_vec])
  else:
      return basic
##==========================Gun&Knife===========================================
def GunKnife_feature_extractor(result_dict,image_name,image_path):
    image = cv2.imread(os.path.join(image_path,image_name))
    h_orig, w_orig = image.shape[:2]
    w_tgt, h_tgt = 256,256
    sx = float(w_tgt) / float(w_orig)
    sy = float(h_tgt) / float(h_orig)
    yolo_results = result_dict.get("Gun&Knife")
    if yolo_results is None:
        return np.zeros(6, dtype=np.float32)

    if isinstance(yolo_results, (list, tuple)):
        yolo_results = yolo_results[0]

    boxes = getattr(yolo_results, "boxes", None)
    names = getattr(yolo_results, "names", {})
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return np.zeros(6, dtype=np.float32)
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()
    except Exception:
        xyxy_arr = np.array(boxes.xyxy)
    try:
        cls_arr = boxes.cls.cpu().numpy()
    except Exception:
        cls_arr = np.array(boxes.cls)
    try:
        conf_arr = boxes.conf.cpu().numpy()
    except Exception:
        conf_arr = np.array(boxes.conf)

    if len(conf_arr) == 0:
        return np.zeros(6, dtype=np.float32)

    max_idx = int(np.argmax(conf_arr))
    max_conf = float(conf_arr[max_idx])
    x1, y1, x2, y2 = map(int, xyxy_arr[max_idx].tolist())
    cls_idx = int(cls_arr[max_idx])
    label = names.get(cls_idx, str(cls_idx))

    print(f"Top detection: {label} | conf={max_conf:.3f} | bbox=({x1},{y1},{x2},{y2})")
   
    return np.array([x1*sx/w_orig, y1*sy/h_orig, x2*sx/w_orig, y2*sy/h_orig, max_conf, cls_idx], dtype=np.float32)


##================Features================

def get_features(result, details, path):
  all_features = {}

  for image_name in result.keys():
    print(f"Extracting features for: {image_name}")

    pose = get_pose_feature(details, result[image_name])
    print(len(pose))
    face = get_face_feature(details, result[image_name])
    print(len(face))
    hand = get_hand_feature(details, result[image_name])
    print(len(hand))
    gunknife = GunKnife_feature_extractor(result, image_name, path)
    print(len(gunknife))
    combined_features = np.concatenate([pose, face, hand, gunknife])
    all_features[image_name] = combined_features

  return all_features
