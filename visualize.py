import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(path, image_name, result_dict):
    image_path = os.path.join(path, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = result_dict[image_name]

    print("Output from MediaPipe.Face")
    face_img = image.copy()
    face_result = result["Face"]
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                h, w, _ = face_img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(face_img, (cx, cy), 1, (255, 0, 255), -1)
    plt.figure(figsize=(8, 8))
    plt.imshow(face_img)
    plt.axis('off')
    plt.title(f"Face Detections: {os.path.basename(image_path)}")
    plt.show()

    print("Output from MediaPipe.Hands")
    hands_img = image.copy()
    hands_result = result["Hands"]
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                h, w, _ = hands_img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(hands_img, (cx, cy), 2, (255, 255, 0), -1)
    plt.figure(figsize=(8, 8))
    plt.imshow(hands_img)
    plt.axis('off')
    plt.title(f"Hand Detections: {os.path.basename(image_path)}")
    plt.show()

    print("Output from MediaPipe.Pose")
    pose_img = image.copy()
    pose_result = result["Pose"]
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks.landmark:
            h, w, _ = pose_img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(pose_img, (cx, cy), 2, (0, 255, 0), -1)
    plt.figure(figsize=(8, 8))
    plt.imshow(pose_img)
    plt.axis('off')
    plt.title(f"Pose Detections: {os.path.basename(image_path)}")
    plt.show()

    print("Output from YOLO Gun&Knife")  
    yolo_results = result["Gun&Knife"][0]
    boxes = getattr(yolo_results, "boxes", None)
    names = getattr(yolo_results, "names", {})

    yolo_img = image.copy() 
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
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
        for i, (b, c, conf) in enumerate(zip(xyxy_arr, cls_arr, conf_arr)):
            x1, y1, x2, y2 = map(int, b.tolist()) 
            cls_idx = int(c)
            label = names.get(cls_idx, str(cls_idx))
            confidence = float(conf)

            print(f" → {label} detected with confidence {confidence:.2f} bbox=[{x1},{y1},{x2},{y2}]")
            cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(yolo_img, f"{label} {confidence:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        print("No YOLO boxes found for this image.")

    plt.figure(figsize=(8, 8))
    plt.imshow(yolo_img)
    plt.axis('off')
    plt.show()
