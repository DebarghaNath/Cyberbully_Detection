import mediapipe as mp
import cv2
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

mp_pose = mp.solutions.pose
bodyPose = mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=2
                       )

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
                      static_image_mode=True,
                      max_num_hands=2,
                      model_complexity=1)

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(
          static_image_mode=True,
          refine_landmarks=True,
          max_num_faces=1
          )

gun_knife = None

def load_yolo(model_path):
    global gun_knife
    gun_knife = YOLO(model_path)

def mediaPipe_bodyPose(image):
    return bodyPose.process(image)

def mediaPipe_handGesture(image):
    return hands.process(image)

def mediaPipe_faceMesh(image):
    return face.process(image)

def Yolo_GunKnife(image):
    return  gun_knife.predict(image)

def preprocess(folder_path, target_size=(256, 256)):

  processed_image = []
  image_file = []
  for file in os.listdir(folder_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
      image_path = os.path.join(folder_path,file)
      image = cv2.imread(image_path)
      if image is None or image.size == 0:
        print(f"skipping the image at {image_path}")
        continue
      image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
      processed_image.append(image)
      image_file.append(file)
  print(f"Loaded {len(processed_image)} valid images from '{folder_path}'")
  return processed_image, image_file



def run_pipeline(folder_path):
  imageset, image_files = preprocess(folder_path)
  results = {}
  with ThreadPoolExecutor(max_workers=3) as executor:
      future_model = {}
      for idx,image in enumerate(imageset):
        image_name = image_files[idx]
        results[image_name] = {"Pose": None, "Hands": None, "Face": None}
        future_model[executor.submit(mediaPipe_bodyPose, image)] = ('Pose', image_name)
        future_model[executor.submit(mediaPipe_handGesture, image)] = ('Hands', image_name)
        future_model[executor.submit(mediaPipe_faceMesh, image)] = ('Face', image_name)
        future_model[executor.submit(Yolo_GunKnife, image)] = ('Gun&Knife', image_name)
      for future in as_completed(future_model):
        model_name, image_name = future_model[future]
        try:
            result = future.result()
            results[image_name][model_name] = result
        except Exception as e:
            print(f"Error in {model_name} for {image_name}: {e}")
      print(f"Completed processing {len(imageset)} images.")
      return results

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", type=str, default="images", help="Path to folder of images")
  parser.add_argument("--model", type=str, default="/content/Weapon_Detection/gun_knife.pt",
                    help="Path to YOLO model weights")
  args = parser.parse_args()
  load_yolo(args.model)
  results = run_pipeline(args.folder)
  print("Pipeline finished successfully.")


