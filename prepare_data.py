import os

import cv2
import numpy as np

from utils import FaceLandmarks


data_dir = './movies'

count = 0
output = []
emotion_count = {}
issues = []

fl = FaceLandmarks()
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        face_landmarks = fl.get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)
            print("OK ", count, emotion_indx, emotion, image_path)
            if emotion not in emotion_count:
                emotion_count[emotion] = 0
            emotion_count[emotion] += 1
        else:
            print("NOK", count, emotion_indx, emotion, image_path)
            issues.append(image_path)
        count+=1

for e, c in emotion_count.items():
    print(e, c)

print('file with issues:')
for i in issues:
    print(i)

np.savetxt('datadb5.txt', np.asarray(output))
