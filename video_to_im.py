#convert video into list of images by frame

import cv2
import os
filePath = '.mp4'
if os.path.isfile(filePath):
    cap = cv2.VideoCapture(filePath)
else:
    print("file doesnt exist")

vidcap = cv2.VideoCapture(filePath)
success, image = vidcap.read()
print(success)
count = 0
while success:
    cv2.imwrite("./before_change/%06d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    if count == 5000:
        break

print("finish! convert video to frame")