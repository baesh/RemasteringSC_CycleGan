#create video with images in certain file

import cv2
import glob

img_array = []
for filename in glob.glob('changed/*.png'):
    print(filename)
    img = cv2.imread(filename)
    img = cv2.resize(img, (128, 128))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('created.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
