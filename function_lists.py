#list of functions we use in train.py and remaster_image.py
import numpy as np
from PIL import Image
from torchvision import transforms
import os


#search files in folder, return their path as list
def file_search(dir_path):
    file_li = []
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_li.append(file_path)

    return file_li

#convert image to tensor type
def img_to_tensor_data(img):
    im_row = 128
    im_col = 128
    img_reshape = transforms.Resize((im_row, im_col))
    img = img_reshape(img)
    transform = transforms.Compose([transforms.PILToTensor()])
    tensor_image = transform(img)
    tensor_image = tensor_image / 255
    return tensor_image

#convert tensor type to image
def data_to_img(data):
    data_image = data * 255
    data_image = data_image.cpu().permute(1, 2, 0).detach().numpy().astype(np.uint8)
    im = Image.fromarray(data_image)
    return im