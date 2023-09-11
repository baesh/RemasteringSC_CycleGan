#change unremastered image to remastered image using trained layer
import torch
from PIL import Image
import os

from layer import Generator
from function_lists import file_search, img_to_tensor_data, data_to_img


model_save_path = './model_log/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Generator_x = Generator().to(device)
Generator_x.load_state_dict(torch.load(os.path.join(model_save_path, 'Generator_x.pth'), map_location=device))

x_file_li = file_search('./before_change')

for i in range(len(x_file_li)):
    x_image = x_file_li[i]
    x_tensor_li = []
    x_img = Image.open(x_image)
    x_tensor_img = img_to_tensor_data(x_img)
    x_tensor_li.append(x_tensor_img)
    x_img_data = torch.stack(x_tensor_li, dim=0).to(device)

    Generator_x.eval()
    result = Generator_x(x_img_data)
    im = data_to_img(result[0])
    im_path = './changed/' + '{0:06d}'.format(i) + '.png'
    im.save(im_path)