import random
import numpy as np
import torch
from torch import nn
from PIL import Image
import os
import gc

from layer import Generator, Discriminator
from function_lists import file_search, img_to_tensor_data, data_to_img



#train by cycle gan method
def train(data_li, model_li, optimizer, device):
    optimizer.zero_grad()

    x_img_data = data_li[0]
    y_img_data = data_li[1]

    Generator_x = model_li[0].train()
    Generator_y = model_li[1].train()
    discriminate_x = model_li[2].train()
    discriminate_y = model_li[3].train()

    fake_y = Generator_x(x_img_data)
    fake_x = Generator_y(y_img_data)

    fake_x_discriminated_result = discriminate_x(fake_x)
    real_x_discriminated_result = discriminate_x(x_img_data)
    fake_y_discriminated_result = discriminate_y(fake_y)
    real_y_discriminated_result = discriminate_y(y_img_data)

    real_x_label = torch.tensor(np.ones((real_x_discriminated_result.size(0), 1, 1, 1)), requires_grad= False).float().to(device)
    fake_x_label = torch.tensor(np.zeros((fake_x_discriminated_result.size(0), 1, 1, 1)) , requires_grad= False).float().to(device)
    real_y_label = torch.tensor(np.ones((real_y_discriminated_result.size(0), 1, 1, 1)) , requires_grad= False).float().to(device)
    fake_y_label = torch.tensor(np.zeros((fake_y_discriminated_result.size(0), 1, 1, 1)) , requires_grad= False).float().to(device)

    bce = nn.BCELoss()
    loss_gan_x = bce(real_x_discriminated_result, real_x_label) + bce(fake_x_discriminated_result, fake_x_label)
    loss_gan_y = bce(real_y_discriminated_result, real_y_label) + bce(fake_y_discriminated_result, fake_y_label)
    l1 = nn.L1Loss()
    loss_cyc = 20 * (l1(x_img_data, Generator_y(fake_y)) + l1(y_img_data, Generator_x(fake_x))) + l1(x_img_data, fake_x) + l1(y_img_data, fake_y)
    gamma = 10
    loss = loss_gan_x + loss_gan_y + gamma * loss_cyc

    #print losses
    print(loss_gan_x, loss_gan_y, loss_cyc)

    loss.backward()
    optimizer.step()

    for model in model_li:
        model.eval()


def main():
    num_epochs = 100000
    learning_rate = 0.000001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_save_path = './model_log/'
    batch = 25
    learning_rate_changed_flag = 0


    Generator_x = Generator().to(device)
    Generator_y = Generator().to(device)
    discriminate_x = Discriminator().to(device)
    discriminate_y = Discriminator().to(device)

    #load data if there is previous data
    flag = os.path.isfile(os.path.join(model_save_path, 'Generator_x.pth')) and os.path.isfile(os.path.join(model_save_path, 'Generator_y.pth')) and os.path.isfile(os.path.join(model_save_path, 'discriminate_x.pth')) and os.path.isfile(os.path.join(model_save_path, 'discriminate_y.pth'))
    if flag:
        print('continue from prev data')
        Generator_x.load_state_dict(torch.load(os.path.join(model_save_path, 'Generator_x.pth'), map_location=device))
        Generator_y.load_state_dict(torch.load(os.path.join(model_save_path, 'Generator_y.pth'), map_location=device))
        discriminate_x.load_state_dict(torch.load(os.path.join(model_save_path, 'discriminate_x.pth'), map_location=device))
        discriminate_y.load_state_dict(torch.load(os.path.join(model_save_path, 'discriminate_y.pth'), map_location=device))
    else:
        print('new start')


    whole_param = list(Generator_x.parameters()) + list(Generator_y.parameters()) + list(discriminate_x.parameters()) + list(discriminate_y.parameters())
    optim = torch.optim.Adam(whole_param, lr=learning_rate)
    model_li = [Generator_x, Generator_y, discriminate_x, discriminate_y]

    #load optimizer if there is previous data
    flag_optim = os.path.isfile(os.path.join(model_save_path, 'optim.pth'))
    if flag_optim:
        print('optimizer_continued')
        optim.load_state_dict(torch.load(os.path.join(model_save_path, 'optim.pth'), map_location=device))
    else:
        print('optim new start')


    for i in range(num_epochs):
        #get random images from ./sample_x_image, and convert them to tensor type
        x_file_li = file_search('./sample_x_image')
        x_file_li = random.sample(x_file_li, batch)
        x_tensor_li = []
        for x_image in x_file_li:
            x_img = Image.open(x_image)
            x_tensor_img = img_to_tensor_data(x_img)
            x_tensor_li.append(x_tensor_img)
        x_img_data = torch.stack(x_tensor_li, dim=0).to(device)

        # get random images from ./sample_y_image, and convert them to tensor type
        y_file_li = file_search('./sample_y_image')
        y_file_li = random.sample(y_file_li, batch)
        y_tensor_li = []
        for y_image in y_file_li:
            y_img = Image.open(y_image)
            y_tensor_img = img_to_tensor_data(y_img)
            y_tensor_li.append(y_tensor_img)
        y_img_data = torch.stack(y_tensor_li, dim=0).to(device)


        data_li = [x_img_data, y_img_data]
        train(data_li, model_li, optim, device)
        gc.collect()
        torch.cuda.empty_cache()

        if i%100 == 0:
            # save current state
            torch.save(Generator_x.state_dict(), os.path.join(model_save_path, 'Generator_x.pth'))
            torch.save(Generator_y.state_dict(), os.path.join(model_save_path, 'Generator_y.pth'))
            torch.save(discriminate_x.state_dict(), os.path.join(model_save_path, 'discriminate_x.pth'))
            torch.save(discriminate_y.state_dict(), os.path.join(model_save_path, 'discriminate_y.pth'))
            torch.save(optim.state_dict(), os.path.join(model_save_path, 'optim.pth'))

            #save created image in ./results_test/
            Generator_x.eval()
            result = Generator_x(x_img_data[0:batch])
            im = data_to_img(result[1])
            im_path = './results_test/' + str(i) + '.png'
            im.save(im_path)

        #change learning rate considering number of repetitions
        if (i > 3000) and learning_rate_changed_flag == 0:
            learning_rate = learning_rate/10
            learning_rate_changed_flag = 1
        if (i > (num_epochs/2)) and learning_rate_changed_flag == 1:
            learning_rate = learning_rate/100
            learning_rate_changed_flag = 2





if __name__ == "__main__":
    main()