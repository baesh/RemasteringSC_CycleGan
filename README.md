# RemasteringSC_CycleGan
Reamastering Starcraft with CycleGan


#Task

Given starcraft game video before remastered, convert the video into remastered version.


#Idea

-Use Cycle Gan for training
-batch size: 25
-trained/generated image size: 128*128
-epoch: 100000
<br>
#Layers

![스크린샷 2023-11-07 162113](https://github.com/baesh/RemasteringSC_CycleGan/assets/18441461/88fa245c-64c7-43d3-a7f5-60b0ba05e2ac)

Generator layer

![스크린샷 2023-11-07 162053](https://github.com/baesh/RemasteringSC_CycleGan/assets/18441461/cae04440-136b-4974-80ba-a62dc6d4c090)

Discriminator layer

![스크린샷 2023-11-07 162327](https://github.com/baesh/RemasteringSC_CycleGan/assets/18441461/945fc9f2-48a0-4877-a7a2-d0770c12e864)

Layer connection

<br>
#Codes

-Change video to images: video_to_im.py
-Training: train.py
-Remastering image using trained data: remaster_image.py
-Change images to video: create_vid.py
-etc: layer.py, function_lists.py ('layer.py' contains generator and discriminator layers, 'function_lists.py' contains functions used in codes)

<br>
#Result

View created.avi




