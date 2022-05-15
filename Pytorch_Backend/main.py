import io
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init 
from torch.autograd import Variable
import os
from os import listdir
from os.path import isfile, join

import numpy as np

import time 
print("Hola Mundo!")

ROOT = os.getcwd()
# videopath = "/home/david/Downloads/srgan/backend/files/video/"
imagepath = "Downloads/PortfolioProjects/Milk/uploads/"

def get_video_frames(path,images_path, frame_limit):
    cap= cv2.VideoCapture(path)
    i=0
    count = 0
    while(cap.isOpened()) and i <= frame_limit:
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(images_path + "/img" + str(i) + '.jpg',frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()




#While qeue is empty wait for customer to upload image

while len(os.listdir(videopath)) == 0:
        time.sleep(1)
        print("Wait...")


#Get location of video on server side once uploaded by user
for file in os.listdir(videopath):
    
    video_file_name = videopath + file



#Convert video upload to frames
get_video_frames(video_file_name,  imagepath,  1250)



#if video_file_name.endswith(".mp4"):
#    get_video_frames(video_file_name, "/media/david/Backup Plus/srgan/backend_frames")

#from Pytorch_Backend.Super_Resolution import SuperResolutionNet
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
torch_model = SuperResolutionNet(upscale_factor=3)


onlyfiles = [ f for f in listdir(imagepath) if isfile(join(imagepath,f)) ]

model_url = '/home/david/.cache/torch/hub/checkpoints/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()

x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)


# Export the model

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
import onnx
onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

img = Image.open(imagepath + onlyfiles[0])

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
 

final_img = final_img.save("/home/dolan/Downloads/PortfolioProjects/Mil/upscaled_imgs/final_image.jpg")

#cv2.imwrite("upscaled_imgs/new.jpg", final_img )