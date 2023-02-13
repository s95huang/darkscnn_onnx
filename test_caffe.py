#!/usr/bin/env python2

import numpy as np
import caffe
import random
import cv2
# import GPUtil

DEBUG = False

alpha = 0.75
# load model

caffe.set_mode_gpu()
model_def = 'deploy.prototxt'
model_weights = 'deploy.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# define transformer
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', np.array([95, 99, 96]))            # subtract the dataset-mean value in each channel

# load image
# image = caffe.io.load_image('test.jpg')
image = caffe.io.load_image('almon1.png')

# resize input image


crop_image = image[:, 312:, :]
resize_image = caffe.io.resize_image(image, [480, 640])
cv_resize_image = cv2.resize(image, (640, 480))
#print(resize_image.shape)
transformed_image = transformer.preprocess('data', resize_image)

# lane predict
net.blobs['data'].data[...] = transformed_image
output = net.forward()
mask_color = np.zeros((480, 640, 3), np.uint8)
# make mask_color all white
# mask_color[:, :, :] = (255, 255, 255)
confidence_thread = 0.95


def random_rgb(num):
   colors = []
   for i in range(num):
       if i == 0:
            colors.append((0, 0, 0))
          #   colors.append((255, 255, 255))

       elif i in range(1, 5) or i == 11:
            colors.append((255, 0, 0))
       elif i in range(6, 10) or i == 12:
            colors.append((0, 255, 0))
       elif i == 5:
            colors.append((0, 0, 255))
       elif i == 10:
            colors.append((255, 255, 0))
   return colors

lane_colors = random_rgb(13)
if DEBUG:
    print(lane_colors)
    print(output['softmax'][0])
    print("lane output shape: ", output['softmax'].shape)

for id, lane in enumerate(output['softmax'][0]):

    index = (lane >= confidence_thread)
    # for row in range(lane.shape[0]):
    #     for col in range(lane.shape[1]):
    #         if lane[row][col] > confidence_thread:
    #             mask_color[row][col] = lane_colors[id]
    mask_color[lane >= confidence_thread] = lane_colors[id]
#     print(lane[0].shape)

# dst = cv2.addWeighted(resize_image, alpha, mask_color, 1-alpha, 0.0)
dst = cv2.addWeighted(cv_resize_image, alpha, mask_color, 1.0-alpha, 0.0, dtype = cv2.CV_32F)


GPUtil.showUtilization()

# use cv2 to show the image
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
