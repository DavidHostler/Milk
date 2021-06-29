# Milk
This is my attempt at implementing a GAN to optimize video quality, via both a backend approach and one in browser. The lack of bandwidth during the pandemic due to stay at home policies showed the need for a cheap and easy way to improve streaming data quality. The solution will be a deep learning model trained in python, then mapped to a corresponding architecture  in the browser via Tensorflow.js.

Credit to Manish Dhakal for providing an excellent tutorial of the subject of super-resolution techniques in his medium article, and how to get started with it using Keras and Tensorflow.
Many other tutorials also exist, but are in Pytorch, so it's a lifesaver that he provided this since this project relies on Tensorflow.js being run tentatively in the browser. I've linked his article below for any interested parties:
  
  https://dev.to/manishdhakal/super-resolution-with-gan-and-keras-srgan-38ma
  
  



So far, I've used Pytorch documentation and a set of pretrained weights to build a fullstack Node app to upload a given image and upscale its resolution by a factor of 3 in the python backend. Soon, I'll add video processing functionality to the backend once I have time. 
