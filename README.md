# Milk




So far, I've used Pytorch documentation and a set of pretrained weights to build a fullstack Node app to upload a given image and upscale its resolution by a factor of 3 in the python backend, using a NodeJS/ExpressJS backend to post files to the server, which are then upscaled via Pytorch using the SRGAN architecure on the Pytorch website, including pretrained weights. 

Using a split terminal, run

node index.js

to activate the Express server on localhost 3000, and in the other terminal run  

python Pytorch_Backend/main.py


simultaneously.
The reason for using ExpressJS and not Django which already uses Python, is that I wished to make this app a quick demonstration of image upscaling and not a commercial application at this point, and ExpressJS requires very, very little code!
Additionally, it's interesting to see a machine use two programs written in separate languages (i.e. Javascript and Python) work together to accomplish a task on the same device. I may be uniquely fascinated in this regard...
