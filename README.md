# Milk




So far, I've used Pytorch documentation and a set of pretrained weights to build a fullstack Node app to upload a given image and upscale its resolution by a factor of 3 in the python backend, using a NodeJS/ExpressJS backend to post files to the server, which are then upscaled via Pytorch using the SRGAN architecure on the Pytorch website, including pretrained weights taken from the Pytorch website.


To set up the project, make sure that you're in the root folder already.
run the following commands to set up a virtual environment called "Torch" in the Pytorch_Backend folder:

### cd Pytorch_Backend
### python3 -m venv Torch


You should now be back in the Pytorch_Backend folder, with a virtual environment folder created in the Pytorch_Backend directory.

Activate the virtual environment now by running

### source Pytorch_Backend/Torch/bin/activate

Now, your terminal should say (Torch) somewhere beside the name of the path of the directory that you're currently inside.

Install dependencies by running

### pip install -r requirements.txt

now once this lengthy install is completed, return to the root directory by

### cd ..


Using a split terminal, run

### npm start

to activate the Express server on localhost 3000.


and in the other terminal run  

### python Pytorch_Backend/main.py


simultaneously.

The reason for doing this in two terminal windows as opposed to using a single bash script to run both programs concurrently, is that the user should 
be able to clearly read the output of each program separately in order to appreciate what's really going on.
One terminal will provide the message "Listening on port 3000..." (that's our NodeJS terminal running the Express app on port 3000)
and the other will simply show the outputs of the Python script main.py, responsible for performing the actual image upscaling.


One way you could think of this is that the Node terminal is running the webpage inside your browser, and the terminal running the Python script
is a process that would normally  take place on a server in the cloud thousands of kilometres away from you.

The reason for using ExpressJS and not Django which already uses Python, is that I wished to make this app a quick demonstration of image upscaling ather than a commercial application at this point, and ExpressJS requires very, very little code!
Additionally, it's interesting to see a machine use two programs written in separate languages (i.e. Javascript and Python) work together to accomplish a task on the same device. Setting up the entire project locally provides the user with a closeup view of the code, rather than hosting it online and letting them only see the front end code, which itself is very basic.

At the end of it all, we have a device that takes in a generic image, and uses deep learning techniques to improve the resolution of the image.
Of course, if you upload an image of higher resolution that it's meant to put out, this will not be all that useful to you. Low-quality images 
are a perfect fit for this app. Try it out and see if you find this concept useful.



Happy hacking!
