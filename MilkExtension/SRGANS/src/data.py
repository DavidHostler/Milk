import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self):
       self.X = []
       self.Y = []

    def get_training_data(self):
        

        for x in os.listdir(db_dir + "FILE_PATH/train/low_res_images"):
            img_x = cv2.imread(db_dir + "FILE_PATH/train/low_res_images/" + x)
            self.X.append(img_x)
    
        self.X = np.array(self.X) / 255
            
        for y in os.listdir(db_dir + "FILE_PATH/train/high_res_images"):
            img_y = cv2.imread(db_dir + "FILE_PATH/train/high_res_images/" + y)
            self.Y.append(img_y)
        self.Y = np.array(self.Y) / 255 
        
        return self.X,self.Y
    

    def get_testing_data(self):
        

        for x in os.listdir(db_dir + "FILE_PATH/test/low_res_images"):
            img_x = cv2.imread(db_dir + "FILE_PATH/test/low_res_images/" + x)
            self.X.append(img_x)
    
        self.X = np.array(self.X) / 255
            
        for y in os.listdir(db_dir + "FILE_PATH/test/high_res_images"):
            img_y = cv2.imread(db_dir + "FILE_PATH/test/high_res_images/" + y)
            self.Y.append(img_y)
        self.Y = np.array(self.Y) / 255 
        
        return self.X,self.Y