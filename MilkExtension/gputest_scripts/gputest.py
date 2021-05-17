import warnings 
import tensorflow as tf 


print('Tensorflow Version: {}'.format(tf.__version__))

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please ensure you followed the steps correctly')
else:

    print('Default GPU device: {}'.format(tf.test.gpu_device_name()) )