import numpy as np
import torch
import umal


if __name__ == '__main__':
    
    tmp = umal.umal()
    tmp.init_training(architecture=umal.internal_network)
    
    tmp.load_weights('weights/2019-10-30-09:44:57_-lr_0.01.pth')

    tmp.predict(nx=500, ny=200, ntaus=90)