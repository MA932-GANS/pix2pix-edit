import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


def main():
    
    print(os.getcwd())
    
    src_dir = os.getcwd() + '/R2G_test/images/'
    x = len(glob.glob(os.path.join(src_dir, '*-inputs.png')))
    sets = np.linspace(1301,1300+x,x)
    sets.astype(int)
    
    print(sets)
    
    for i in sets:

        r = cv2.cvtColor(cv2.imread(os.path.join(src_dir, 'R2G_%d-inputs.png'%(i))),cv2.COLOR_BGR2RGB).astype(np.uint8)
        g = cv2.cvtColor(cv2.imread(os.path.join(src_dir, 'R2G_%d-targets.png'%(i))),cv2.COLOR_BGR2RGB).astype(np.uint8)
        out = r + g

        tmp1 = Image.fromarray(out, 'RGB')
        
        tmp1.save('Artificial Cells/real_cell_%d.png'%(i), format='PNG')
        
main()