import pandas as pd 
import numpy as np
import glob 
import os

dir_obj = '/data/disk1/hungpham/object-detection-generator/trash_4d_to_test/*'
list_dir_img = glob.glob(dir_obj)

for dir_img in list_dir_img:
	with open('info_test_trash.txt', 'a') as f:
		info = 'logo,' + dir_img + ',0'
		f.write(info)
		f.write('\n')
