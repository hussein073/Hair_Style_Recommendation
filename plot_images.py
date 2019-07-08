
## FUNCTIONS TO OVERLAYS ALL PICS!!
get_ipython().magic('matplotlib inline')
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
import glob, os
import operator
from PIL import Image
import pathlib
from pathlib import Path


image_dir = ["data/pics_for_overlaps/Sarah",
             "data/pics_for_overlaps/Allison",
             "data/pics_for_overlaps/Amanda_S",
             "data/pics_for_overlaps/Gisele",
             "data/pics_for_overlaps/Keira",
             "data/pics_for_overlaps/Squares"
            ]

plt.figure(figsize=(20,10))
from PIL import Image, ImageDraw,ImageFont
font = ImageFont.truetype("fonts/Arial.ttf", 20)
n_row = 2
n_col = 3
g = 0
text = ["Sarah-round","Allison-oval","Amanda-heart",'Gisele-long','Keira-square','All Squares']
for ddir in image_dir:
    a =  .6
    i = 0
    g += 1
    for f in os.listdir(ddir):
        if f.endswith('.jpg'):
            file, ext = os.path.splitext(f)

            im = Image.open(ddir+'/'+f)
            image = cv2.imread(ddir+'/'+f)
            a = a-.01
            i += 1
            draw = ImageDraw.Draw(im)
            draw.text((10,10) ,text[g-1], fill=None, font=font, anchor=None)
            draw.text((10,30) ,str(i)+" Pics", fill=None, font=font, anchor=None)
            plt.subplot(n_row, n_col, g )
            plt.imshow(im, alpha = a)
