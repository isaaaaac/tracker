# -*- coding: utf-8 -*-
import glob
import random
from PIL import Image,ImageDraw

b_width = 640
b_heigth = 360
# 生成原始的box
ori_label = Image.new('L',(b_width,b_heigth), (0))
ori_label_draw = ImageDraw.Draw(ori_label)

ori_x = 206
ori_y = 135
f_width = 109
f_height = 54

#10级
size_guss = 10
split_w = int(1.0 * (f_width/2)/size_guss)
split_h = int(1.0 * (f_height/2)/size_guss)
split_t = int(1.0 * 255 / size_guss)
for split_i in range(size_guss):
    ori_label_draw.rectangle((ori_x+split_w*split_i ,ori_y+split_h * split_i
                            ,ori_x+f_width-split_w*split_i, ori_y+f_height-split_h*split_i)
                            ,fill=split_t*split_i)
ori_label.show()
ori_label.save('./data_test/2/img1_box.jpg')