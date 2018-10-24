# -*- coding: utf-8 -*-
import glob
import random
from PIL import Image,ImageDraw
#一些宏定义


list_background = glob.glob('./background/*.jpg')
size_background = len(list_background)

list_front = glob.glob('./front/*.jpg')
size_front = len(list_front)

def gen_label():
    for index in range(1000):
        print(index)
        #从背景和前景中随机抽取一张图片
        background = list_background[random.randint(0, size_background-1)]
        front = list_front[random.randint(0, size_front-1)]
        
        #打开背景图片
        b_img = Image.open(background)
        f_img = Image.open(front)

        
        # 获取图片的size
        b_width = b_img.size[0]
        b_height = b_img.size[1]

        f_width = f_img.size[0] + 10
        f_height = f_img.size[1] + 10


        
        # 生成原始的xy
        ori_x = random.randint(f_width, b_width-2 * f_width)
        ori_y = random.randint(f_height, b_height- 2 * f_height)
        b_img.paste(f_img, (ori_x,ori_y))
        ori_img = b_img
        
        # 生成原始的box
        ori_label = Image.new('L',(b_width,b_height), (0))
        ori_label_draw = ImageDraw.Draw(ori_label)
        
        #10级
        size_guss = 10
        split_w = int(1.0 * (f_width/2)/size_guss)
        split_h = int(1.0 * (f_height/2)/size_guss)
        split_t = int(1.0 * 255 / size_guss)
        for split_i in range(size_guss):
            ori_label_draw.rectangle((ori_x+split_w*split_i ,ori_y+split_h * split_i
                                    ,ori_x+f_width-split_w*split_i, ori_y+f_height-split_h*split_i)
                                    ,fill=split_t*split_i)

        
                
          
        
        # 生成要移动的xy
        dst_img = Image.open(background)
        # 对背景图片crop 偏移一下
        b_img_x1 = random.randint(0,50)
        b_img_y1 = random.randint(0,50)
        b_img_x2 = random.randint(0,50)
        b_img_y2 = random.randint(0,50)

        dst_img = dst_img.crop([0+b_img_x1,0+b_img_y1,b_width-b_img_x2,b_height-b_img_y2])
        dst_img = dst_img.resize([b_width,b_height]) 
        
        # 偏移尺寸
        d_size = f_height
        dst_x = ori_x + random.randint(-d_size, d_size)
        dst_y = ori_y + random.randint(-d_size, d_size)
        
        # 对前景图片crop 偏移一下
        f_img_x1 = random.randint(0,f_width/5)
        f_img_y1 = random.randint(0,f_width/5)
        f_img_x2 = random.randint(0,f_height/5)
        f_img_y2 = random.randint(0,f_height/5)
        
        f_img = f_img.crop([0+f_img_x1,0+f_img_y1,f_img.size[0]-f_img_x2,f_img.size[1]-f_img_y2])
        
        f_img = f_img.resize([f_width,f_height])
        
        dst_img.paste(f_img, (dst_x,dst_y))
        
        
        # 生成目标的box
        dst_label = Image.new('L',(b_width,b_height), (0))
        dst_label_draw = ImageDraw.Draw(dst_label)
        for split_i in range(size_guss):
            dst_label_draw.rectangle((dst_x+split_w*split_i ,dst_y+split_h * split_i
                                    ,dst_x+f_width-split_w*split_i, dst_y+f_height-split_h*split_i)
                                    ,fill=split_t*split_i)
 

        ori_img.save('./data/img1/%d.jpg'%(index))
        ori_label.save('./data/img1_box/%d.jpg'%(index))
        
        dst_img.save('./data/img2/%d.jpg'%(index))
        dst_label.save('./data/img2_box/%d.jpg'%(index))
gen_label()