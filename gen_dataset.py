import cv2
import random
import os

img_w = 256  
img_h = 256  

image_sets = ['1.png','2.png','3.png','4.png','5.png']

def creat_dataset(image_num = 15000):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in range(len(image_sets)):
        count = 0
        src_img = cv2.imread('./data/src/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('./data/label/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            cv2.imwrite(('./train/src/%d.png' % g_count),src_roi)
            cv2.imwrite(('./train/label/%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1
            
    

if __name__=='__main__':  
    creat_dataset()
