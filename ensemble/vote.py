import numpy as np
import cv2
import argparse

RESULT_PREFIXX = ['./result1/','./result2/','./result3/']

# each mask has 5 classes: 0~4

def vote_per_image(image_id):
    result_list = []
    for j in range(len(RESULT_PREFIXX)):
        im = cv2.imread(RESULT_PREFIXX[j]+str(image_id)+'.png',0)
        result_list.append(im)
        
    # each pixel
    height,width = result_list[0].shape
    vote_mask = np.zeros((height,width))
    for h in range(height):
        for w in range(width):
            record = np.zeros((1,5))
            for n in range(len(result_list)):
                mask = result_list[n]
                pixel = mask[h,w]
                #print('pix:',pixel)
                record[0,pixel]+=1
           
            label = record.argmax()
            #print(label)
            vote_mask[h,w] = label
    
    cv2.imwrite('vote_mask'+str(image_id)+'.png',vote_mask)
        

vote_per_image(3)