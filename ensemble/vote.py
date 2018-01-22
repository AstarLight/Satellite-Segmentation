import numpy as np
import skimage
import skimage.io as io
import pandas as pd
from tqdm import tqdm
from utils.preprocess import encode_segmap,segmap
import sys,os
import argparse
def vote_per_image(args):
	imgs=[]
	for imgpath in args.inputs:
		if(not os.path.exists(imgpath)):
			print "Path not exists: ",imgpath
		img = io.imread(imgpath)
		if(int(args.vote_vis)):
			img = encode_segmap(img)
		imgs.append(img)
	vote_map = np.zeros_like(imgs[0])
	num_files = len(args.inputs)
	for i in tqdm(range(vote_map.shape[0])):
		for j in range(vote_map.shape[1]):
			pre_list=[]
			for index in range(num_files):
				#extract bin map
				if(args.bincls):
					if(imgs[index][i][j]==args.cls_index):
						pre_list.append(imgs[index][i][j])
					else:
						pre_list.append(5) # unknown class
				#regular vote
				else:
					if(imgs[index][i][j]>0 and imgs[index][i][j]<=4):
						pre_list.append(imgs[index][i][j])
			most_label = np.argmax(np.bincount(pre_list))
			vote_map[i][j] = most_label
			pre_list = np.array(pre_list)

			if(args.road_first and np.count_nonzero(pre_list==args.cls_index)>0):
				vote_map[i][j] = args.cls_index

   	color_mask = segmap(vote_map)
   	return vote_map,color_mask


if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Params')
	parser.add_argument('--inputs', nargs='*', type=str)
	parser.add_argument('--output', nargs='?', type=str)
	#parser.add_argument('--road_first',nargs='?',type=int,default=False)
	parser.add_argument('--vote_vis',nargs='?',type=int,default=False)
	parser.add_argument('--bincls',nargs='?',type=int,default=0)
	parser.add_argument('--road_first',nargs='?',type=int,default=0)
	parser.add_argument('--cls_index',nargs='?',type=int,default=0)
	args = parser.parse_args()
	print args.inputs

	output_name=args.output
	vote_map,color_mask = vote_per_image(args)

	#save vote label and color map
	io.imsave(output_name,vote_map)
	io.imsave(os.path.join(os.path.dirname(output_name),"vis_"+os.path.basename(args.inputs[0])),color_mask)