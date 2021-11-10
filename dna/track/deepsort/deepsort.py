from re import L
from dna import det
from cv2 import determinant
import dna
from dna import Size2d
from dna.utils import draw_ds_detections, draw_ds_tracks
import nn_matching
from .tracker import Tracker 
from application_util import preprocessing as prep
from application_util import visualization
from detection import Detection

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from scipy.stats import multivariate_normal

from dna import Box

WHITE = (255,255,255)
YELLOW = (0,255,255)
RED = (0,0,255)
BLUE = (255,0,0)

def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask


class deepsort_rbc():
	def __init__(self, domain: Box, wt_path, params):
		self.domain = domain

		#loading this encoder is slow, should be done only once.
		#self.encoder = generate_detections.create_box_encoder("deep_sort/resources/networks/mars-small128.ckpt-68577")		
		self.encoder = torch.load(wt_path)			
			
		self.encoder = self.encoder.cuda()
		self.encoder = self.encoder.eval()
		print("Deep sort model loaded from path: ", wt_path)

		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", params.metric_threshold , 100)
		self.tracker= Tracker(domain, self.metric, params=params)

		self.gaussian_mask = get_gaussian_mask().cuda()

		self.transforms = torchvision.transforms.Compose([ \
				torchvision.transforms.ToPILImage(),\
				torchvision.transforms.Resize((128,128)),\
				torchvision.transforms.ToTensor()])


	def reset_tracker(self):
		self.tracker= Tracker(self.metric)

	#Deep sort needs the format `top_left_x, top_left_y, width,height
	
	def format_yolo_output( self,out_boxes):
		for b in range(len(out_boxes)):
			out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2]/2
			out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3]/2
		return out_boxes				

	def pre_process(self,frame,detections):
		transforms = torchvision.transforms.Compose([ \
			torchvision.transforms.ToPILImage(),\
			torchvision.transforms.Resize((128,128)),\
			torchvision.transforms.ToTensor()])

		crops = []
		for d in detections:
			for i in range(len(d)):
				if d[i] <0:
					d[i] = 0	

			img_h,img_w,img_ch = frame.shape

			xmin,ymin,w,h = d

			if xmin > img_w:
				xmin = img_w

			if ymin > img_h:
				ymin = img_h

			xmax = xmin + w
			ymax = ymin + h

			ymin = abs(int(ymin))
			ymax = abs(int(ymax))
			xmin = abs(int(xmin))
			xmax = abs(int(xmax))

			try:
				crop = frame[ymin:ymax,xmin:xmax,:]
				crop = transforms(crop)
				crops.append(crop)
			except:
				continue

		crops = torch.stack(crops)

		return crops

	def extract_features_only(self,frame,coords):

		for i in range(len(coords)):
			if coords[i] <0:
				coords[i] = 0	


		img_h,img_w,img_ch = frame.shape
				
		xmin,ymin,w,h = coords

		if xmin > img_w:
			xmin = img_w

		if ymin > img_h:
			ymin = img_h

		xmax = xmin + w
		ymax = ymin + h

		ymin = abs(int(ymin))
		ymax = abs(int(ymax))
		xmin = abs(int(xmin))
		xmax = abs(int(xmax))
		
		crop = frame[ymin:ymax,xmin:xmax,:]
		#crop = crop.astype(np.uint8)

		#print(crop.shape,[xmin,ymin,xmax,ymax],frame.shape)

		crop = self.transforms(crop)
		crop = crop.cuda()

		gaussian_mask = self.gaussian_mask

		input_ = crop * gaussian_mask
		input_ = torch.unsqueeze(input_,0)

		features = self.encoder.forward_once(input_)
		features = features.detach().cpu().numpy()

		corrected_crop = [xmin,ymin,xmax,ymax]

		return features,corrected_crop


	def run_deep_sort(self, frame, bboxes, scores):
		if len(bboxes) > 0:
			features = self.extract_features(frame, bboxes)
			dets = [Detection(bbox, score, feature)	\
					for bbox, score, feature in zip(bboxes, scores, features)]
			outboxes = np.array([d.tlwh for d in dets])
			outscores = np.array([d.confidence for d in dets])
			indices = prep.non_max_suppression(outboxes, 0.8, outscores)
			dets = [dets[i] for i in indices]
		else:
			dets = []

		##################################################################################
		# kwlee
		if dna.DEBUG_SHOW_IMAGE:
			import cv2
			from dna import color

			convas = frame.copy()
			convas = draw_ds_detections(convas, dets, color.GREEN, color.BLACK, line_thickness=1)
			cv2.imshow("dets", convas)
			cv2.waitKey(1)
		##################################################################################

		self.tracker.predict()

		##################################################################################
		# kwlee
		if dna.DEBUG_SHOW_IMAGE:
			import cv2
			from dna import color
			convas = draw_ds_tracks(frame.copy(), self.tracker.tracks, color.RED, color.BLACK, 1,
									dna.DEBUG_TARGET_TRACKS)
			cv2.imshow("predictions", convas)
			cv2.waitKey(1)
		##################################################################################

		deleteds = self.tracker.update(dets)

		return self.tracker, deleteds

	def extract_features(self, frame, bboxes):
		processed_crops = self.pre_process(frame, bboxes).cuda()
		processed_crops = self.gaussian_mask * processed_crops

		features = self.encoder.forward_once(processed_crops)
		features = features.detach().cpu().numpy()
		if len(features.shape)==1:
			features = np.expand_dims(features,0)

		return features