"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Detect forks from images with trained YOLOv7 ONNX model
"""

import cv2 as cv
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from utils.dataset import TestDataset
from utils.utils import plot_one_box, show_predicted_image, adjust_image
from os.path import exists, isdir
from os import makedirs
from random import randint
import argparse

def predict_bbox(session, images):
	"""
	Predict bounding boxes from images
	"""
	outname = [i.name for i in session.get_outputs()]

	dict_output = {}
	for i, samples in enumerate(images):
		im, ratio, dwdh, name = samples['image'], samples['ratio'], samples['dwdh'], samples['name']
		im = np.ascontiguousarray(im/255)
		out = session.run(outname, {'images':im})[0]
		dict_output[f"batch {i}"] = {"preds": out, "ratio": ratio, "dwdh": dwdh, "name": name}

	return dict_output

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Predict with YOLOv7-fork ONNX model')
    
	parser.add_argument('--model', required=True, metavar='/path/to/model.onnx', help="Path to ONNX model")
	parser.add_argument('--input', required=True, help="Path to images or path to image")
	parser.add_argument('--batch', default=1, help="Batch size")
	parser.add_argument('--save', default=False, action='store_true', help="Save predicted image")
	parser.add_argument('--dontshow', default=False, action='store_true', help="Don't show predicted image")
	parser.add_argument('--cuda', default=False, action='store_true', help="Set execution on GPU")

	args = parser.parse_args()
	for key, value in args._get_kwargs():
		if value is not None:
			print(f'{key.capitalize()}: {value}')
	print()

	# Check if the input is a dir
	input_isdir = isdir(args.input)

	# Load the model
	print('Loading model...', flush=True)
	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.cuda else ['CPUExecutionProvider']
	session = ort.InferenceSession(args.model, providers=providers)

	# Get output name and input shape
	outname = [i.name for i in session.get_outputs()]
	input_shape = session.get_inputs()[0].shape
	h, w = input_shape[2], input_shape[3]

	# Load the images
	print('Loading images...', flush=True)
	if input_isdir:
		dataset = TestDataset(args.input, shape=(h, w))
		images = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
	else:
		images = [adjust_image(args.input, shape=(h, w))]

	# Predict from images
	print('Making predictions...', flush=True)
	dict_output = predict_bbox(session, images)
	
	names = ['fork']
	colors = {name: [randint(0, 255) for _ in range(3)] for name in names}
	# colors = {name: [104, 184, 82] for name in names} # green

	if args.save and not exists(f'data/responses'):
		makedirs(f'data/responses')

	# For each image, plot the results
	print('Plotting results...', flush=True)
	for i, key in enumerate(dict_output.keys()):
		pred, ratio, dwdh, name = dict_output[key]['preds'], dict_output[key]['ratio'][0], dict_output[key]['dwdh'], dict_output[key]['name'][0]
		ratio = float(ratio)
		dwdh = float(dwdh[0]), float(dwdh[1])

		# Load original image
		if input_isdir:
			image = dataset.__getsrc__(i)
		else:
			image = cv.imread(args.input)

		# Adjust bounding box to original image
		for prediction in pred:
			batch_id, x0, y0, x1, y1, cls_id, score = prediction
			box = np.array([x0,y0,x1,y1])
			box -= np.array(dwdh*2)
			box /= ratio
			box = box.round().astype(np.int32).tolist()
			cls_id = int(cls_id)
			score = round(float(score),3)
			label = names[cls_id]
			color = colors[label]
			label += ' ' + str(score)
			plot_one_box(box, image, label=label, color=color, line_thickness=1)
	
		if args.save:
			path = 'data/responses/' + name
			cv.imwrite(path, image)
		
		if not args.dontshow:
			show_predicted_image(image)
		