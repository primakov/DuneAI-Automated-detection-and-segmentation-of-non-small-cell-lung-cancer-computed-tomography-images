# -*- coding: utf-8 -*-
"""
Spyder Editor
s.primakov@maastrichtuniversity.nl
This is a temporary script file.
"""


import numpy as np
import SimpleITK as sitk
import cv2
import os
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
import matplotlib.patches as patches
#Keras & TF
import keras
import keras.backend as K
from keras.models import load_model
import math, random


# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
# 
# Initially: No boundary points known
def make_circle(points):
	# Convert to float and randomize order
	shuffled = [(float(x), float(y)) for (x, y) in points]
	random.shuffle(shuffled)
	
	# Progressively add points to circle or recompute circle
	c = None
	for (i, p) in enumerate(shuffled):
		if c is None or not is_in_circle(c, p):
			c = _make_circle_one_point(shuffled[ : i + 1], p)
	return c


# One boundary point known
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left  = None
	right = None
	px, py = p
	qx, qy = q
	
	# For each point not in the two-point circle
	for r in points:
		if is_in_circle(circ, r):
			continue
		
		# Form a circumcircle and classify it on left or right side
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
			right = c
	
	# Select which circle to return
	if left is None and right is None:
		return circ
	elif left is None:
		return right
	elif right is None:
		return left
	else:
		return left if (left[2] <= right[2]) else right


def make_diameter(a, b):
	cx = (a[0] + b[0]) / 2.0
	cy = (a[1] + b[1]) / 2.0
	r0 = math.hypot(cx - a[0], cy - a[1])
	r1 = math.hypot(cx - b[0], cy - b[1])
	return (cx, cy, max(r0, r1))


def make_circumcircle(a, b, c):
	# Mathematical algorithm from Wikipedia: Circumscribed circle
	ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2.0
	oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2.0
	ax = a[0] - ox;  ay = a[1] - oy
	bx = b[0] - ox;  by = b[1] - oy
	cx = c[0] - ox;  cy = c[1] - oy
	d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
	if d == 0.0:
		return None
	x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
	y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
	ra = math.hypot(x - a[0], y - a[1])
	rb = math.hypot(x - b[0], y - b[1])
	rc = math.hypot(x - c[0], y - c[1])
	return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
	return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
	return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def calculate_values(mask_array_predicted,params,mask_array_orig=None):
    recist,idx = 0,0
    for i,temp_slice in enumerate(mask_array_predicted):
        if np.sum(temp_slice.flatten())>10:
            x_ind,y_ind = np.where(temp_slice==1)
            circ = make_circle(zip(x_ind,y_ind))
            slice_temp_diameter = int(2*circ[2]*params[0])
            if slice_temp_diameter>recist:
                recist = slice_temp_diameter
                idx =i
                circle = circ
    volume_predicted = np.round(np.sum(mask_array_predicted.flatten())*params[0]*params[1]*params[2]*0.001,2)
    
    if mask_array_orig:
        recist_orig,idx = 0,0
        mask_array_orig = np.squeeze(mask_array_orig)
        for i,temp_slice in enumerate(mask_array_orig):
            if np.sum(temp_slice.flatten())>10:
                x_ind,y_ind = np.where(temp_slice==1)
                circ = make_circle(zip(x_ind,y_ind))
                slice_temp_diameter = int(2*circ[2]*params[0])
                if slice_temp_diameter>recist_orig:
                    recist_orig = slice_temp_diameter
                    idx=i
                    circle = circ
                    
        volume_orig = np.round(np.sum(mask_array_orig.flatten())*params[0]*params[1]*params[2]*0.001,2)
        return recist,recist_orig,volume_predicted,volume_orig,idx
    else:
        return recist,volume_predicted,idx,circ
        
