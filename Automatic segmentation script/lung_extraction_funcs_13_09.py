# -*- coding: utf-8 -*-
"""
author: s.primakov
Spyder Editor
.
"""

import numpy as np
import cv2
from skimage.measure import label,regionprops
from scipy.ndimage import measurements
from skimage.filters import roberts
from numpy import convolve
from skimage import measure
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import os,re
from statsmodels.tsa.api import ExponentialSmoothing
from tqdm import tqdm


#kreating kernel for morphological transformations
def get_kernel():
    kernel=np.zeros((11,11),np.uint8)
    kernel=cv2.circle(kernel,(5,5),4,1,-1)
    for i in [4,6]:
        for j in [1,9]:
            kernel[i,j]=1
            kernel[j,i]=1
    return kernel

def find_bounds_z_direction(image):
    z_dist = np.sum(image,axis = (1,2))     
    zi = np.where(z_dist!=0)
    min_bound = np.min(zi)
    max_bound = np.max(zi)
    center_pos = int((max_bound - min_bound)/2)
    return center_pos,min_bound,max_bound

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def resize_3d_img(img,shape,interp = cv2.INTER_NEAREST):
    init_img = img.copy()
    temp_img = np.zeros((init_img.shape[0],shape[1],shape[2]))
    new_img = np.zeros(shape)
    for i in range(0,init_img.shape[0]):
        temp_img[i,...] = cv2.resize(init_img[i,...],dsize=(shape[2],shape[1]),interpolation = interp)   
    for j in range(0,temp_img.shape[1]):
        new_img[:,j,:] = (cv2.resize(temp_img[:,j,:],dsize=(shape[2],shape[0]),interpolation = interp)) 
    return new_img

def create_flipped_mask(image,i_orig,kernel,method=0):   #image is a lung mask image, i_orig is an image volume
    _,min_bound,max_bound = find_bounds_z_direction(image)
    fl_mask = np.zeros_like(image)
    lc_buff=[]
    alpha = 0.3
    lungs_center_ind ={}
    
    if method:
    ## With adjustment for spine senter irregularity
        st_pt=[]
        lc_buff =[]
        for i in range(min_bound,min_bound+5):
            rib_view = 1*(i_orig[i,...] > 100)
            rib_view[370:,:] = 0
            distr = np.sum(rib_view,axis=0)
            filtered_distr = movingaverage(distr, 7)
            c_limit = int(0.15 * rib_view.shape[0])
            thresh_l1 = np.max(distr)
            distr_indicies = np.where(distr > (thresh_l1*0.15))
            lungs_center = int(rib_view.shape[0]/2-c_limit)+np.where(filtered_distr[int(rib_view.shape[0]/2-c_limit):int(rib_view.shape[0]/2+c_limit)] == np.max(filtered_distr[int(rib_view.shape[0]/2-c_limit):int(rib_view.shape[0]/2+c_limit)]))[0][0]
            st_pt.append(lungs_center)
            
        s0 = np.mean(st_pt)  
        for i in range(min_bound,max_bound+1):
            rib_view = 1*(i_orig[i,...] > 100)
            rib_view[370:,:] = 0
            distr = np.sum(rib_view,axis=0)
            filtered_distr = movingaverage(distr, 7)
            c_limit = int(0.15 * rib_view.shape[0])
            thresh_l1 = np.max(distr)
            distr_indicies = np.where(distr > (thresh_l1*0.15))
            lungs_center = int(rib_view.shape[0]/2-c_limit)+np.where(filtered_distr[int(rib_view.shape[0]/2-c_limit):int(rib_view.shape[0]/2+c_limit)] == np.max(filtered_distr[int(rib_view.shape[0]/2-c_limit):int(rib_view.shape[0]/2+c_limit)]))[0][0]
            st_pt.append(lungs_center)
            print(lungs_center)
            lungs_center_ind[i] = int((alpha*lungs_center)+(1-alpha)*s0)
            s0 = lungs_center
        
    else:
    ## quick method no adjustment
        for i in range(min_bound,max_bound+1):
            rib_view = 1*(i_orig[i,...]>100)
            distr = np.sum(rib_view,axis=0)
            thresh_l1 = np.max(distr)
            distr_indicies = np.where(distr > (thresh_l1*0.15))
            lc_buff.append((np.min(distr_indicies)+np.max(distr_indicies))/2)
        lungs_center = int(np.mean(lc_buff))
        for i in range(min_bound,max_bound+1):
            lungs_center_ind[i] = lungs_center    
        
        
    for i in range(min_bound,max_bound+1):
        ax = image[i,...]
        if lungs_center_ind[i] > int(image.shape[2]/2):
            part1 = ax[:,2*lungs_center_ind[i] -image.shape[2]:lungs_center_ind[i] ]
            part2 = np.hstack((ax[:,lungs_center_ind[i]:],np.zeros((image.shape[1],2*lungs_center_ind[i] -image.shape[2]))))
            part2 = cv2.dilate(np.array(part2,np.uint8),kernel,iterations=1)
            part1 = cv2.dilate(np.array(part1,np.uint8),kernel,iterations=1)
            fl_mask[i,...] = np.hstack((np.flip(part2,axis=1),np.flip(part1,axis=1)))
            un = np.array(fl_mask[i,...],np.uint8)|np.array(ax,np.uint8)
            fl_mask[i,...] = un
        elif lungs_center_ind[i]  < int(image.shape[2]/2):
            part1 = np.hstack((np.zeros((image.shape[1],image.shape[2]-2*lungs_center_ind[i] )),ax[:,:lungs_center_ind[i] ]))
            part2 = ax[:,lungs_center_ind[i] :image.shape[2]-(image.shape[2]-2*lungs_center_ind[i] )]
            part2 = cv2.dilate(np.array(part2,np.uint8),kernel,iterations=1)
            part1 = cv2.dilate(np.array(part1,np.uint8),kernel,iterations=1)
            fl_mask[i,...] = np.hstack((np.flip(part2,axis=1),np.flip(part1,axis=1)))
            un = np.array(fl_mask[i,...],np.uint8)|np.array(ax,np.uint8)
            fl_mask[i,...] = un
        else:
            fl = np.flip(ax,axis=1)
            fl_mask[i,...] = np.array(ax,np.uint8)|np.array(fl,np.uint8)
        
    return fl_mask,min_bound,max_bound
    

def largest_labeled_volumes(im, bg=-1,ratio=0.6):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        try:
            max_val = np.max(counts)
            counts = counts/max_val
            smax = np.where(counts == (counts[((counts>ratio)*(counts<1))]))[0][0]

            return vals[[np.argmax(counts),smax]]
        except:   
            return [vals[np.argmax(counts)]]
    else:
        return None
    
def max_connected_volume_extraction(image):
    #image should be int
    img_buff = image.copy()
    img_buff_lb = image.copy()
    img_buff = 1*(img_buff>0)
    img_buff_lb = np.array(img_buff_lb*10,np.uint8) 
    output_mask = np.zeros_like(img_buff)
    connectivity_array = np.zeros_like(img_buff)
    label_image = label(img_buff_lb)

    for i in range(0,len(img_buff)):
        if i == 0:
            connectivity_array[i,...] = img_buff[i+1,...]&img_buff[i,...]
        elif i == len(img_buff)-1:
            connectivity_array[i,...] = img_buff[i,...]&img_buff[i-1,...]
        else:
            connectivity_array[i,...] = (img_buff[i-1,...]&img_buff[i,...])|(img_buff[i+1,...]&img_buff[i,...]) 

    multip_arr = label_image*connectivity_array
    areas = list(np.unique(multip_arr[multip_arr>0]))
    ##volumetric post processing
    if True:
        if len(areas)>1:
            max_lab_ar = 1
            temp_max = np.sum(1*multip_arr==1)
            for a in areas:
                if np.sum(1*multip_arr==a)>temp_max:
                    temp_max = np.sum(1*multip_arr==a)
                    max_lab_ar =a
            for a in areas:
                #if np.sum(1*multip_arr==a)*params['original_spacing'][2]*0.001>1.5:
                #    pass
                #else:
                if a != max_lab_ar:
                    multip_arr[multip_arr==a]=0
                else:
                    pass
            areas = list(np.unique(multip_arr[multip_arr>0]))

    for val in areas:
        output_mask[label_image==val]=1
    return output_mask

    
def get_seg_lungs(img,return_only_mask= True):
    im = img.copy()
    kernel = get_kernel()
    # Convert into a binary image. 
    binary = (im < -320) 
    
    sq = img.shape[1]*img.shape[2]
    seg_lung =np.zeros_like(img)
    
    
    for nm,ax_slice in enumerate(binary):
    
        # Remove the blobs connected to the border of the image
        cleared = clear_border(ax_slice)
        # Label the image
        label_image = label(cleared)
        
        # Keep the labels with 3 largest areas
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 3:
            for region in regionprops(label_image):
                if region.area < areas[-3]:
                    for coordinates in region.coords:                
                           label_image[coordinates[0], coordinates[1]] = 0
        binary_sl = label_image > 0
        
        
        # Fill in the small holes inside the lungs
        edges = roberts(binary_sl)
        binary_sl = ndi.binary_fill_holes(edges)
        seg_lung[nm,...] = binary_sl
        
    
    for i,axial_slice in enumerate(seg_lung):   #Delete Trahea from mask 
        trahea_coeff = 0.0053*(sq/(512**2))
        trahea_coeff = 0.0069*(sq/(512**2)) #sebast
        labels = measure.label(axial_slice,connectivity=1,background=0)

        vals, counts = np.unique(labels, return_counts=True)
        counts = counts[vals != 0]
        vals = vals[vals != 0]
        if (counts/(sq*1) < trahea_coeff).any():
            for l in vals[counts/(sq*1) < trahea_coeff]:
                labels[labels == l] = 0
            labels = labels != 0
            seg_lung[i,...] = labels  
                
    # Remove table and other air pockets insided body
    labels = measure.label(seg_lung, background=0)

    for i in np.unique(labels):
        temp_center = measurements.center_of_mass(labels==i)[1]
        if temp_center> seg_lung.shape[1]*0.75:
            seg_lung[labels==i]=0

    labels = measure.label(seg_lung, background=0)        
    l_max = largest_labeled_volumes(labels, bg=0)
    if l_max is not None: # There are air pockets
        if len(l_max)>1:
            labels[labels == l_max[1]] = l_max[0]
            seg_lung[labels != l_max[0]] = 0

        else:
             seg_lung[labels != l_max[0]] = 0
            
    
    # Searching for lungs center and flip the lungs
    seg_lung,min_bound,max_bound = create_flipped_mask(seg_lung,im,kernel)
           

    for i,axial_slice in enumerate(seg_lung):
        bi=np.array((axial_slice),np.uint8)
        opening = cv2.morphologyEx((1-bi),cv2.MORPH_OPEN,kernel,iterations=7)
        dilat = cv2.dilate((1-opening),kernel,iterations=2)
        seg_lung[i,...] = cv2.erode(dilat,kernel,iterations=1)
        
    #Stretch the lung mask in z direction +-2 slices
    if (min_bound>2)and (max_bound<seg_lung.shape[0]-2):
        try:

            for i in range(1,3):
                seg_lung[min_bound -i,...] = seg_lung[min_bound,...]
                seg_lung[max_bound +i,...] = seg_lung[max_bound,...]
            min_bound-=2
            max_bound+=2
        except:
            print('some problrm with patient ')
    if return_only_mask:
        return seg_lung
    else:
        return seg_lung,min_bound,max_bound
   
    
def apply_mask(image,mask,threshold=-1000): #apply mask to image and save HU values
    new_img = image.copy()
    new_img[np.where(mask==0)] = threshold
    
    return new_img

def parse_dataset(general_path,img_only = True):
    patients = next(os.walk(general_path))[1]
    Patient_dict={}
    temp_img=''
    temp_mask=''
    i=0
    
    for patient in tqdm(patients):
        for root, dirs, files in os.walk(os.path.join(general_path,patient)):
            for file in files:
                if not img_only:
                    if re.search('mask',file.lower()):
                        temp_mask = file
                        for imgfile in files:
                            if re.search('volume',imgfile.lower()) or re.search('image',imgfile.lower()):
                                temp_img = imgfile
                                Patient_dict[i]=[os.path.join(general_path,patient,temp_img),os.path.join(general_path,patient,temp_mask)]
                                i+=1
                                temp_mask=''
                                temp_img=''
                else:
                    
                    if re.search('volume',file.lower()) or re.search('image',file.lower()):
                        temp_img = file
                        Patient_dict[i]=[os.path.join(general_path,patient,temp_img)]
                        i+=1
                        temp_img=''
                
    print(len(Patient_dict),r' Patients found')
    return Patient_dict
                