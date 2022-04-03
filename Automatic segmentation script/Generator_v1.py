# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:48:10 2018

@author: s.primakov
"""
import sys
import lung_extraction_funcs_13_09 as le
import os,re,random,time,sys,scipy
import SimpleITK as sitk
import keras
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

                      
class Patient_data_generator(keras.utils.Sequence):
    
    def __init__(self,patient_dict, batch_size=8, image_size=512,tumor_size_low_lim = 0,sl_th = None, zero_sl_ratio = 1, model_params={}, 
                 shuffle=False, augment=False, predict=False,img_only = False, normalize = False, norm_coeff=[], 
                 use_window = False, window_params=[],hu_intensity = False, resample_int_val = False, resampling_step =25,
                 equalize_hist = False, verbosity = False, size_eval = False, calculate_snr =False,reshape=False, extract_lungs=False):
        
        self.patient_dict = patient_dict
        self.filenames = list(patient_dict.keys())
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.hu_intensity = hu_intensity
        self.predict = predict
        self.normalize = normalize
        self.norm_coeff =norm_coeff
        self.equalize_hist=equalize_hist
        self.calculate_snr = calculate_snr
        self.use_window = use_window
        self.window_params = window_params
        self._tum_size = tumor_size_low_lim
        self._slice_thickness = sl_th
        self.zero_sl_ratio = zero_sl_ratio
        self.resample_int_val = resample_int_val
        self.step = resampling_step
        self.img_only = img_only
        self.tum_size_distr=[]
        self.tum_volume_distr={}
        self.zero_dict={}
        self.tum_dict={}
        self.reshape = reshape
        self.verbosity=verbosity
        self.extract_lungs = extract_lungs
        self.model_params=model_params
        self.on_epoch_end()
        self.__epoch_zero_dict = None
        self.__epoch_tum_dict = None
        self._normalization_init()
        self._patients_init(size_eval)
        
    def _resize_3d_img(self,img,shape,interp = cv2.INTER_CUBIC):
        init_img = img.copy()
        temp_img = np.zeros((init_img.shape[0],shape[1],shape[2]))
        new_img = np.zeros(shape)
        for i in range(0,init_img.shape[0]):
            temp_img[i,...] = cv2.resize(init_img[i,...],dsize=(shape[2],shape[1]),interpolation = interp)   
        for j in range(0,shape[1]):
            new_img[:,j,:] = (cv2.resize(temp_img[:,j,:],dsize=(shape[2],shape[0]),interpolation = interp)) 
        return new_img     
        
    def _load_patient(self, filename):
        # load patient files as numpy arrays 
        
        if self.verbosity:
            print('-'*7,'Loading patient ',filename,'-'*7)
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.patient_dict[filename][0]))
        msk = sitk.GetArrayFromImage(sitk.ReadImage(self.patient_dict[filename][1]))
        if self.verbosity:
            print('-'*7,'Img spacing ',sitk.ReadImage(self.patient_dict[filename][0]).GetSpacing(),'-'*7)
            print('Image shape:',img.shape)
        
        #print(img.shape,msk.shape)
        #Set minimal values for HU intensities
        if self.hu_intensity:#Set lower limit to -1000 HU
            img[img <-1000] = -1000
            if self.verbosity:
                print('Adjust intensity') 
                
        if self.extract_lungs:
            temp_img = img.copy()
            temp_args = le.get_seg_lungs(img=temp_img,return_only_mask=False)
            if temp_args is not None:
                lung_mask,minb,maxb=temp_args
                lung_mask_with_tumor_mask = lung_mask.copy()
                kernel = le.get_kernel()
                
                
                ##stretch lung mask if the tumour is outside it, only for training
                kernel_sm =cv2.circle(kernel,(3,3),4,1,-1)
                lung_mask_indicies = np.where(np.sum(lung_mask,axis=(1,2))!=0)
                min_ind_l_mask,max_ind_l_mask = np.min(lung_mask_indicies),np.max(lung_mask_indicies)
                try:
                    tum_mask_indicies = np.where(np.sum(msk,axis=(1,2))!=0)
                    min_ind_tum_mask,max_ind_tum_mask = np.min(tum_mask_indicies),np.max(tum_mask_indicies)
                except:
                    min_ind_tum_mask,max_ind_tum_mask = min_ind_l_mask,max_ind_l_mask
                    
                    
                
                if min_ind_tum_mask < min_ind_l_mask:
                    temp_slice = lung_mask[min_ind_l_mask,...]
                    for i in range(min_ind_tum_mask,min_ind_l_mask):
                        tmp = temp_slice*(1*(img[i,...]<-400))
                        tmp = cv2.morphologyEx(np.array(tmp,np.uint8),cv2.MORPH_CLOSE,kernel_sm,iterations=4)
                        tmp = cv2.morphologyEx(np.array(tmp,np.uint8),cv2.MORPH_OPEN,kernel_sm,iterations=2)
                        lung_mask[i,...] = tmp
                    if self.verbosity:
                        print('Mask stretched down')
                    min_ind_l_mask = min_ind_tum_mask
                    
                if max_ind_tum_mask>max_ind_l_mask:
                    temp_slice = lung_mask[max_ind_l_mask,...]
                    for i in range(max_ind_l_mask,max_ind_tum_mask):
                        tmp = temp_slice*(1*(img[i,...]<-400))
                        tmp = cv2.morphologyEx(np.array(tmp,np.uint8),cv2.MORPH_CLOSE,kernel_sm,iterations=4)
                        tmp = cv2.morphologyEx(np.array(tmp,np.uint8),cv2.MORPH_OPEN,kernel_sm,iterations=2)
                        lung_mask[i,...] = tmp
                    if self.verbosity:
                        print('Mask stretched up')
                    max_ind_l_mask = max_ind_tum_mask
                    
                
                for k,layer in enumerate(msk):
                    temp_layer = layer.copy()
                    lung_mask_with_tumor_mask[k,...]= np.array(lung_mask_with_tumor_mask[k,...],np.uint8)|cv2.dilate(np.array(temp_layer,np.uint8),kernel,iterations=3)
        
                new_img = le.apply_mask(img,lung_mask_with_tumor_mask)
                img = new_img
                temp_list = set(self.__epoch_zero_dict[filename])
                for item in temp_list:
                    if (item<= min_ind_l_mask) or (item >= max_ind_l_mask):
                        self.__epoch_zero_dict[filename].remove(item)
                
                
                if self.verbosity:
                    print('Applying mask done ')
            else:
                print('Lungs are not extracted for patient',filename[0][:-11].split('\\')[-1])
                pass    
            
        ####  Reshaping the image&mask
        if True:
            img_for_reshape = img.copy()
            msk_for_reshape = msk.copy()
            temp_data_orig = sitk.ReadImage(self.patient_dict[filename][0])

            new_img = np.array(self._resize_3d_img(img_for_reshape,(int(img.shape[0]),
                                                    int(temp_data_orig.GetSpacing()[0]*img.shape[1]),
                                                    int(temp_data_orig.GetSpacing()[1]*img.shape[2])),cv2.INTER_NEAREST),np.int16)
    
            
            if not self.img_only:
                new_mask = np.array(self._resize_3d_img(msk_for_reshape,(int(img.shape[0]),
                                                          int(temp_data_orig.GetSpacing()[0]*img.shape[1]),
                                                          int(temp_data_orig.GetSpacing()[1]*img.shape[2])),cv2.INTER_NEAREST),np.bool) 
            
            if new_img.shape[1]>512:
                x_st = int((new_img.shape[1]-self.image_size)/2)
                x_end = int(x_st+self.image_size)
                new_img_temp = np.ones((new_img.shape[0],self.image_size,self.image_size),np.int16)*-1000
                new_img_temp = new_img[:,x_st:x_end,x_st:x_end]
                new_img = new_img_temp
                
                if not self.img_only:
                    new_mask_temp = np.zeros_like(new_img_temp,np.uint8)
                    new_mask_temp = new_mask[:,x_st:x_end,x_st:x_end]
                    new_mask = new_mask_temp
                    
            else:
                x_st = int((self.image_size-new_img.shape[1])/2)
                x_end = int(x_st + new_img.shape[1])
                new_img_temp = np.ones((new_img.shape[0],self.image_size,self.image_size),np.int16)*-1000
                new_img_temp[:,x_st:x_end,x_st:x_end] = new_img
                new_img = new_img_temp
                
                if not self.img_only:
                    new_mask_temp = np.zeros_like(new_img_temp,np.uint8)
                    new_mask_temp[:,x_st:x_end,x_st:x_end] = new_mask
                    new_mask = new_mask_temp
                
                 
                
    
            if self.verbosity:
                if not self.img_only:
                    print(new_img.dtype,new_mask.dtype)
                    print('Initial shape',img.shape)
                    print('Image in mm shape',new_img.shape)
            
            img = new_img
            msk = new_mask
        ####    
        
        if self.equalize_hist and img.dtype == np.uint8:
            img = cv2.equalizeHist(np.squeeze(img))
            if self.verbosity:
                print('Histogram equalization')
            
        if self.use_window and self.window_params:
            img = self._apply_window(img)
            if self.verbosity:
                print('Apply window')
                
                
        if self.resample_int_val and self.step:
            img = self._resample_intensities(img)
            if self.verbosity:
                print('Resample intensity values')        
    
    
        if self.normalize and self.norm_coeff:
            img = self._normalize_image(img)
            if self.verbosity:
                print('Normalize intensity values')
                
        
        # if augment then horizontal flip half the time
        if self.augment:# and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
            if self.verbosity:
                print('Using augmentation')
        
        if self.verbosity:
            print('Image and mask shape: ',img.shape,msk.shape)
            
        return img, msk
    
    def _load_patient_for_predict(self, filename, eval_mode=False,img_type=np.int16):
        # load image as numpy array
        temp_data = sitk.ReadImage(self.patient_dict[filename][0])   
        img = np.array(sitk.GetArrayFromImage(temp_data),img_type)
        print(img.shape)
        image_transform_params={'original_shape':img.shape,'normalized_shape':None, 'crop_type':None, 'xy_st':None,
                                'xy_end':None, 'z_st':None, 'z_end':None, 'img_origin':temp_data.GetOrigin(),'original_spacing':temp_data.GetSpacing() }
        
        if self.img_only:
            msk = None
        else:
            msk = sitk.GetArrayFromImage(sitk.ReadImage(self.patient_dict[filename][1]))
        # resize image
        
        if self.reshape and not eval_mode:
            new_shape = (int(img.shape[0]),int(temp_data.GetSpacing()[0]*img.shape[1]),
                                                      int(temp_data.GetSpacing()[1]*img.shape[2]))
            
            
            new_img = np.array(le.resize_3d_img(img,new_shape,cv2.INTER_NEAREST),np.int16)
            if not self.img_only:
                msk = np.array(le.resize_3d_img(msk,new_shape,cv2.INTER_NEAREST),np.uint8) > 0
                
            image_transform_params['normalized_shape'] = new_shape
    
            if new_img.shape[1]>self.image_size:
                x_st = int((new_img.shape[1]-self.image_size)/2)
                x_end = int(x_st+self.image_size)
                new_img_temp = np.ones((new_img.shape[0],self.image_size,self.image_size),np.int16)*-1000
                new_img_temp = new_img[:,x_st:x_end,x_st:x_end]
                img = new_img_temp
                
                if not self.img_only:
                    msk_temp = np.zeros((new_img.shape[0],self.image_size,self.image_size),np.uint8)
                    msk_temp = msk[:,x_st:x_end,x_st:x_end]
                    msk = msk_temp
                    
                #save parameters for image transformation
                image_transform_params['crop_type'] = 0
                image_transform_params['xy_st'] = x_st
                image_transform_params['xy_end'] = x_end
                
            else:
                x_st = int((self.image_size - new_img.shape[1])/2)
                x_end = int(x_st+new_img.shape[1])
                new_img_temp = np.ones((new_img.shape[0],self.image_size,self.image_size),np.int16)*-1000
                new_img_temp[:,x_st:x_end,x_st:x_end] = new_img
                img = new_img_temp
                
                if not self.img_only:
                    msk_temp = np.zeros((new_img.shape[0],self.image_size,self.image_size),np.uint8)
                    msk_temp[:,x_st:x_end,x_st:x_end] = msk
                    msk = msk_temp
                
                image_transform_params['crop_type'] = 1
                image_transform_params['xy_st'] = x_st
                image_transform_params['xy_end'] = x_end
                       
        
        if not self.img_only:         
            assert img.shape == msk.shape
                
        if self.extract_lungs:
            lung_mask,minb,maxb = le.get_seg_lungs(img,return_only_mask=False)
            
            if not self.img_only:
                lung_mask_with_tumor_mask = lung_mask.copy()
                kernel = le.get_kernel()
                
                for k,layer in enumerate(msk):
                    temp_layer = layer.copy()
                    lung_mask_with_tumor_mask[k,...]= np.array(lung_mask_with_tumor_mask[k,...],np.uint8)|cv2.dilate(np.array(temp_layer,np.uint8),kernel,iterations=3)
        
                new_img = le.apply_mask(img,lung_mask_with_tumor_mask)
                img = new_img[minb:maxb,...]
                msk = msk[minb:maxb,...]
                
                if self.verbosity:
                    print('Applying mask done')
                
            else:
                new_img = le.apply_mask(img,lung_mask) 
                img = new_img[minb:maxb,...]
                
            
            image_transform_params['z_st'] = minb
            image_transform_params['z_end'] = maxb
            
            
        
        if self.hu_intensity: #Set lower limit to -1000 HU
            img[img <-1000] = -1000
            
        if self.equalize_hist and img.dtype == np.uint8:
            img = cv2.equalizeHist(np.squeeze(img))
            
        if self.use_window and self.window_params:
            img = self._apply_window(img) 
            
        if self.calculate_snr:
            temp_img = img.copy()
            snr = self._calculate_snr(temp_img)
            
        if self.resample_int_val and self.step:
            img = self._resample_intensities(img)
            
        if self.normalize and self.norm_coeff and not eval_mode:
            img = self._normalize_image(img)
        
        if self.calculate_snr:
            return img,msk,snr,image_transform_params
        else:
            return img,msk,image_transform_params
        
        
    def __getitem__(self, index):
        #predict mode returns preprocessed patients images
        if self.predict:
            filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
            # load files
            imgs = [self._load_patient_for_predict(filename) for filename in filenames]
          
            if self.calculate_snr:
                 # create numpy batch
                imgs,msks,snrs,params = zip(*imgs)
                imgs = np.array(imgs)
                msks = np.array(msks)
                return imgs,msks,[self.patient_dict[filename][0] for filename in filenames],snrs,params
            else:
                imgs,msks,params = zip(*imgs)
                imgs = np.array(imgs)
                msks = np.array(msks)
                return imgs,msks,[self.patient_dict[filename][0] for filename in filenames],params
            
        # train mode: return images and masks
        else:
            if not index:
                self.__epoch_zero_dict = self.zero_dict.copy()
                self.__epoch_tum_dict = self.tum_dict.copy()
                if self.verbosity:
                    print('-'*7,'Inicialize epoch dictionaries:','-'*7)
            # load files
            if self.verbosity:
                print('Index:',index)
            arr=[]
            zero_arr=[]
                
            tum_key = list(self.__epoch_tum_dict.keys())[0]
            t_img,t_msk = self._load_patient(tum_key)
            zero_key = tum_key
            
            try:
                self.__epoch_zero_dict[zero_key]
            except KeyError:
                zero_key = list(self.__epoch_zero_dict.keys())[0]
                
            
            while len(zero_arr) < int(self.batch_size*self.zero_sl_ratio) :
                if self.__epoch_zero_dict[zero_key]:
                    temp_zero_ind = self.__epoch_zero_dict.get(zero_key).pop()
                    if np.sum(t_img[temp_zero_ind,...].flatten())<20:
                        print('*'*50)
                        print('problrm with index:',temp_zero_ind)
                        print('*'*50)
                        
                    zero_arr.append((t_img[temp_zero_ind,...],t_msk[temp_zero_ind,...]))   #zeroslices
                else:
                    _=self.__epoch_zero_dict.pop(zero_key)
                    zero_key = list(self.__epoch_zero_dict.keys())[0]
                    t_img,t_msk = self._load_patient(zero_key)
            
            while len(arr) < self.batch_size -int(self.batch_size*self.zero_sl_ratio) :
                if self.__epoch_tum_dict[tum_key]:
                    temp_tum_ind = self.__epoch_tum_dict.get(tum_key).pop()
                    arr.append((t_img[temp_tum_ind,...],t_msk[temp_tum_ind,...]))
                else:
                    _=self.__epoch_tum_dict.pop(tum_key)
                    tum_key = list(self.__epoch_tum_dict.keys())[0]
                    t_img,t_msk = self._load_patient(tum_key)
                    
            temp_list = arr+zero_arr
            random.shuffle(temp_list)
            imgs,msks = zip(*temp_list)
            imgs = np.expand_dims(np.array(imgs),-1)
            msks = np.expand_dims(np.array(msks)>0,-1)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle and not self.predict:
            random.shuffle(self.filenames)
        if self.verbosity:
                print('Epoch end')
        self.__epoch_tum_dict = {}
        self.__epoch_zero_dict = {}
            
    def _normalization_init(self):
        if self.normalize and len(self.norm_coeff)!=2 and not self.predict:
            print('-'*7,'Statistics evaluation started','-'*7)
            sys.stdout.flush()
            
            mean = np.mean([np.mean(self._load_patient_for_predict(filename,eval_mode=True)[0].flatten()) for filename in tqdm(self.filenames)])
            std = np.mean([np.std(self._load_patient_for_predict(filename,eval_mode=True)[0].flatten()) for filename in tqdm(self.filenames)])
            
            print('-'*7,'Mean and std are calculated','-'*7)
            print('-'*7,'Mean = %d'%mean,'-'*7,'Std= %d'%std,'-'*7)
            self.norm_coeff = [mean,std]
            
    def _normalize_image(self,image):
        image-=self.norm_coeff[0]
        image/=self.norm_coeff[1]
        return image
    
    def _apply_window(self,img):
        new_img = img.copy()
        WW = self.window_params[0]
        WL = self.window_params[1]
        assert (WW<2000) and (WW >0) and  (WL<200) and (WL >-1000)
        up_lim,low_lim = WL+WW/2,WL-WW/2
        if low_lim< -1000:
            low_lim = -1000
        if self.verbosity:
            print('Window limits ',low_lim,up_lim)   
        new_img[np.where(img<low_lim)] = low_lim
        new_img[np.where(img>up_lim)] = up_lim
        return new_img
    
    def _calculate_snr(self,img):
        temp_img = img[np.where(img>-999)]
        signal = np.mean(temp_img)
        std_noise = np.std(temp_img)
        SNR_dB_temp = 20*np.log10(np.abs(signal/std_noise))
        
        return SNR_dB_temp
        
    def _resample_intensities(self,img):
        v_count=0
        filtered = img.copy()
        filtered+=1000
        resampled = np.zeros_like(filtered)
        max_val_img = np.max(filtered.flatten())
        for st in range(self.step,max_val_img+self.step,self.step):
            resampled[(filtered<=st)&(filtered>=st-self.step)] = v_count
            v_count+=1
        if self.verbosity:
            print('Amount of unique values, original img: ',len(np.unique(img.flatten())),'resampled img: ',len(np.unique(resampled.flatten())))
        return np.array(resampled,dtype=np.uint8)
    
    
    def _patients_init(self,size_eval):
        if not self.predict:
            print('-'*7,'Loading patient info','-'*7)
            if size_eval:
                print('-'*7,'Tumor size evaluation started','-'*7)
            sys.stdout.flush()
            temp_tumor_dict,temp_zero_dict={},{}
            for pat in tqdm(self.filenames):
                temp_msk_im = sitk.ReadImage(self.patient_dict[pat][1])
                temp_img = sitk.ReadImage(self.patient_dict[pat][0])
                temp_msk = sitk.GetArrayFromImage(temp_msk_im)
                
                if self._slice_thickness and int(temp_img.GetSpacing()[2]) != self._slice_thickness:
                    pass
                else:
                    z_tumor_distr = np.sum(temp_msk,axis=(1,2))
                    tum_indicies =  np.where(z_tumor_distr>self._tum_size/np.prod([*temp_msk_im.GetSpacing()][:2]))[0]
                    zero_indicies =  np.where(z_tumor_distr==0)[0]
                    random.shuffle(tum_indicies)
                    random.shuffle(zero_indicies)
                    temp_tumor_dict[pat] = tum_indicies.tolist()
                    temp_zero_dict[pat] = zero_indicies.tolist()
                    
                    if size_eval:
                        temp_distr = list(filter(lambda x: x>0,np.round(z_tumor_distr*np.prod([*temp_msk_im.GetSpacing()][:2]),decimals=1)))   #mm^2
                        self.tum_size_distr+=temp_distr
                        self.tum_volume_distr[self.patient_dict[pat][0].split('\\')[-2]] = np.round(np.sum(z_tumor_distr)*temp_msk_im.GetSpacing()[0]*temp_msk_im.GetSpacing()[1]*temp_msk_im.GetSpacing()[2]*0.001,2)    #ml
                        
            self.tum_dict = temp_tumor_dict
            self.zero_dict = temp_zero_dict
        else: print('-'*7,'Loading patients in predict mode','-'*7)
            
     
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            len_tum_dict = np.sum([len(self.tum_dict[x]) for x in self.tum_dict.keys()])
            len_zero_dict = np.sum([len(self.zero_dict[x]) for x in self.zero_dict.keys()])
            if self.verbosity:
                print('-'*7,'Generator contained %d tumor slices and %d empty slices'%(len_tum_dict,len_zero_dict),'-'*7)
            
            size = np.min([int(len_tum_dict/(0.001+self.batch_size -int( self.batch_size*self.zero_sl_ratio))),
                           int(len_zero_dict/(0.001+int( self.batch_size*self.zero_sl_ratio)))])
    
            if size <= int((len_tum_dict+len_zero_dict)/self.batch_size):
                return size
            else :
                return int((len_tum_dict+len_zero_dict)/self.batch_size)
            
            # return full batches only, to maintain training multi GPU models
            
        
   
    
    
    
#usage example    
#    
#st =time.time()
#patients_gen = Patient_data_generator(Patient_dict,batch_size=8, image_size=512, shuffle=True, augment=False, predict=False, 
#                                      normalize = False,tumor_size_low_lim = 20, zero_sl_ratio = 0,use_window = True,
#                                      window_params=[1500,-600],size_eval=True,verbosity =True)#,norm_coeff = [-748,421] )
#end =time.time() - st
#print('Time spend:',round(end/60.,2),' mins')
#
#print(len(patients_gen))
#print(np.mean(list(patients_gen.tum_volume_distr.values())))
#st =time.time()
#val_gen = Patient_data_generator(val_pat_dict,batch_size=1, image_size=512, shuffle=False, augment=False, predict=True, 
#                                      normalize = True,norm_coeff = [-745.25934,406.79068],use_window = True,window_params=[1500,-600] )
#end =time.time() - st
#print('Time spend:',round(end/60.,2),' mins')

  

#for img,mask in patients_gen:
#    print(img.shape,mask.shape)
#    for i in range(img.shape[0]):
#        plt.figure(figsize=(10,10))
#        plt.subplot(121)
#        plt.imshow(img[i,...].reshape(512,512),cmap='bone')
#        plt.subplot(122)
#        plt.imshow(mask[i,...].reshape(512,512),cmap='bone')
#    plt.figure()
#    plt.hist(img[0,...].flatten())
#    break


#for img in val_gen:
#    print(img.shape)
#    img = np.squeeze(img)
#    for i in range(0,img.shape[0],10):
#        plt.figure(figsize=(10,5))
#        plt.subplot(121)
#        plt.imshow(img[i,...].reshape(512,512),cmap='bone')
#        plt.subplot(122)
#        plt.hist(img[i,...].flatten())
#        print(np.mean(img[i,...].flatten()),np.std(img[i,...].flatten()))
    #plt.subplot(133)
    #plt.hist(img[0,...].flatten())
#    break
    
    

