# Data augmentation script to increase training data through translation,
# scaling, and contrast modifications 

# Departments of Medical Physics and Radiology
# University of Wisconsin-Madison, WI, USA.
# - Ruiqi Geng (rgeng5@wisc.edu)
# - Diego Hernando (dhernando@wisc.edu)
# - Dec 20, 2022

# Please cite the following paper:
# Geng, R., Buelo, C. J., Sundaresan, M., Starekova, J.,
# Panagiotopoulos, N., Oechtering, T. H., ... & Hernando, D. (2022).
# Automated MR image prescription of the liver using deep learning:
# Development, evaluation, and prospective implementation. Journal of
# Magnetic Resonance Imaging. doi: 10.1002/jmri.28564. Epub 2022 Dec 30.
# PMID: 36583550.

import os
import sys
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance
import glob
import os
from pathlib import Path

os.chdir("img/")
# Scaling, translation, & contrast

for file in glob.glob("*.png"):
    try:
        img_name = os.path.splitext(file)
        txt_name = img_name[0]+'.txt'
        img = Image.open(file)
        noise_mean = np.mean(np.partition(img, 100)[:100])

        # import txt files
        labels0 = np.loadtxt(txt_name)
        if labels0.shape[0] > 5:
            print(img_name[0]+'Number of labels exceeds limit')
        
        if labels0.shape[0] > 0:
            labels = np.zeros(labels0.shape)

            # Scaling
            for sf0 in range(2, 15, 4): # 4 times
                sf = sf0*0.1
                # print(sf)
                width = round(512 * sf)
                # width and height have to be even numbers
                if np.mod(width,2)==1:
                    width = width + 1

                    height = width
                    img_new = img.resize([width,height])

                    if sf <= 1:
                        img1=np.pad(img_new, ((round((512-width)/2),round((512-width)/2)),(round((512-height)/2),round((512-height)/2))), constant_values=8)
                        img1[img1<noise_mean]=noise_mean
                        img2 = Image.fromarray(img1)

                    else:
                        img1=img_new.crop((width/2-256,height/2-256,width/2+256,height/2+256))
                        img2 = img1

                    for ll in range(0,labels0.shape[0],1):
                        labels[ll,1] = 0.5-(0.5-labels0[ll,1])*sf
                        labels[ll,2] = 0.5-(0.5-labels0[ll,2])*sf
                        labels[ll,3] = labels0[ll,3]*sf
                        labels[ll,4] = labels0[ll,4]*sf
                        labels[:,0] = labels0[:,0]

                    # Specify contraints

                    limit_class=max(labels0[:,0])
                    if limit_class == 2:
                        limit_class = 1
                    liver_index=list(labels0[:,0]).index(limit_class) #0 for liver; 1 for body
                    
                    x_min=round((labels[liver_index,1]-labels[liver_index,3]/2)*512)
                    y_min=round((labels[liver_index,2]-labels[liver_index,4]/2)*512)
                    x_max=round((labels[liver_index,1]+labels[liver_index,3]/2)*512)
                    y_max=round((labels[liver_index,2]+labels[liver_index,4]/2)*512)

                    if x_min < 0 or x_max > 512 or y_min < 0 or y_max > 512:
                        break

                    # Translation
                    a = 1
                    b = 0
                    # c = 5 #left/right (i.e. 5/-5)
                    d = 0
                    e = 1
                    # f = 5 #up/down (i.e. 5/-5)
                    labelsT = np.zeros(labels.shape)

                    for c in range(-50,60,50): # 3 times
                        for f in range(-50,60,50): # 3 times
                        
                            labelsT[:,1] = labels[:,1]-c/512
                            labelsT[:,2] = labels[:,2]-f/512
                            labelsT[:,3] = labels[:,3]
                            labelsT[:,4] = labels[:,4]
                            labelsT[:,0] = labels[:,0]

                            imgT = img2.transform(img2.size, Image.AFFINE, (a, b, c, d, e, f))
                            # plt.figure(figsize=(17,7))
                            # plt.subplot(1,2,1)
                            # plt.title('image')
                            # plt.imshow(imgT,cmap = plt.get_cmap(name = 'gray'))

                            # Specify contraints
                            liver_index=list(labelsT[:,0]).index(limit_class) #0 for liver; 1 for body
                            
                            x_min=round((labelsT[liver_index,1]-labelsT[liver_index,3]/2)*512)
                            y_min=round((labelsT[liver_index,2]-labelsT[liver_index,4]/2)*512)
                            x_max=round((labelsT[liver_index,1]+labelsT[liver_index,3]/2)*512)
                            y_max=round((labelsT[liver_index,2]+labelsT[liver_index,4]/2)*512)
                        
                            if x_min < 0 or x_max > 512 or y_min < 0 or y_max > 512:
                                continue
                            
                            # Contrast
                            enhancer = ImageEnhance.Contrast(imgT)
                            for f2 in range (2,20,4):   #5 times
                                factor = f2/10
                                im_output = enhancer.enhance(factor)
                                #im_output.save(img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.png','png')
                                im_output.save('../Augmented/'+img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.png','png')
                                #np.savetxt(img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.txt', labelsT, delimiter=' ', fmt=['%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'])
                                np.savetxt('../Augmented/'+img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.txt', labelsT, delimiter=' ', fmt=['%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'])

        else:
            # print(img_name[0]+'label empty')
            # Scaling
            for sf0 in range(2, 15, 4): # 4 times
                sf = sf0*0.1
                # print(sf)
                width = round(512 * sf)
                # width and height have to be even numbers
                if np.mod(width,2)==1:
                    width = width + 1

                    height = width
                    img_new = img.resize([width,height])

                    if sf <= 1:
                        img1=np.pad(img_new, ((round((512-width)/2),round((512-width)/2)),(round((512-height)/2),round((512-height)/2))), constant_values=8)
                        img1[img1<noise_mean]=noise_mean
                        img2 = Image.fromarray(img1)

                    else:
                        img1=img_new.crop((width/2-256,height/2-256,width/2+256,height/2+256))
                        img2 = img1

                    # Translation
                    a = 1
                    b = 0
                    # c = 5 #left/right (i.e. 5/-5)
                    d = 0
                    e = 1
                    # f = 5 #up/down (i.e. 5/-5)

                    for c in range(-50,60,50): # 3 times
                        for f in range(-50,60,50): # 3 times
                            imgT = img2.transform(img2.size, Image.AFFINE, (a, b, c, d, e, f))
                            # plt.figure(figsize=(17,7))
                            # plt.subplot(1,2,1)
                            # plt.title('image')
                            # plt.imshow(imgT,cmap = plt.get_cmap(name = 'gray'))

                            # Contrast
                            enhancer = ImageEnhance.Contrast(imgT)
                            for f2 in range (2,20,4):   #5 times
                                factor = f2/10
                                im_output = enhancer.enhance(factor)
                                #im_output.save(img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.png','png')
                                im_output.save('../Augmented/'+img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.png','png')
                                labels0=[]
                                np.savetxt('../Augmented/'+img_name[0]+'_S'+str(round(sf,1))+'_X'+str(-c)+'_Y'+str(-f)+'_C'+str(factor)+'.txt', labels0)
    
    except:
        pass


