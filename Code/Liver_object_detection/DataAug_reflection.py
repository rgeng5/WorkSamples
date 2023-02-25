# Data augmentation script to increase training data through reflections 

# Departments of Medical Physics and Radiology
# University of Wisconsin-Madison, WI, USA.
# - Ruiqi Geng (rgeng5@wisc.edu)
# - Dec 20, 2022

# Please cite the following paper:
# Geng, R., Buelo, C. J., Sundaresan, M., Starekova, J.,
# Panagiotopoulos, N., Oechtering, T. H., ... & Hernando, D. (2022).
# Automated MR image prescription of the liver using deep learning:
# Development, evaluation, and prospective implementation. Journal of
# Magnetic Resonance Imaging. doi: 10.1002/jmri.28564. Epub 2022 Dec 30.
# PMID: 36583550.

import os
import numpy as np
from PIL import Image, ImageOps
import glob

os.chdir("img/")
# flip, mirror, flip_mirror
for file in glob.glob("*.png"):
    try:
        img_name = os.path.splitext(file)
        txt_name = img_name[0]+'.txt'
        img = Image.open(file)

        # import txt files
        labels0 = np.loadtxt(txt_name)
        if labels0.shape[0] > 5:
            print(img_name[0]+'Number of labels exceeds limit')
            # continue
        
        if labels0.shape[0] > 0:
            print(img_name[0])
            labels = np.zeros(labels0.shape)
            labelsM = np.zeros(labels0.shape)
            labelsFM = np.zeros(labels0.shape)

            # flip
            img_flip = ImageOps.flip(img)

            for ll in range(0,labels0.shape[0],1):
                labels[ll,1] = labels0[ll,1]
                labels[ll,2] = 1-labels0[ll,2]
                labels[ll,3] = labels0[ll,3]
                labels[ll,4] = labels0[ll,4]
                labels[:,0] = labels0[:,0]
            img_flip.save('../Augmented/'+img_name[0]+'_Flip.png','png')
            np.savetxt('../Augmented/'+img_name[0]+'_Flip.txt', labels, delimiter=' ', fmt=['%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'])

            # mirror
            img_mirror = ImageOps.mirror(img)
            for ll in range(0,labels0.shape[0],1):
                labelsM[ll,1] = 1-labels0[ll,1]
                labelsM[ll,2] = labels0[ll,2]
                labelsM[ll,3] = labels0[ll,3]
                labelsM[ll,4] = labels0[ll,4]
                labelsM[:,0] = labels0[:,0]
            img_mirror.save('../Augmented/'+img_name[0]+'_Mirror.png','png')
            np.savetxt('../Augmented/'+img_name[0]+'_Mirror.txt', labelsM, delimiter=' ', fmt=['%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'])

            # flip + mirror
            img_flip_mirror = ImageOps.mirror(img_flip)
            for ll in range(0,labels0.shape[0],1):
                labelsFM[ll,1] = 1-labels0[ll,1]
                labelsFM[ll,2] = 1-labels0[ll,2]
                labelsFM[ll,3] = labels0[ll,3]
                labelsFM[ll,4] = labels0[ll,4]
                labelsFM[:,0] = labels0[:,0]
            img_flip_mirror.save('../Augmented/'+img_name[0]+'_Flip_Mirror.png','png')
            np.savetxt('../Augmented/'+img_name[0]+'_Flip_Mirror.txt', labelsFM, delimiter=' ', fmt=['%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'])

        else:
            # print(img_name[0]+'label empty')
            # flip
            img_flip = ImageOps.flip(img)
            img_flip.save('../Augmented/'+img_name[0]+'_Flip.png','png')
            labels=[]
            np.savetxt('../Augmented/'+img_name[0]+'_Flip.txt', labels)

            # mirror
            img_mirror = ImageOps.mirror(img)
            img_mirror.save('../Augmented/'+img_name[0]+'_Mirror.png','png')
            labelsM=[]
            np.savetxt('../Augmented/'+img_name[0]+'_Mirror.txt', labelsM)

            # flip + mirror
            img_flip_mirror = ImageOps.mirror(img_flip)
            img_flip_mirror.save('../Augmented/'+img_name[0]+'_Flip_Mirror.png','png')
            labelsFM=[]
            np.savetxt('../Augmented/'+img_name[0]+'_Flip_Mirror.txt', labelsFM)

    except:
        pass
