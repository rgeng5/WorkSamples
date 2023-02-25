# Automated MR Image Prescription of the Liver Using Deep Learning: Development, Evaluation, and Prospective Implementation

Ruiqi Geng[1,2], MSc, Collin J. Buelo[1,2], MSc, Mahalakshmi Sundaresan[3], MSc, Jitka Starekova[1], MD, Nikolaos Panagiotopoulos[1,4], MD, Thekla H. Oechtering[1,4], MD, Edward M. Lawrence[1], MD, PhD, Marcin Ignaciuk[1], MD, Scott B. Reeder[1,2,5,6,7], MD, PhD, Diego Hernando[1,2,3,7], PhD

Departments of Radiology[1], Medical Physics[2], Electrical and Computer Engineering[3], Medicine[5], Emergency Medicine[6], Biomedical Engineering[7], University of Wisconsin, Madison, WI, USA (1111 Highland Ave, Madison, WI 53705)
[4]Department of Radiology and Nuclear Medicine, Universität zu Lübeck, Lübeck, Germany (Haus A, Ratzeburger Allee 160, 23562 Lübeck, Germany)


**Introduction**

In this paper, we developed, implemented, and evaluated an AI-based (YOLOv3) fully automated prescription method for liver MRI. The AI-based automated liver image prescription demonstrated promising performance across the patients, pathologies, and field strengths studied. 
The demo scripts and the YOLOv3 weights obtained after training for this work are shared here.

**Scripts**

Markup : * Pre-processing script to increase contrast around the liver and convert DICOMs to PNGs, slice by slice: PreparePNG.m

 	 * Two data augmentation schemes were used to enlarge the image dataset for training: reflections and full (reflections + translation + scaling + contrast). DataAug_refection.py performs reflections only; it can be run before DataAug_translate_scale_contrast.py to perform all 4 types of data augmentation

⋅⋅*Data extraction script to obtain meta data from DICOM headers: GrabInfo.m

⋅⋅*Post-processing script to obtain 3D bounding boxes from 2D bounding boxes by manual labeling and YOLO predictions: TestEval.m

⋅⋅*Data visualization script to display results and analyze AI performance across subpopulations: figures.m

⋅⋅*To set up Docker images for darknet Yolo v4, v3 and v2: https://github.com/cjbueloMP/darknet-docker, forked from https://github.com/daisukekobayashi/darknet-docker


**YOLOv3 configuration files**

⋅⋅*Full network without data augmentation: yolov3.cfg	
	
⋅⋅*Tiny network without data augmentation: yolov3-tiny.cfg


#### Please cite the following paper

Geng, R., Buelo, C. J., Sundaresan, M., Starekova, J., Panagiotopoulos, N., Oechtering, T. H., ... & Hernando, D. (2022). Automated MR image prescription of the liver using deep learning: Development, evaluation, and prospective implementation. Journal of Magnetic Resonance Imaging. doi: 10.1002/jmri.28564. Epub 2022 Dec 30. PMID: 36583550.
