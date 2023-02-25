%% Post-processing script to obtain 3D bounding boxes from 2D bounding boxes by manual labeling and YOLO predictions

% Departments of Medical Physics and Radiology
% University of Wisconsin-Madison, WI, USA.
% - Ruiqi Geng (rgeng5@wisc.edu)
% - Diego Hernando (dhernando@wisc.edu)
% - Dec 20, 2022

% Please cite the following paper:
% Geng, R., Buelo, C. J., Sundaresan, M., Starekova, J.,
% Panagiotopoulos, N., Oechtering, T. H., ... & Hernando, D. (2022).
% Automated MR image prescription of the liver using deep learning:
% Development, evaluation, and prospective implementation. Journal of
% Magnetic Resonance Imaging. doi: 10.1002/jmri.28564. Epub 2022 Dec 30.
% PMID: 36583550.
%% raw_test is an array that contains all meta data and bounding box coordinates
pts=unique(raw_test(:,2));
for p=1:length(pts)
    disp(num2str(p/length(pts)))
    files=dir(['/dicoms_folder/' sprintf('%04.4d',pts(p))]);
    for f=1:length(files)-2
    info=dicominfo(['/dicoms_folder/' sprintf('%04.4d',pts(p)) '/' files(f+2).name]);
    yolo_i=intersect(find(pt_index==pts(p)),find(DICOM_index==info.InstanceNumber));
    raw_test(yolo_i,4)=f;
    if info.ImageOrientationPatient==[1;0;0;0;1;0]
        orientation=1; %Axial
    else
        if info.ImageOrientationPatient==[0;1;0;0;0;-1]
            orientation=2; %Sag
        else
            orientation=3; %Cor
        end
    end
    raw_test(yolo_i,5)=orientation;
    raw_test(yolo_i,6)=info.ImagePositionPatient(1);
    raw_test(yolo_i,7)=info.ImagePositionPatient(2);
    raw_test(yolo_i,8)=info.ImagePositionPatient(3);
    raw_test(yolo_i,9)=info.PixelSpacing(1);
    raw_test(yolo_i,10)=info.PixelSpacing(2);
    raw_test(yolo_i,11)=info.SliceThickness;
    raw_test(yolo_i,12)=info.SpacingBetweenSlices;
    raw_test(yolo_i,13)=info.Rows;
    raw_test(yolo_i,14)=info.Columns;
    end
end

%% get AxImgList, SagImgList, and CorImgList
AxImgList={'Ax'};
SagImgList={'Sag'};
CorImgList={'Cor'};

countAx=1;
countSag=1;
countCor=1;

for l=1:length(raw_test)
    if raw_test(l,5)==1
        countAx=countAx+1;
        AxImgList{countAx}=[sprintf('%04.4d',raw_test(l,2)) '-' num2str(raw_test(l,3)) '.png'];
    end
    
    if raw_test(l,5)==2
        countSag=countSag+1;
        SagImgList{countSag}=[sprintf('%04.4d',raw_test(l,2)) '-' num2str(raw_test(l,3)) '.png'];
    end
    
    if raw_test(l,5)==3
        countCor=countCor+1;
        CorImgList{countCor}=[sprintf('%04.4d',raw_test(l,2)) '-' num2str(raw_test(l,3)) '.png'];
    end
end

writecell(AxImgList', 'testAx.txt');
writecell(SagImgList', 'testSag.txt')
writecell(CorImgList', 'testCor.txt')

%% if these tables exists
CorImgList=readtable('testCor.txt');
AxImgList=readtable('testAx.txt');
SagImgList=readtable('testSag.txt');
pts=unique(raw_test(:,2));
%% get predicted coordinates !!changes with training iterations!!
raw_test(:,15:70)=0;
Predicted=readtable('test_results.txt','Delimiter', '(');
LabelList={'LiverAx','BodyAx','ArmsAx','LiverCor','BodyCor','LiverSag','BodySag'};
count_box=0;
for i=1:height(Predicted)
    label=extractBefore(cell2mat(table2cell(Predicted(i,1))),":");
    i
    for l=1:length(LabelList)
        if strcmp(label,cell2mat(LabelList(l))) %find the label
            if isempty(cell2mat(table2cell(Predicted(i,2))))~=1 %in case no bbox was predicted
            coor0=sscanf(cell2mat(table2cell(Predicted(i,2))),'left_x: %d top_y: %d width: %d height: %d)');
            % normalize predicted coordinates
            left_x=coor0(1);
            top_y=coor0(2);
            width=coor0(3);
            hheight=coor0(4);
            center_x=(left_x+width/2)/512; %All PNG were interpolated to 512*512
            center_y=(top_y+hheight/2)/512;
            width_norm=width/512;
            height_norm=hheight/512;
            coor=[center_x,center_y,width_norm,height_norm];
            
            if length(cell2mat(table2cell(Predicted(i-1,1)))) > 30 %first box in the image
            %GetPNG=sscanf(cell2mat(table2cell(Predicted(i-1,1))),'Enter Image Path: /data/img/test/%d-%d.png: Predicted in %d milli-seconds.');
            GetPNG=sscanf(cell2mat(table2cell(Predicted(i-1,1))),'Enter Image Path: img/img0/%d-%d.png: Predicted in %d milli-seconds.');

            pt=GetPNG(1);
            png_num=GetPNG(2);
            count_box=1;
            for r=1:size(raw_test,1) %find where to put in the raw_test table
                if raw_test(r,2)==pt && raw_test(r,3)==png_num
                    raw_test(r,count_box*10+5)=l;
                    raw_test(r,count_box*10+6:count_box*10+9)=coor;
                end
            end
            
            else %multiple boxes in same png
                count_box=count_box+1;
            for r=1:size(raw_test,1) %find where to put in the raw_test table
                if raw_test(r,2)==pt && raw_test(r,3)==png_num
                    raw_test(r,count_box*10+5)=l;
                    raw_test(r,count_box*10+6:count_box*10+9)=coor;
                end
            end
                
            end
            else
                disp([num2str(i) 'line has empty bbox.'])
            end
        end
    end
end
%% get labeled coordinates in coronal slices
searchFolder='/labeled_folder/';

for a=1:height(CorImgList)
    a
    png_name=cell2mat(table2cell(CorImgList(a,1)));
    txt_name= strrep(png_name,'png','txt');
    if exist([searchFolder txt_name]) == 2
    fileID = fopen([searchFolder txt_name],'r');
    formatSpec = '%f';
    A = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    label_num=length(A)/5;
    pp=sscanf(txt_name,'%d-%d.txt');
    pt=pp(1);
    png_num=pp(2);
    for n=1:label_num
        label_code=A((n-1)*5+1)+1;
        for r=1:length(raw_test) %find where to put in the raw_test table
            if raw_test(r,2)==pt && raw_test(r,3)==png_num
                pos=find(raw_test(r,15:end)==label_code);
                    if size(pos,2)==0
                        raw_test(r,55+5*(n-1):55+5*(n-1)+4)=[label_code; A((n-1)*5+2:(n-1)*5+5)]';
                    else
                    raw_test(r,14+pos+5:18+pos+5)=[label_code; A((n-1)*5+2:(n-1)*5+5)]';
                    end
            end
        end
    end
    end
end

%% get labeled coordinates in axial slices
for a=1:height(AxImgList)
    a
    png_name=cell2mat(table2cell(AxImgList(a,1)));
    txt_name= strrep(png_name,'png','txt');
    if exist([searchFolder txt_name]) == 2
    fileID = fopen([searchFolder txt_name],'r');
    formatSpec = '%f';
    A = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    label_num=length(A)/5;
    pp=sscanf(txt_name,'%d-%d.txt');
    pt=pp(1);
    png_num=pp(2);
    for n=1:label_num
        label_code=A((n-1)*5+1)+1;
        for r=1:length(raw_test) %find where to put in the raw_test table
            if raw_test(r,2)==pt && raw_test(r,3)==png_num
                pos=find(raw_test(r,15:end)==label_code);
                    if size(pos,2)==0
                        raw_test(r,55+5*(n-1):55+5*(n-1)+4)=[label_code; A((n-1)*5+2:(n-1)*5+5)]';
                    else
                    raw_test(r,14+pos+5:18+pos+5)=[label_code; A((n-1)*5+2:(n-1)*5+5)]';
                    end
            end
        end
    end
    end
end

%% get labeled coordinates in sagittal slices
for a=1:height(SagImgList)
    a
    png_name=cell2mat(table2cell(SagImgList(a,1)));
    txt_name= strrep(png_name,'png','txt');
    if exist([searchFolder txt_name]) == 2
    fileID = fopen([searchFolder txt_name],'r');
    formatSpec = '%f';
    A = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    label_num=length(A)/5;
    pp=sscanf(txt_name,'%d-%d.txt');
    pt=pp(1);
    png_num=pp(2);
    for n=1:label_num
        label_code=A((n-1)*5+1)+1;
        for r=1:length(raw_test) %find where to put in the raw_test table
            if raw_test(r,2)==pt && raw_test(r,3)==png_num
                pos=find(raw_test(r,15:end)==label_code);
                    if size(pos,2)==0
                        raw_test(r,55+5*(n-1):55+5*(n-1)+4)=[label_code; A((n-1)*5+2:(n-1)*5+5)]';
                    else
                    raw_test(r,14+pos+5:18+pos+5)=[label_code; A((n-1)*5+2:(n-1)*5+5)]';
                    end
            end
        end
    end
    end
end
%% calculate overlaps & display individual 2D images
union_set=[];
count_diff=0;
count_diff_2=0;
yolo_diff=[]; %record the i that is differently classified
yolo_diff_2=[]; %record the i that is differently classified
allIoU=[];
allIoU=zeros([7 8000]);
set_corr_bbox=[];
set_wrong_bbox=[];
%
for l=1:7 %length(LabelList) %stats of each class %!!Resume imwrite!!!!!!!!!!
    count_bbox=0;
    count_corr_bbox=0;
    count_wrong_bbox=0;
    union_set=[];
    union_set=zeros([1 8000]);
    
    for i=1:length(raw_test) %loop through png files
        pt=sprintf('%04.4d',raw_test(i,2));
        im_num= num2str(raw_test(i,3));
        if raw_test(i,5) == 1
        view='Ax';
        else
            if raw_test(i,5) == 2
                view='Sag';
            else
                view='Cor';
            end
        end


        for c=15:10:74 %loop through columns to find different classes of bbox
        
            bbox_label=[0,0,0,0];
            bbox_pred=[0,0,0,0];
            bbox_pred2=[0,0,0,0];
            
            if raw_test(i,c)==l 
                count_bbox=count_bbox+1;
                if raw_test(i,c+5)==l
                    count_corr_bbox=count_corr_bbox+1;
                    
                    bbox_pred=raw_test(i,c+1:c+4);
                    bbox_pred(1)=raw_test(i,c+1)-raw_test(i,c+3)/2;
                    bbox_pred(2)=raw_test(i,c+2)-raw_test(i,c+4)/2;
                    bbox_label=raw_test(i,c+6:c+9);
                    bbox_label(1)=raw_test(i,c+6)-raw_test(i,c+8)/2;
                    bbox_label(2)=raw_test(i,c+7)-raw_test(i,c+9)/2;
                    overlapRatio = bboxOverlapRatio(bbox_pred,bbox_label);
                    union_set(count_corr_bbox)=overlapRatio;
                    
%                     rectangle('Position',bbox_label*512,'EdgeColor','b','LineWidth',2)
%                     hold on
%                     rectangle('Position',bbox_pred*512,'EdgeColor','y','LineWidth',2)
%                     hold on %Only takes the first predicted bbox
%                     img = getframe(gcf);
%                     imwrite(img.cdata, ['2Dresults/' pt '_' view '_' im_num '_' cell2mat(LabelList(l)) '.png']);
                else
                    count_wrong_bbox=count_wrong_bbox+1;
                    
                end
                
            end
            
        end
    end
    allIoU=[allIoU;union_set];
    set_corr_bbox=[set_corr_bbox;count_corr_bbox];
    set_wrong_bbox=[set_wrong_bbox;count_wrong_bbox];
end
allIoU(:,max(set_corr_bbox)+1:end)=[];
allIoU(1:7,:)=[];

%% 2D Axial bounding boxes
pred_set_Ax=[];label_set_Ax=[];
pred_Ax_SI=[];label_Ax_SI=[];

for pt=1:length(pts) %for each patient
    
    pred_Ax_SI_min=[0,0,0,0];pred_Ax_SI_max=[0,0,0,0];
    label_Ax_SI_min=[0,0,0,0];label_Ax_SI_max=[0,0,0,0];
    
    pt_num=pts(pt);
    %disp(num2str(pt_num))
    
    Ax_rows0=intersect(find(raw_test(:,2)==pt_num),find(raw_test(:,5)==1));
    Sag_rows0=intersect(find(raw_test(:,2)==pt_num),find(raw_test(:,5)==2));
    Cor_rows0=intersect(find(raw_test(:,2)==pt_num),find(raw_test(:,5)==3));
   
    %Axial View
    
    for bb=1:3
    count=0;
    count2=0;
    for c=15:10:40
    Ax_pred=[];
    Ax_rows1=find(raw_test(Ax_rows0,c)==bb); %if that view has lth class of labels
    %Ax_rows=intersect(Ax_rows0(Ax_rows1),Ax_rows0(Ax_rows2));

    if size(Ax_rows1,1)>0
    count=count+1;
    Ax_pred0=raw_test(Ax_rows0(Ax_rows1),c+1:c+4);
    Ax_pred0(Ax_pred0(:,1)==0,:)=[];
    Ax_pred(:,1)=Ax_pred0(:,1)-Ax_pred0(:,3)/2;
    Ax_pred(:,2)=Ax_pred0(:,2)-Ax_pred0(:,4)/2;
    Ax_pred(:,3)=Ax_pred0(:,1)+Ax_pred0(:,3)/2;
    Ax_pred(:,4)=Ax_pred0(:,2)+Ax_pred0(:,4)/2;
    Ax_x_min_pred=min(Ax_pred(:,1));
    Ax_y_min_pred=min(Ax_pred(:,2));
    Ax_x_max_pred=max(Ax_pred(:,3));
    Ax_y_max_pred=max(Ax_pred(:,4));
    
    X_left=raw_test(Ax_rows0(Ax_rows1(1)),6); %X (left) physical origin
    X_spacing=raw_test(Ax_rows0(Ax_rows1(1)),9);
    X_columns=raw_test(Ax_rows0(Ax_rows1(1)),13);
    X_min_pred=X_left+Ax_x_min_pred*X_columns*X_spacing;
    X_max_pred=X_left+Ax_x_max_pred*X_columns*X_spacing;

    Y_ante=raw_test(Ax_rows0(Ax_rows1(1)),7); %Y (anterior) physical origin
    Y_spacing=raw_test(Ax_rows0(Ax_rows1(1)),10);
    Y_columns=raw_test(Ax_rows0(Ax_rows1(1)),14);
    Y_min_pred=Y_ante+Ax_y_min_pred*Y_columns*Y_spacing;
    Y_max_pred=Y_ante+Ax_y_max_pred*Y_columns*Y_spacing;
    
%     CenterW_X_pred=X_min_pred+(X_max_pred-X_min_pred)/2;
%     CenterW_Y_pred=Y_min_pred+(Y_max_pred-Y_min_pred)/2;
%     W_Ax_pred=X_max_pred-X_min_pred;
%     D_Ax_pred=Y_max_pred-Y_min_pred;
    pred_set_Ax(:,pt,bb)=squeeze([X_min_pred,X_max_pred,Y_min_pred,Y_max_pred]);
    pred_Ax_SI_min(count)=min(raw_test(Ax_rows0(Ax_rows1),8));
    pred_Ax_SI_max(count)=max(raw_test(Ax_rows0(Ax_rows1),8));
    end
    end
    pred_Ax_SI(1,pt,bb)=min(pred_Ax_SI_min);
    pred_Ax_SI(2,pt,bb)=max(pred_Ax_SI_max);
    
    label_c=sort([20,30,40,50,55,60,65],'descend');
    dummyset=[];
    for cc=1:length(label_c)
    c=label_c(cc);
    Ax_rows2=find(raw_test(Ax_rows0,c)==bb);
    Ax_label=[];
    if size(Ax_rows2,1)>0
    count2=count2+1;
    Ax_label0=raw_test(Ax_rows0(Ax_rows2),c+1:c+4);
    Ax_label0(Ax_label0(:,1)==0,:)=[];
    Ax_label(:,1)=Ax_label0(:,1)-Ax_label0(:,3)/2;
    Ax_label(:,2)=Ax_label0(:,2)-Ax_label0(:,4)/2;
    Ax_label(:,3)=Ax_label0(:,1)+Ax_label0(:,3)/2;
    Ax_label(:,4)=Ax_label0(:,2)+Ax_label0(:,4)/2;
    
%     Ax_x_min_label=min(Ax_label(:,1));
%     Ax_y_min_label=min(Ax_label(:,2));
%     Ax_x_max_label=max(Ax_label(:,3));
%     Ax_y_max_label=max(Ax_label(:,4));
    
    dummyset=[dummyset;[Ax_label(:,1),Ax_label(:,2),Ax_label(:,3),Ax_label(:,4)]];
    
    %Ax_label(:,1)
%     CenterW_X_label=X_min_label+(X_max_label-X_min_label)/2;
%     CenterW_Y_label=Y_min_label+(Y_max_label-Y_min_label)/2;
%     W_Ax_label=X_max_label-X_min_label;
%     D_Ax_label=Y_max_label-Y_min_label;
    label_Ax_SI_min(count2)=min(raw_test(Ax_rows0(Ax_rows2),8));
    label_Ax_SI_max(count2)=max(raw_test(Ax_rows0(Ax_rows2),8));
    end
    end
    label_Ax_SI(1,pt,bb)=min(label_Ax_SI_min);
    label_Ax_SI(2,pt,bb)=max(label_Ax_SI_max);

    if isempty(dummyset)==1    
    Ax_x_min_label=nan;
    Ax_y_min_label=nan;
    Ax_x_max_label=nan;
    Ax_y_max_label=nan;
    else
    Ax_x_min_label=min(dummyset(:,1));
    Ax_y_min_label=min(dummyset(:,2));
    Ax_x_max_label=max(dummyset(:,3));
    Ax_y_max_label=max(dummyset(:,4));
    end
    
    X_min_label=X_left+Ax_x_min_label*X_columns*X_spacing;
    X_max_label=X_left+Ax_x_max_label*X_columns*X_spacing;

    Y_min_label=Y_ante+Ax_y_min_label*Y_columns*Y_spacing;
    Y_max_label=Y_ante+Ax_y_max_label*Y_columns*Y_spacing;
    
    label_set_Ax(:,pt,bb)=squeeze([X_min_label,X_max_label,Y_min_label,Y_max_label]);
   
    end
end

%% 2D coronal bounding boxes
pred_set_Cor=[];label_set_Cor=[];
pred_Cor_AP=[];label_Cor_AP=[];

for pt=1:length(pts) %for each patient
    
    pred_Cor_AP_min=[0,0,0,0];pred_Cor_AP_max=[0,0,0,0];
    label_Cor_AP_min=[0,0,0,0];label_Cor_AP_max=[0,0,0,0];
    
    pt_num=pts(pt);
    %disp(num2str(pt_num))
    
    Cor_rows0=intersect(find(raw_test(:,2)==pt_num),find(raw_test(:,5)==3));

    %Cor View
    
    for bb=4:5
    count=0;
    count2=0;
    for c=15:10:40
    Cor_pred=[];
    Cor_rows1=find(raw_test(Cor_rows0,c)==bb); %if that view has lth class of labels
    %Cor_rows=intersect(Cor_rows0(Cor_rows1),Cor_rows0(Cor_rows2));
    
    if size(Cor_rows1,1)>0
    count=count+1;
    Cor_pred0=raw_test(Cor_rows0(Cor_rows1),c+1:c+4);
    
    Cor_pred0(Cor_pred0(:,1)==0,:)=[];
    Cor_pred(:,1)=Cor_pred0(:,1)-Cor_pred0(:,3)/2;
    Cor_pred(:,2)=Cor_pred0(:,2)-Cor_pred0(:,4)/2;
    Cor_pred(:,3)=Cor_pred0(:,1)+Cor_pred0(:,3)/2;
    Cor_pred(:,4)=Cor_pred0(:,2)+Cor_pred0(:,4)/2;
    Cor_x_min_pred=min(Cor_pred(:,1));
    Cor_y_min_pred=min(Cor_pred(:,2));
    Cor_x_max_pred=max(Cor_pred(:,3));
    Cor_y_max_pred=max(Cor_pred(:,4));
       
    X_left=raw_test(Cor_rows0(Cor_rows1(1)),6); %X (left) physical origin
    X_spacing=raw_test(Cor_rows0(Cor_rows1(1)),9);
    X_columns=raw_test(Cor_rows0(Cor_rows1(1)),13);
    X_min_pred=X_left+Cor_x_min_pred*X_columns*X_spacing;
    X_max_pred=X_left+Cor_x_max_pred*X_columns*X_spacing;
        
    Z_top=raw_test(Cor_rows0(Cor_rows1(1)),8);%Z (superior) physical origin
    Z_spacing=raw_test(Cor_rows0(Cor_rows1(1)),10);
    Z_columns=raw_test(Cor_rows0(Cor_rows1(1)),14);
    Z_max_pred=Z_top-Cor_y_min_pred*Z_columns*Z_spacing;
    Z_min_pred=Z_top-Cor_y_max_pred*Z_columns*Z_spacing;
        
%     CenterW_X_pred=X_min_pred+(X_max_pred-X_min_pred)/2;
%     CenterW_Y_pred=Y_min_pred+(Y_max_pred-Y_min_pred)/2;
%     W_Cor_pred=X_max_pred-X_min_pred;
%     D_Cor_pred=Y_max_pred-Y_min_pred;
    pred_set_Cor(:,pt,bb)=squeeze([X_min_pred,X_max_pred,Z_min_pred,Z_max_pred]);
    
    pred_Cor_AP_min(count)=min(raw_test(Cor_rows0(Cor_rows1),7));
    pred_Cor_AP_max(count)=max(raw_test(Cor_rows0(Cor_rows1),7));
    end
    end
    pred_Cor_AP(1,pt,bb)=min(pred_Cor_AP_min);
    pred_Cor_AP(2,pt,bb)=max(pred_Cor_AP_max);

    label_c=sort([20,30,40,50,55,60,65],'descend');
    dummyset=[];
    for cc=1:length(label_c)
    c=label_c(cc);
    Cor_rows2=find(raw_test(Cor_rows0,c)==bb);
    Cor_label=[];

    if size(Cor_rows2,1)>0 && sum(raw_test(Cor_rows0,c))~=0
    count2=count2+1;
    Cor_label0=raw_test(Cor_rows0(Cor_rows2),c+1:c+4);
    Cor_label0(Cor_label0(:,1)==0,:)=[];
    Cor_label(:,1)=Cor_label0(:,1)-Cor_label0(:,3)/2;
    Cor_label(:,2)=Cor_label0(:,2)-Cor_label0(:,4)/2;
    Cor_label(:,3)=Cor_label0(:,1)+Cor_label0(:,3)/2;
    Cor_label(:,4)=Cor_label0(:,2)+Cor_label0(:,4)/2;
    
    dummyset=[dummyset;[Cor_label(:,1),Cor_label(:,2),Cor_label(:,3),Cor_label(:,4)]];
    
    label_Cor_AP_min(count2)=min(raw_test(Cor_rows0(Cor_rows2),7));
    label_Cor_AP_max(count2)=max(raw_test(Cor_rows0(Cor_rows2),7));

    
    %else
        %dummyset=[nan,nan,nan,nan];
    end
    end
    label_Cor_AP(1,pt,bb)=min(label_Cor_AP_min);
    label_Cor_AP(2,pt,bb)=max(label_Cor_AP_max);
    
    Cor_x_min_label=min(dummyset(:,1));
    Cor_y_min_label=min(dummyset(:,2));
    Cor_x_max_label=max(dummyset(:,3));
    Cor_y_max_label=max(dummyset(:,4));
    
%     Cor_x_min_label=min(Cor_label(:,1));
%     Cor_y_min_label=min(Cor_label(:,2));
%     Cor_x_max_label=max(Cor_label(:,3));
%     Cor_y_max_label=max(Cor_label(:,4));
    
    
    X_min_label=X_left+Cor_x_min_label*X_columns*X_spacing;
    X_max_label=X_left+Cor_x_max_label*X_columns*X_spacing;
    
    Z_max_label=Z_top-Cor_y_min_label*Z_columns*Z_spacing;
    Z_min_label=Z_top-Cor_y_max_label*Z_columns*Z_spacing;
    
%     CenterW_X_label=X_min_label+(X_max_label-X_min_label)/2;
%     CenterW_Y_label=Y_min_label+(Y_max_label-Y_min_label)/2;
%     W_Cor_label=X_max_label-X_min_label;
%     D_Cor_label=Y_max_label-Y_min_label;

    label_set_Cor(:,pt,bb)=squeeze([X_min_label,X_max_label,Z_min_label,Z_max_label]);

    
    end
end

%% 2D sagital bounding boxes
pred_set_Sag=[];label_set_Sag=[];
pred_Sag_RL=[];label_Sag_RL=[];

for pt=1:length(pts) %for each patient
    
    pred_Sag_RL_min=[0,0,0,0];pred_Sag_RL_max=[0,0,0,0];
    label_Sag_RL_min=[0,0,0,0];label_Sag_RL_max=[0,0,0,0];
    
    pt_num=pts(pt);
    %pt_num=536;
    %disp(num2str(pt_num))
    
    Sag_rows0=intersect(find(raw_test(:,2)==pt_num),find(raw_test(:,5)==2));

    %Sag View
    
    for bb=6:7
    count=0;
    count2=0;
    for c=15:10:40
    Sag_pred=[];
    Sag_rows1=find(raw_test(Sag_rows0,c)==bb); %if that view has lth class of labels
    %Sag_rows=intersect(Sag_rows0(Sag_rows1),Sag_rows0(Sag_rows2));
    
    if size(Sag_rows1,1)>0
    count=count+1;
    Sag_pred0=raw_test(Sag_rows0(Sag_rows1),c+1:c+4);
    
    Sag_pred0(Sag_pred0(:,1)==0,:)=[];
    Sag_pred(:,1)=Sag_pred0(:,1)-Sag_pred0(:,3)/2;
    Sag_pred(:,2)=Sag_pred0(:,2)-Sag_pred0(:,4)/2;
    Sag_pred(:,3)=Sag_pred0(:,1)+Sag_pred0(:,3)/2;
    Sag_pred(:,4)=Sag_pred0(:,2)+Sag_pred0(:,4)/2;
    Sag_x_min_pred=min(Sag_pred(:,1));
    Sag_y_min_pred=min(Sag_pred(:,2));
    Sag_x_max_pred=max(Sag_pred(:,3));
    Sag_y_max_pred=max(Sag_pred(:,4));
       
    X_left=raw_test(Sag_rows0(Sag_rows1(1)),7); %X (anteior) physical origin
    X_spacing=raw_test(Sag_rows0(Sag_rows1(1)),9);
    X_columns=raw_test(Sag_rows0(Sag_rows1(1)),13);
    X_min_pred=X_left+Sag_x_min_pred*X_columns*X_spacing;
    X_max_pred=X_left+Sag_x_max_pred*X_columns*X_spacing;
        
    Z_top=raw_test(Sag_rows0(Sag_rows1(1)),8);%Z (superior) physical origin
    Z_spacing=raw_test(Sag_rows0(Sag_rows1(1)),10);
    Z_columns=raw_test(Sag_rows0(Sag_rows1(1)),14);
    Z_max_pred=Z_top-Sag_y_min_pred*Z_columns*Z_spacing;
    Z_min_pred=Z_top-Sag_y_max_pred*Z_columns*Z_spacing;

%     CenterW_X_pred=X_min_pred+(X_max_pred-X_min_pred)/2;
%     CenterW_Y_pred=Y_min_pred+(Y_max_pred-Y_min_pred)/2;
%     W_Sag_pred=X_max_pred-X_min_pred;
%     D_Sag_pred=Y_max_pred-Y_min_pred;
    pred_set_Sag(:,pt,bb)=squeeze([X_min_pred,X_max_pred,Z_min_pred,Z_max_pred]);
    pred_Sag_RL_min(count)=min(raw_test(Sag_rows0(Sag_rows1),6));
    pred_Sag_RL_max(count)=max(raw_test(Sag_rows0(Sag_rows1),6));
    end
    end
    
    pred_Sag_RL(1,pt,bb)=min(pred_Sag_RL_min);
    pred_Sag_RL(2,pt,bb)=max(pred_Sag_RL_max);

    label_c=sort([20,30,40,50,55,60,65],'descend');
    dummyset=[];

    for cc=1:length(label_c)
    c=label_c(cc);
    Sag_rows2=find(raw_test(Sag_rows0,c)==bb);
    Sag_label=[];
    if size(Sag_rows2,1)>0
    count2=count2+1;
    Sag_label0=raw_test(Sag_rows0(Sag_rows2),c+1:c+4);
    Sag_label0(Sag_label0(:,1)==0,:)=[];
    Sag_label(:,1)=Sag_label0(:,1)-Sag_label0(:,3)/2;
    Sag_label(:,2)=Sag_label0(:,2)-Sag_label0(:,4)/2;
    Sag_label(:,3)=Sag_label0(:,1)+Sag_label0(:,3)/2;
    Sag_label(:,4)=Sag_label0(:,2)+Sag_label0(:,4)/2;
    
    dummyset=[dummyset;[Sag_label(:,1),Sag_label(:,2),Sag_label(:,3),Sag_label(:,4)]];
     
%     Sag_x_min_label=min(Sag_label(:,1));
%     Sag_y_min_label=min(Sag_label(:,2));
%     Sag_x_max_label=max(Sag_label(:,3));
%     Sag_y_max_label=max(Sag_label(:,4));
    label_Sag_RL_min(count2)=min(raw_test(Sag_rows0(Sag_rows2),6));
    label_Sag_RL_max(count2)=max(raw_test(Sag_rows0(Sag_rows2),6));
    end 
    end

    label_Sag_RL(1,pt,bb)=min(label_Sag_RL_min);
    label_Sag_RL(2,pt,bb)=max(label_Sag_RL_max);
    
    if isempty(dummyset)==1    
    Sag_x_min_label=nan;
    Sag_y_min_label=nan;
    Sag_x_max_label=nan;
    Sag_y_max_label=nan;
    else
    Sag_x_min_label=min(dummyset(:,1));
    Sag_y_min_label=min(dummyset(:,2));
    Sag_x_max_label=max(dummyset(:,3));
    Sag_y_max_label=max(dummyset(:,4));
    end
    X_min_label=X_left+Sag_x_min_label*X_columns*X_spacing;
    X_max_label=X_left+Sag_x_max_label*X_columns*X_spacing;
    
    Z_max_label=Z_top-Sag_y_min_label*Z_columns*Z_spacing;
    Z_min_label=Z_top-Sag_y_max_label*Z_columns*Z_spacing;
    
%     CenterW_X_label=X_min_label+(X_max_label-X_min_label)/2;
%     CenterW_Y_label=Y_min_label+(Y_max_label-Y_min_label)/2;
%     W_Sag_label=X_max_label-X_min_label;
%     D_Sag_label=Y_max_label-Y_min_label;
    label_set_Sag(:,pt,bb)=squeeze([X_min_label,X_max_label,Z_min_label,Z_max_label]);
    end
end

%%
label_set=label_set_Sag;
label_set(:,:,1:3)=label_set_Ax;
label_set(:,:,4:5)=label_set_Cor(:,:,4:5);

pred_set=pred_set_Sag;
pred_set(:,:,1:3)=pred_set_Ax;
pred_set(:,:,4:5)=pred_set_Cor(:,:,4:5);

%% 3D liver bounding boxes
X_min_label_liver=min(label_set(1,:,1),label_set(1,:,4));
X_max_label_liver=max(label_set(2,:,1),label_set(2,:,4));
Y_min_label_liver=min(label_set(3,:,1),label_set(1,:,6));
Y_max_label_liver=max(label_set(4,:,1),label_set(2,:,6));
Z_min_label_liver=min(label_set(3,:,4),label_set(3,:,6));
Z_max_label_liver=max(label_set(4,:,4),label_set(4,:,6));

LiverBox_label=[X_min_label_liver;X_max_label_liver;Y_min_label_liver;Y_max_label_liver;Z_min_label_liver;Z_max_label_liver];

X_min_pred_liver=min(pred_set(1,:,1),pred_set(1,:,4));
X_max_pred_liver=max(pred_set(2,:,1),pred_set(2,:,4));
Y_min_pred_liver=min(pred_set(3,:,1),pred_set(1,:,6));
Y_max_pred_liver=max(pred_set(4,:,1),pred_set(2,:,6));
Z_min_pred_liver=min(pred_set(3,:,4),pred_set(3,:,6));
Z_max_pred_liver=max(pred_set(4,:,4),pred_set(4,:,6));

LiverBox_pred=[X_min_pred_liver;X_max_pred_liver;Y_min_pred_liver;Y_max_pred_liver;Z_min_pred_liver;Z_max_pred_liver];

%% appearrance check
Range_set_label=label_Sag_RL;
Range_set_label(:,:,1:3)=label_Ax_SI;
Range_set_label(:,:,4:5)=label_Cor_AP(:,:,4:5);

Range_set_pred=pred_Sag_RL;
Range_set_pred(:,:,1:3)=pred_Ax_SI;
Range_set_pred(:,:,4:5)=pred_Cor_AP(:,:,4:5);
%% 3D Liver bounding boxes after appearrance check
SI_min_label=min(min(min(X_min_label_liver,Range_set_label(1,:,1)),Range_set_label(1,:,2)),Range_set_label(1,:,3))
SI_max_label=max(max(max(X_max_label_liver,Range_set_label(2,:,1)),Range_set_label(2,:,2)),Range_set_label(2,:,3))
AP_min_label=min(min(Y_min_label_liver,Range_set_label(1,:,4)),Range_set_label(1,:,5))
AP_max_label=max(max(Y_max_label_liver,Range_set_label(2,:,4)),Range_set_label(2,:,5))
RL_min_label=min(min(Z_min_label_liver,Range_set_label(1,:,6)),Range_set_label(1,:,7))
RL_max_label=max(max(Z_max_label_liver,Range_set_label(2,:,6)),Range_set_label(2,:,7))

LiverBox_label_all=[SI_min_label;SI_max_label;AP_min_label;AP_max_label;RL_min_label;RL_max_label];

SI_min_pred=min(min(min(X_min_pred_liver,Range_set_pred(1,:,1)),Range_set_pred(1,:,2)),Range_set_pred(1,:,3))
SI_max_pred=max(max(max(X_max_pred_liver,Range_set_pred(2,:,1)),Range_set_pred(2,:,2)),Range_set_pred(2,:,3))
AP_min_pred=min(min(Y_min_pred_liver,Range_set_pred(1,:,4)),Range_set_pred(1,:,5))
AP_max_pred=max(max(Y_max_pred_liver,Range_set_pred(2,:,4)),Range_set_pred(2,:,5))
RL_min_pred=min(min(Z_min_pred_liver,Range_set_pred(1,:,6)),Range_set_pred(1,:,7))
RL_max_pred=max(max(Z_max_pred_liver,Range_set_pred(2,:,6)),Range_set_pred(2,:,7))

LiverBox_pred_all=[SI_min_pred;SI_max_pred;AP_min_pred;AP_max_pred;RL_min_pred;RL_max_pred];

%% 3D Body bounding boxes
X_min_label_body=min(label_set(1,:,2),label_set(1,:,5));
X_max_label_body=max(label_set(2,:,2),label_set(2,:,5));
Y_min_label_body=min(label_set(3,:,2),label_set(1,:,7));
Y_max_label_body=max(label_set(4,:,2),label_set(2,:,7));
Z_min_label_body=min(label_set(3,:,5),label_set(3,:,7));
Z_max_label_body=max(label_set(4,:,5),label_set(4,:,7));

BodyBox_label=[X_min_label_body;X_max_label_body;Y_min_label_body;Y_max_label_body;Z_min_label_body;Z_max_label_body];

X_min_pred_body=min(pred_set(1,:,2),pred_set(1,:,5));
X_max_pred_body=max(pred_set(2,:,2),pred_set(2,:,5));
Y_min_pred_body=min(pred_set(3,:,2),pred_set(1,:,7));
Y_max_pred_body=max(pred_set(4,:,2),pred_set(2,:,7));
Z_min_pred_body=min(pred_set(3,:,5),pred_set(3,:,7));
Z_max_pred_body=max(pred_set(4,:,5),pred_set(4,:,7));

BodyBox_pred=[X_min_pred_body;X_max_pred_body;Y_min_pred_body;Y_max_pred_body;Z_min_pred_body;Z_max_pred_body];

% appearrance check
Range_set_label=label_Sag_RL;
Range_set_label(:,:,1:3)=label_Ax_SI;
Range_set_label(:,:,4:5)=label_Cor_AP(:,:,4:5);

Range_set_pred=pred_Sag_RL;
Range_set_pred(:,:,1:3)=pred_Ax_SI;
Range_set_pred(:,:,4:5)=pred_Cor_AP(:,:,4:5);
%
SI_min_label=min(min(min(X_min_label_body,Range_set_label(1,:,1)),Range_set_label(1,:,2)),Range_set_label(1,:,3))
SI_max_label=max(max(max(X_max_label_body,Range_set_label(2,:,1)),Range_set_label(2,:,2)),Range_set_label(2,:,3))
AP_min_label=min(min(Y_min_label_body,Range_set_label(1,:,4)),Range_set_label(1,:,5))
AP_max_label=max(max(Y_max_label_body,Range_set_label(2,:,4)),Range_set_label(2,:,5))
RL_min_label=min(min(Z_min_label_body,Range_set_label(1,:,6)),Range_set_label(1,:,7))
RL_max_label=max(max(Z_max_label_body,Range_set_label(2,:,6)),Range_set_label(2,:,7))

BodyBox_label_all=[SI_min_label;SI_max_label;AP_min_label;AP_max_label;RL_min_label;RL_max_label];

SI_min_pred=min(min(min(X_min_pred_body,Range_set_pred(1,:,1)),Range_set_pred(1,:,2)),Range_set_pred(1,:,3))
SI_max_pred=max(max(max(X_max_pred_body,Range_set_pred(2,:,1)),Range_set_pred(2,:,2)),Range_set_pred(2,:,3))
AP_min_pred=min(min(Y_min_pred_body,Range_set_pred(1,:,4)),Range_set_pred(1,:,5))
AP_max_pred=max(max(Y_max_pred_body,Range_set_pred(2,:,4)),Range_set_pred(2,:,5))
RL_min_pred=min(min(Z_min_pred_body,Range_set_pred(1,:,6)),Range_set_pred(1,:,7))
RL_max_pred=max(max(Z_max_pred_body,Range_set_pred(2,:,6)),Range_set_pred(2,:,7))

BodyBox_pred_all=[SI_min_pred;SI_max_pred;AP_min_pred;AP_max_pred;RL_min_pred;RL_max_pred];

%% 3D boxes for Axial, Coronal, and Sagittal prescriptions
SagBox_pred_all=[LiverBox_pred_all(1:2,:); BodyBox_pred_all(3:6,:)];
SagBox_label_all=[LiverBox_label_all(1:2,:); BodyBox_label_all(3:6,:)];
CorBox_pred_all=[BodyBox_pred_all(1:2,:); LiverBox_pred_all(3:4,:); BodyBox_pred_all(5:6,:)];
CorBox_label_all=[BodyBox_label_all(1:2,:); LiverBox_label_all(3:4,:); BodyBox_label_all(5:6,:)];
AxBox_pred_all=[BodyBox_pred_all(1:4,:); LiverBox_pred_all(5:6,:)];
AxBox_label_all=[BodyBox_label_all(1:4,:); LiverBox_label_all(5:6,:)];

%% Shifts in 3D Liver, Axial, Coronal, and Sagittal prescriptions
shiftLiver=LiverBox_pred_all-LiverBox_label_all;
shiftAx=AxBox_pred_all-AxBox_label_all;
shiftCor=CorBox_pred_all-CorBox_label_all;
shiftSag=SagBox_pred_all-SagBox_label_all;

shiftLiver1=shiftLiver;
shiftLiver1(2,:)=-shiftLiver(2,:);
shiftLiver1(4,:)=-shiftLiver(4,:);
shiftLiver1(6,:)=-shiftLiver(6,:);

shiftAx1=shiftAx;
shiftAx1(2,:)=-shiftAx(2,:);
shiftAx1(4,:)=-shiftAx(4,:);
shiftAx1(6,:)=-shiftAx(6,:);

shiftCor1=shiftCor;
shiftCor1(2,:)=-shiftCor(2,:);
shiftCor1(4,:)=-shiftCor(4,:);
shiftCor1(6,:)=-shiftCor(6,:);

shiftSag1=shiftSag;
shiftSag1(2,:)=-shiftSag(2,:);
shiftSag1(4,:)=-shiftSag(4,:);
shiftSag1(6,:)=-shiftSag(6,:);
%'Inferier','Superior','Posterior','Anterior','Left','Right'

figure; plot([1:6],shiftLiver1','*k');set(gca,'XTick',[1:6]);xlabel('Liver Box Edge'); ylabel('Offset: Auto - Manual (mm)')
figure; plot([1:6],shiftAx1','*k');set(gca,'XTick',[1:6]);xlabel('Axial Box Edge'); ylabel('Offset: Auto - Manual (mm)')
figure; plot([1:6],shiftCor1','*k');set(gca,'XTick',[1:6]);xlabel('Coronal Box Edge'); ylabel('Offset: Auto - Manual (mm)')
figure; plot([1:6],shiftSag1','*k');set(gca,'XTick',[1:6]);xlabel('Sagittal Box Edge'); ylabel('Offset: Auto - Manual (mm)')