%% Data visualization script to display results and analyze AI performance across subpopulations

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
%% Histograms 

for l=1:length(LabelList)
 disp([cell2mat(LabelList(l)) ' - Number of mislabel: ' num2str(set_wrong_bbox(l))])
 figure;
 histogram(allIoU(find(allIoU(l,:)>0)),'Normalization','count') %different class = 71
 title([cell2mat(LabelList(l)) ' IoU Histogram'])
end

% over all
disp(['Overall - Number of mislabel: ' num2str(sum(set_wrong_bbox(:)))])
figure;

figure;
histogram(allIoU(find(allIoU(:)>0)),0.05:0.05:1,'Normalization','probability')
median(allIoU(find(allIoU(l,:)>0)))
iqr(allIoU(find(allIoU(l,:)>0)))

title(['Overall IoU Histogram'])
axis([0.15 1 0 0.35])
set(gca,'fontname','Arial','FontSize',20,'YTick',[0:0.1:0.3], 'XTick', [0:0.2:1])
%% IoU stats
mean(allIoU(find(allIoU(:)>0)))
std(allIoU(find(allIoU(:)>0)))
median(allIoU(find(allIoU(:)>0)))
min(allIoU(find(allIoU(:)>0)))
max(allIoU(find(allIoU(:)>0)))
iqr(allIoU(find(allIoU(:)>0)))
%% Boxplots
close all
figure; h=boxplot(shiftLiver1');set(h,'LineWidth',3,'MarkerSize',10,'Color','k');axis([0.5 6.5 -100 100]);set(gca,'XTick',[1:6],'fontname','Arial','FontSize',20);xlabel('Liver Box Edge'); ylabel('Offset: Auto - Manual (mm)')
figure; h=boxplot(shiftAx1');set(h,'LineWidth',3,'MarkerSize',10,'Color','k');axis([.5 6.5 -100 100]);set(gca,'XTick',[1:6],'fontname','Arial','FontSize',20);xlabel('Axial Box Edge'); ylabel('Offset: Auto - Manual (mm)')
figure; h=boxplot(shiftCor1');set(h,'LineWidth',3,'MarkerSize',10,'Color','k');axis([.5 6.5 -100 100]);set(gca,'XTick',[1:6],'fontname','Arial','FontSize',20);xlabel('Coronal Box Edge'); ylabel('Offset: Auto - Manual (mm)')
figure; h=boxplot(shiftSag1');set(h,'LineWidth',3,'MarkerSize',10,'Color','k');axis([.5 6.5 -100 100]);set(gca,'XTick',[1:6],'fontname','Arial','FontSize',20);xlabel('Sagittal Box Edge'); ylabel('Offset: Auto - Manual (mm)')

%% find % AI volumes that covers at least the liver
count_p=0;
for p=1:45
    count_d=0;
    for d=5:6
        if abs(shiftAx1(d,p)) <= 9
            count_d=count_d+1
        end
    end
    
    if count_d==2
        count_p=count_p+1;
    end
end

%% Plot AI and manual labeling of a patient's localizer images in .gif and .png
pt_num=999;
data0=raw_test(:,:);
data=sortrows(data0,3);
pt_index=1;
for img=1:size(data,1)
    if data(img,5)==1 %Ax
        filename=[num2str(pt_num) 'Ax.gif'];
        h=figure;
        axis tight manual 
        axis off;
        imshow(['/AutoprescriptionLiver/' sprintf('%03.3d',pt_num) '/IM-0001-' sprintf('%04.4d',data(img,3)) '.dcm.png'])
        if data(img,8) > BodyBox_label_all(5,pt_index) && data(img,8) < BodyBox_label_all(6,pt_index)
            hold on
            dispboxAx=[(BodyBox_label_all(1,pt_index)-data(img,6))/data(img,9)*512/data(img,13),...             
                ((BodyBox_label_all(3,pt_index)-data(img,7))/data(img,10))*512/data(img,14),...
                (BodyBox_label_all(2,pt_index)-BodyBox_label_all(1,pt_index))/data(img,9)*512/data(img,13),...
                (BodyBox_label_all(4,pt_index)-BodyBox_label_all(3,pt_index))/data(img,10)*512/data(img,14)];
            %disp(dispboxAx);
            rectangle('Position',dispboxAx,'EdgeColor','c','LineWidth',2)
        end
        
        if data(img,8) > BodyBox_pred_all(5,pt_index) && data(img,8) < BodyBox_pred_all(6,pt_index)
            hold on %512 vs 256
            dispboxAxPred=[(BodyBox_pred_all(1,pt_index)-data(img,6))/data(img,9)*512/data(img,13),...             
                ((BodyBox_pred_all(3,pt_index)-data(img,7))/data(img,10))*512/data(img,14),...
                (BodyBox_pred_all(2,pt_index)-BodyBox_pred_all(1,pt_index))/data(img,9)*512/data(img,13),...
                (BodyBox_pred_all(4,pt_index)-BodyBox_pred_all(3,pt_index))/data(img,10)*512/data(img,14)];
            disp(dispboxAxPred);
            rectangle('Position',dispboxAxPred,'EdgeColor','y','LineWidth',2)
            %pause;
        end
        Image = getframe(gcf);
 
% Uncomment this section if outputting .png files
%         set(gca, 'unit', 'normalize');
%         set(gca, 'position', [0 0 1 1]);
%         print(gcf,[num2str(pt_num) '_Ax_' num2str(raw_test(e,3)) '.png'], '-dpng', '-r300');
        %imwrite(gca, [num2str(pt_num) '_Ax_' num2str(raw_test(e,3)) '.png']);
            im = frame2im(Image); 
            [imind,cm] = rgb2ind(im,256);
            img
              if img == 1 
                  imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
              else 
                  imwrite(imind,cm,filename,'gif','WriteMode','append'); 
              end
    end
    hold off
    
    if data(img,5)==3 %Cor
        filename=[num2str(pt_num) 'Cor.gif'];
        h=figure;
        axis tight manual 
        axis off;
        imshow(['/AutoprescriptionLiver/' sprintf('%03.3d',pt_num) '/IM-0001-' sprintf('%04.4d',data(img,3)) '.dcm.png'])
        if data(img,7) > BodyBox_label_all(3,pt_index) && data(img,7) < BodyBox_label_all(4,pt_index)
            hold on
            dispboxCor=[(BodyBox_label_all(1,pt_index)-data(img,6))/data(img,9)*512/data(img,13),...             
                -((BodyBox_label_all(6,pt_index)-data(img,8))/data(img,10))*512/data(img,14),...
                (BodyBox_label_all(2,pt_index)-BodyBox_label_all(1,pt_index))/data(img,9)*512/data(img,13),...
                (BodyBox_label_all(6,pt_index)-BodyBox_label_all(5,pt_index))/data(img,10)*512/data(img,14)];
            %disp(dispboxCor);
            rectangle('Position',dispboxCor,'EdgeColor','c','LineWidth',2)
        end
        
        if data(img,7) > BodyBox_pred_all(3,pt_index) && data(img,7) < BodyBox_pred_all(4,pt_index)
            hold on
            dispboxCorPred=[(BodyBox_pred_all(1,pt_index)-data(img,6))/data(img,9)*512/data(img,13),...             
                -((BodyBox_pred_all(6,pt_index)-data(img,8))/data(img,10))*512/data(img,14),...
                (BodyBox_pred_all(2,pt_index)-BodyBox_pred_all(1,pt_index))/data(img,9)*512/data(img,13),...
                (BodyBox_pred_all(6,pt_index)-BodyBox_pred_all(5,pt_index))/data(img,10)*512/data(img,14)];
            disp(dispboxCorPred);
            rectangle('Position',dispboxCorPred,'EdgeColor','y','LineWidth',2)
            %pause;
        end        
        Image = getframe(gcf);
        im = frame2im(Image); 
        [imind,cm] = rgb2ind(im,256);
        img
          if img == 6 
              imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
          else 
              imwrite(imind,cm,filename,'gif','WriteMode','append'); 
          end
    end
    hold off
    
    
    if data(img,5)==2
        filename=[num2str(pt_num) 'Sag.gif'];
        h=figure;
        axis tight manual 
        axis off;
        imshow(['/AutoprescriptionLiver/' sprintf('%03.3d',pt_num) '/IM-0001-' sprintf('%04.4d',data(img,3)) '.dcm.png'])
        if data(img,6) > BodyBox_label_all(1,pt_index) && data(img,6) < BodyBox_label_all(2,pt_index)
            hold on %512 vs 256
            dispboxSag=[...             
                ((BodyBox_label_all(3,pt_index)-data(img,7))/data(img,9))*512/data(img,13),...
                -(BodyBox_label_all(6,pt_index)-data(img,8))/data(img,10)*512/data(img,14),...        
                (BodyBox_label_all(4,pt_index)-BodyBox_label_all(3,pt_index))/data(img,9)*512/data(img,13),...
                (BodyBox_label_all(6,pt_index)-BodyBox_label_all(5,pt_index))/data(img,10)*512/data(img,14)];
            %disp(dispboxSag);
            rectangle('Position',dispboxSag,'EdgeColor','c','LineWidth',2)
        end
        
        if data(img,6) > BodyBox_pred_all(1,pt_index) && data(img,6) < BodyBox_pred_all(2,pt_index)
            hold on %512 vs 256
            dispboxSagPred=[...
                ((BodyBox_pred_all(3,pt_index)-data(img,7))/data(img,9))*512/data(img,13),...
                -(BodyBox_pred_all(6,pt_index)-data(img,8))/data(img,10)*512/data(img,14),...        
                (BodyBox_pred_all(4,pt_index)-BodyBox_pred_all(3,pt_index))/data(img,9)*512/data(img,13),...
                (BodyBox_pred_all(6,pt_index)-BodyBox_pred_all(5,pt_index))/data(img,10)*512/data(img,14)];
            disp(dispboxSagPred);
            rectangle('Position',dispboxSagPred,'EdgeColor','y','LineWidth',2)
            %pause;
        end
        Image = getframe(gcf);
        im = frame2im(Image); 
        [imind,cm] = rgb2ind(im,256);
        img
          if img == 11
              imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
          else 
              imwrite(imind,cm,filename,'gif','WriteMode','append'); 
          end
    end
    hold off
end

%% Analyze patient population
run3testList={};
for p=1:length(listTest)
    for t=1:length(PTwithBMI)
    if cell2mat(PTwithBMI(t,7))==listTest(p)
    run3testList(p,:)=PTwithBMI(t,:);
    end
    end
end


%% Patient info summary
age_avg=mean(cell2mat(run3testList(:,3)))
age_max=max(cell2mat(run3testList(:,3)))
age_min=min(cell2mat(run3testList(:,3)))
sex_F=length(find(cell2mat(run3testList(:,4))=='F'))
sex_M=45-sex_F
BMI_avg=mean(cell2mat(run3testList(:,5)))
BMI_max=max(cell2mat(run3testList(:,5)))
BMI_min=min(cell2mat(run3testList(:,5)))
cirrhosis_Y=0;
for i=1:45
    if strcmp(cell2mat(run3testList(i,6)),'Yes')
        cirrhosis_Y=cirrhosis_Y+1;
    end
end

%% Patient pathology
Excelindex=[];
for g=1:length(run3trainList)
    for t=1:height(PatientInfo)
    if strcmp(cell2mat(run3trainList(g,2)), cell2mat(table2cell(PatientInfo(t,6))))
        run3trainList(g,8)=table2cell(PatientInfo(t,28));
        Excelindex(g)=t+1;
    end
    end
end

%
resection_count = length(find(contains(run3testList(:,8),'resection','IgnoreCase',true))) + length(find(contains(run3testList(:,8),'hepatectomy','IgnoreCase',true)))
atrophy_count = length(find(contains(run3testList(:,8),'atrophy','IgnoreCase',true))) + length(find(contains(run3testList(:,8),'hyperplasia','IgnoreCase',true)))
ascites_count = length(find(contains(run3testList(:,8),'ascites','IgnoreCase',true)))
iron_count = length(find(contains(run3testList(:,8),'iron','IgnoreCase',true)))
fatty_count = length(find(contains(run3testList(:,8),'fatty liver','IgnoreCase',true)))
steatosis_count = length(find(contains(run3testList(:,8),'steatosis','IgnoreCase',true)))
infiltration_count = length(find(contains(run3testList(:,8),'fatty infiltration','IgnoreCase',true)))
polycyst_count = length(find(contains(run3testList(:,8),'polycystic','IgnoreCase',true)))
hepatomegaly_count = length(find(contains(run3testList(:,8),'hepatomegaly','IgnoreCase',true)))
pleural_count=length(find(contains(run3testList(:,8),'pleural effusion','IgnoreCase',true)))+length(find(contains(run3testList(:,8),'riedel lobe','IgnoreCase',true)))
tumor_count = length(find(contains(run3testList(:,8),'tumor','IgnoreCase',true)))
mets_count = length(find(contains(run3testList(:,8),'mets','IgnoreCase',true)))+length(find(contains(run3testList(:,8),'metastasis','IgnoreCase',true)))+length(find(contains(run3testList(:,8),'metastases','IgnoreCase',true)))+length(find(contains(run3testList(:,8),'metastatic','IgnoreCase',true)))
lesion_count = length(find(contains(run3testList(:,8),'lesion','IgnoreCase',true)))



