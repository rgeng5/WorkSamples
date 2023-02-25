%% Data extraction script to obtain meta data from DICOM headers

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
%%
DICOMfolder='/home/.../all_DICOMs/';
SAVEfolder='/home/.../saved/';

load list.mat %lists all the patients for this training

%% split views
pts=list(1:100);
count=0;
for p=1:length(pts)
    disp(num2str(p/length(pts)))
    files=dir([DICOMfolder sprintf('%04.4d',pts(p))]);
    for f=1:length(files)-2
    count=count+1;
    raw_test(count,1)=count;
    raw_test(count,2)=pts(p);    
    info=dicominfo([DICOMfolder sprintf('%04.4d',pts(p)) '/' files(f+2).name]);
    raw_test(count,3)=info.InstanceNumber;
    raw_test(count,4)=f;
    if info.ImageOrientationPatient==[1;0;0;0;1;0]
        orientation=1; %Axial
    else
        if info.ImageOrientationPatient==[0;1;0;0;0;-1]
            orientation=2; %Sag
        else
            orientation=3; %Cor
        end
    end
    raw_test(count,5)=orientation;
    raw_test(count,6)=info.ImagePositionPatient(1);
    raw_test(count,7)=info.ImagePositionPatient(2);
    raw_test(count,8)=info.ImagePositionPatient(3);
    raw_test(count,9)=info.PixelSpacing(1);
    raw_test(count,10)=info.PixelSpacing(2);
    raw_test(count,11)=info.SliceThickness;
    raw_test(count,12)=info.SpacingBetweenSlices;
    raw_test(count,13)=info.Rows;
    raw_test(count,14)=info.Columns;
    end
end

%save([SAVEfolder 'Run4TestInfo.mat'],'raw_test','-v7.3')
