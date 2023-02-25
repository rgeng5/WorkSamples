%% Pre-processing script to increase contrast around the liver and convert DICOMs to PNGs, slice by slice

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
main_folder='/main_folder/';
PNGfolder='/PNGfolder/';
pt_list=dir(main_folder);
for pt=1
    folder=[main_folder pt_list(3+pt).name];
    %mkdir([PNGfolder num2str(pt)])
    subfolder_list=dir(folder);
    folder=[folder '/' subfolder_list(end).name];
    
    subfolder_list=dir(folder);
    folder=[folder '/' subfolder_list(end).name];
    
    files=dir(folder);
    for i=1:length(files)-2
        I =double(dicomread([folder '/' files(i+2).name]));
        
        % Saturate high image intensities to increase contrast around the liver
        img=mat2gray(imadjust(mat2gray(I),[0 0.2]));
 
        %figure,imshow(img,[]);
        mkdir([PNGfolder num2str(pt)])
        imwrite(img,[PNGfolder num2str(pt) '/' files(i+2).name '.png']);
    end

end
close all