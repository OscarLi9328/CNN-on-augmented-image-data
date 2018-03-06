%%%% augmenting the images %%%%


clc
clear 

dataDir= './data/wallpapers/';
outputDir = './data/wallpapers/train_aug';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

train_folder = 'train';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

ncat = categories(input_all.Labels);
for i=1:length(ncat)
    cat_dir = strcat(output_folder,'/',char(ncat(i)));
    if ~exist(cat_dir,'dir')
    mkdir(cat_dir);
    end
end

catagories = unique(train_all.Labels);

% augmentation of the images 5 times
k = 1;
a = 1; 
aa = 1;
kk = 1;
kkk = 1;
fprintf('Start augmented...'); t = tic;

rot_angle = zeros(length(train_all.Files), 5);
scale = zeros(length(train_all.Files), 5);
translation = cell(length(train_all.Files),5);

for i = 1:size(train_all.Files)

    % 1. augemented 1
    temp= imread(char(train_all.Files(i))); 
    [im_1, rot_ang, scale_ratio, tran_out] = augmentImage(temp);
    rot_angle(i, 1) = rot_ang;
    scale(i, 1) = scale_ratio;
    translation{i,1} = tran_out;
    
    for class = 1:length(catagories)
        
        if train_all.Labels(i) == catagories(class)
            folder = [dataDir,'train_aug/', char(train_all.Labels(i))];
            baseFileName = sprintf('%s_%d_1.jpg', char(train_all.Labels(i)), k); 
            fullFileName = fullfile(folder, baseFileName);
            imwrite(im_1, fullFileName);
            
            if (i>2) && (train_all.Labels(i) ~= train_all.Labels(i-1))
                k = 1;
            else
                k = k + 1;
            end
            
        end
    end
    
  
    % 2. augemented 2
    [im_2, rot_ang, scale_ratio, tran_out] = augmentImage(temp);
    rot_angle(i, 2) = rot_ang;
    scale(i, 2) = scale_ratio;
    translation{i, 2} = tran_out;
    
    
    for class = 1:length(catagories)
        
        if train_all.Labels(i) == catagories(class)
            folder = [dataDir,'train_aug/', char(train_all.Labels(i))];
            baseFileName = sprintf('%s_%d_2.jpg', char(train_all.Labels(i)), a); 
            fullFileName = fullfile(folder, baseFileName);
            imwrite(im_2, fullFileName);
            
            if (i>2) && (train_all.Labels(i) ~= train_all.Labels(i-1))
                a = 1;
            else
                a = a + 1;
            end
            
        end
    end
    
    % 3. augmented 3
    [im_3, rot_ang, scale_ratio, tran_out] = augmentImage(temp);
    rot_angle(i, 3) = rot_ang;
    scale(i, 3) = scale_ratio;
    translation{i, 3} = tran_out;
    
    
    for class = 1:length(catagories)
        
        if train_all.Labels(i) == catagories(class)
            folder = [dataDir,'train_aug/', char(train_all.Labels(i))];
            baseFileName = sprintf('%s_%d_3.jpg', char(train_all.Labels(i)), kk); 
            fullFileName = fullfile(folder, baseFileName);
            imwrite(im_3, fullFileName);
            
            if (i>2) && (train_all.Labels(i) ~= train_all.Labels(i-1))
                kk = 1;
            else
                kk = kk + 1;
            end
            
        end
    end
    
    % 4. augmented 4
    [im_4, rot_ang, scale_ratio, tran_out] = augmentImage(temp);
    rot_angle(i, 4) = rot_ang;
    scale(i, 4) = scale_ratio;
    translation{i, 4} = tran_out;
    
    
    for class = 1:length(catagories)
        
        if train_all.Labels(i) == catagories(class)
            folder = [dataDir,'train_aug/', char(train_all.Labels(i))];
            baseFileName = sprintf('%s_%d_4.jpg', char(train_all.Labels(i)), kkk); 
            fullFileName = fullfile(folder, baseFileName);
            imwrite(im_4, fullFileName);
            
            if (i>2) && (train_all.Labels(i) ~= train_all.Labels(i-1))
                kkk = 1;
            else
                kkk = kkk + 1;
            end
            
        end
    end
    
    % 5. augmented 5
    [im_5, rot_ang, scale_ratio, tran_out] = augmentImage(temp);
    rot_angle(i, 5) = rot_ang;
    scale(i, 5) = scale_ratio;
    translation{i, 5} = tran_out;
    
     
    for class = 1:length(catagories)
        
        if train_all.Labels(i) == catagories(class)
            folder = [dataDir,'train_aug/', char(train_all.Labels(i))];
            baseFileName = sprintf('%s_%d_5.jpg', char(train_all.Labels(i)), aa); 
            fullFileName = fullfile(folder, baseFileName);
            imwrite(im_5, fullFileName);
            
            if (i>2) && (train_all.Labels(i) ~= train_all.Labels(i-1))
                aa = 1;
            else
                aa = aa+1;
            end
            
        end
    end
    
end
fprintf('Done in %.02f secs\n', toc(t)); 
save('aug.mat');

