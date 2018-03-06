clc
clear 

%%%% loading the datasets
dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

train_folder = 'train';
test_folder  = 'test';
% uncomment after you create the augmentation dataset
train_folder = 'train_aug';
test_folder  = 'test_aug';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%%
rng('default');
numEpochs = 10; % 10 for both learning rates
batchSize = 250;
nTraining = length(train.Labels);

% Define the Network Structure, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 
layers = [
    imageInputLayer([128 128 1]); % Input to the network is a 128x128x1 sized image 
    convolution2dLayer(5,20,'Padding',[2 2],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(25); % Fullly connected layer with 50 activations
    dropoutLayer(.4); % Dropout layer
    fullyConnectedLayer(17); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ]

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',5e-4,...% learning rate
    'CheckpointPath', checkpointDir,... % save the checkpoint
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    % 'OutputFcn',@plotTrainingAccuracy,...

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

% Test on the training data
YTest = classify(net1, train);
train_acc_1 = mean(YTest==train.Labels)
train_class_1 = confusionmat(YTest, train.Labels);

% Test on the validation data
YTest = classify(net1,val);
val_acc_1 = mean(YTest==val.Labels)
val_class_1 = confusionmat(YTest, val.Labels);

% Test on the Testing data
YTest = classify(net1,test);
test_acc_1 = mean(YTest==test.Labels)
test_class_1 = confusionmat(YTest, test.Labels);

figure()
plotTrainingAccuracy_All(info1,numEpochs);

% It seems like it isn't converging after looking at the graph but lets
%   try dropping the learning rate to show you how.  

options = trainingOptions('sgdm','MaxEpochs',20,...
    'InitialLearnRate',1e-4,... % learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs,...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
%     'OutputFcn',@plotTrainingAccuracy,...

 t = tic;
[net2,info2] = trainNetwork(train,net1.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

% Test on the training data
YTest = classify(net2, train);
train_acc_2 = mean(YTest==train.Labels)
train_class_2 = confusionmat(YTest, train.Labels);

% Test on the validation data
YTest = classify(net2,val);
val_acc_2 = mean(YTest==val.Labels)
val_class_2 = confusionmat(YTest, val.Labels);


% Test on the Testing data
YTest = classify(net2,test);
test_acc_2 = mean(YTest==test.Labels)
test_class_2 = confusionmat(YTest, test.Labels);

figure()
plotTrainingAccuracy_All(info2,numEpochs);

figure()
plotConfMat(val_class_1);
figure()
plotConfMat(val_class_2);
figure()
plotConfMat(train_class_1);
figure()
plotConfMat(train_class_2);
figure()
plotConfMat(test_class_1);
figure()
plotConfMat(test_class_2);


