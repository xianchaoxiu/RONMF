 % demo method with all datasets for acc, NMI, purity, F-score

clc;
clear;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',  1));
addpath(genpath(pwd))

% root_path = 'D:/matlab code/dim_1225_RONMF/codes/code_and_datasets/Saved_Results/compare_results/methods/';

% choose a dataset
% {'AR','COIL20','COIL100','MNIST','UMIST','USPS','Yale','YaleB'}
% dataset = 'COIL20'; % colorful obj
% dataset = 'COIL100' ; %obj
% dataset = 'USPS';    %handwriten num
% dataset = 'AR';     %face
% dataset = 'UMIST';  %face
% dataset = 'MNIST';
dataset = 'Yale'; 
% dataset = 'YaleB';
% dataset = 'Yale_n';
% dataset = 'or10P'; %face
% dataset = 'CAL';


switch dataset
    case 'MNIST'  % 70000 sample  784 feature_dim 
        load('MNIST.mat');  % fea 70000 x 10  gnd  70000 x 1  10classes 1class about 7000samples
        % fea = fea';
        fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach  = 500;               % number of sample in every class is 500

    case 'AR'     % 700   sample  19800 feature_dim 
        load('ar_98.mat');  % Tr_dataMatrix 19800 x 700  gnd  1 x 700 100classes 1class about 7samples
        fea = fea';
        fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        % gnd = Tr_sampleLabels';  % 700 x 1  
        % fea = Tr_dataMatrix';     % 700 x  19800
        nEach = 7; 

    case 'UMIST'  % 575   sample  644 feature_dim
        load('umist.mat');  % X    644 x 575   gnd    575 x 1   20classes 1class about 25 samples 
        fea = X';
        % fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach    = 19;

    case 'USPS'   % 9298  sample  256 feature_dim
        % load('USPS.mat');  % fea 9298 x 256  gnd  9298 x 1   10classes 1class about 1000 samples銆?
        load('usps_g_70.mat');  % fea 9298 x 256  gnd  9298 x 1   10classes 1class about 1000 samples銆?
        fea = Normalize255(fea); % -1~ 1 --> 0~255
        nEach    = 708;

    case 'USPS_n'   % 9298  sample  256 feature_dim
        load('USPS_n.mat');  % fea 9298 x 256  gnd  9298 x 1   10classes 1class about 1000 samples銆?
        % fea = Normalize255(fea); % -1~ 1 --> 0~255
        nEach    = 708;    

    case 'COIL20' % 1440   sample   1024  feature_dim
        load('COIL20.mat');  % fea 1440 x 1024  gnd  1440 x 1   20classes 1class 72 samples
        % load('coil20_g_70.mat');  % fea 1440 x 1024  gnd  1440 x 1   20classes 1class 72 samples
        % fea = fea';
        fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach    = 72;

    case 'COIL100' % 7200   sample   1024  feature_dim
        load('COIL100.mat');  % fea 7200 x 1024  gnd  7200 x 1   100classes 1class 72 samples
        % fea = fea';
        % fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach    = 72;

    case 'Yale' % 165   sample   1024  feature_dim
        % load('yale_98.mat');  % fea 165 x 1024  gnd  165 x 1   15classes 1class 15 samples
        % load('Yale_32x32.mat');  % fea 165 x 1024  gnd  165 x 1   15classes 1class 15 samples
        load('Yale_g_10.mat');  % fea 165 x 1024  gnd  165 x 1   15classes 1class 15 samples
        % fea = fea';
        % fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach    = 11; 

    case 'Yale_n' % 165   sample   1024  feature_dim
        load('Yale_n.mat');  % fea 165 x 1024  gnd  165 x 1   15classes 1class 15 samples
        % fea = fea';
        % % fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach    = 11;     

    case 'YaleB' % 2414   sample   1024  feature_dim
        load('YaleB_32x32.mat');  % fea 2414 x 1024  gnd  2414 x 1   38classes 1class about 64 samples
        % fea = fea';
        % fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        nEach    = 59;

    case 'or10P' % 100   sample   1024  feature_dim
        load('or10P.mat');  % fea 100 x 10304  gnd  100 x 1   10classes 1class about  samples
        fea = X;
        gnd = Y;
        % fea = Normalize255(fea)/255; % -1~ 1 --> 0~255
        %         % 加载 CAL 数据集后执行
        % class_counts = histcounts(gnd, length(unique(gnd)));
        % min_samples = min(class_counts);
        % fprintf('最小样本数: %d\n', min_samples);
        nEach    = 10;  

   case 'CAL' % 9145   sample   1024  feature_dim
        load('Caltech-101.mat');  % fea 9145 x 1024  gnd  9145 x 1   102classes 1class about  samples
        % fea = X';
        fea = Normalize255(double(fea))/255; % -1~ 1 --> 0~255
        nEach    = 31;         
end

nClass  = length(unique(gnd));   %nClass

%normalization the feature
if strcmp(dataset,'COIL100') || strcmp(dataset,'AR')  || strcmp(dataset,'or10P') || strcmp(dataset,'Caltech-101')
    fea = NormalizeFea(double(fea));
else
    fea = NormalizeFea(fea);
end


nDim    = size(fea,2);
%--------------------------------------------------------------------------
Cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10];
% Cluster = [ 6 ];
percent = 0.3;
alpha = 1;
nCase    = length(Cluster);  % 9
nRun     = 10; 

%--------------------------------------------------------------------------
acc_rnmf = zeros(nCase,nRun);

nmi_rnmf = zeros(nCase,nRun);

pur_rnmf = zeros(nCase,nRun);

fsc_rnmf = zeros(nCase,nRun);

ACC = zeros(9,1);
NMI = zeros(9,1);
PUR = zeros(9,1);
FSC = zeros(9,1);

fprintf('K \n');

tic;
caseIter = 0;
for k =  Cluster   %2-10
    caseIter = caseIter + 1;
    nSample  = nEach*k;
    PercentIter = 0;
    for runIter = 1:nRun  
        index  = 0;
        Samples = zeros(nSample,nDim);
        labels  = zeros(nSample,1);
        
        shuffleClasses = randperm(nClass);
        if strcmp(dataset,'MNIST')
            I = ones(1,10);
            shuffleClasses = shuffleClasses - I;   % label for MNIST is 0-9
        end
        
        for class = 1:k
            idx = find(gnd == shuffleClasses(class));
            sampleEach = fea(idx(1:nEach),:);
            Samples(index+1:index+nEach,:) = sampleEach;
            labels(index+1:index+nEach,:)  = class;
            index = index + nEach;
        end
        
        %------------------------------------------------------------------
        feaSet = Samples;
        gndSet = labels;
        semiSplit = false(size(gndSet));
        
        for class = 1:k  %
            idx = find(gndSet == class);
            shuffleIndexes = randperm(length(idx));
            nSmpUnlabel = floor((percent)*length(idx));
            semiSplit(idx(shuffleIndexes(1:nSmpUnlabel))) = true;
        end
        
        nfeaSet = size(feaSet,1);  %   500k
        nSmpLabeled = sum(semiSplit); % 500k * percent(0.3)
        
        % shuffle the data sets and lables 
        shuffleIndexes = randperm(nfeaSet); % 1 * 500k
        feaSet = feaSet(shuffleIndexes,:); % 500k * 784 shuffle
        gndSet = gndSet(shuffleIndexes); % 500k * 1 shuffle
        semiSplit = semiSplit(shuffleIndexes);  % 500k * 1 logical
        
        % constructing the similarity diagnal matrix based on labled data
        S = diag(semiSplit); % semiSplit(500k * 1 logical) diagonal metirx 500k * 500k
        
        % constructing the label constraint matrix for LpCNMF and CNMF
        E = eye(k); % k * k diagonal metirx
        A_mid = E(:,gndSet(semiSplit)); % k * (500k*percent(0.3))
        
        %璁＄畻杈呭姪鐭╅樀
        A_lpcnmf = zeros(k,nfeaSet); % k * 500k
        A_lpcnmf(:,semiSplit) = A_mid; % column with label turn to A_mid
        A_cnmf =[A_mid' zeros(nSmpLabeled, nfeaSet-nSmpLabeled); zeros(nfeaSet-nSmpLabeled,k) eye(nfeaSet - nSmpLabeled)];
        D1 = ones(k,nSmpLabeled); % k * (500k * percent(0.3))
        D_mid = D1(:,gndSet(semiSplit)); % k * (500k * percent(0.3))
        D2 = zeros(k,nfeaSet - nSmpLabeled); % k * (500k * 0.7)
        A_cdcf = [D_mid  D2]; % k * 500k 

        %Dissimilarity matrix
        d = zeros(nfeaSet, nfeaSet); % 500k * 500k
        for id=1:nSmpLabeled 
            for jd=id:nSmpLabeled 
                if id==jd
                    d(id,jd)=0;
                elseif  gndSet(id) == gndSet(jd)
                    d(id,jd)=0;
                else
                    d(id,jd)=1;
                end
            end
        end        
        D = d + d';

        %Similarity matrix
        s = zeros(nfeaSet, nfeaSet);
        for is=1:nSmpLabeled 
            for js=is:nSmpLabeled 
                if is==js
                    s(is,js)=1;
                elseif gndSet(is) == gndSet(js)
                    s(is,js)=1;
                else
                    s(is,js)=0;
                end
            end
        end
        S_bar = s + s';
        
        %------------------------------------------------------------------------------------
        %Clustering by Label Propagation Constrained Non-negative Matrix Factorization(RNMF)
        tic;
        options = [];
        options.WeightMode = 'HeatKernel';
        options.NeighborMode = 'Supervised';
        options.k = 5;
        options.t = 1;
        options.gnd = gndSet;
        options.maxIter = 2;  % 2
        W = constructW(feaSet,options);
        [~, Zest_rnmf, F] = RNMF(feaSet', A_lpcnmf', k, k, S, W, options); % change  A_lpcnmf
        Zest_rnmf = F*Zest_rnmf;        
        % label = litekmeans(Zest_rnmf, k, 'MaxIter', 200, 'Replicates', 20);
        label = litekmeans(Zest_rnmf, k, 'Replicates', 10);
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');
        time_rnmf(caseIter,runIter) = toc; % End timer and store timetoc;
        

        % save ('ronmf.mat','feaSet','label');

        acc_rnmf(caseIter,runIter) = acc; 
        nmi_rnmf(caseIter,runIter) = nmi;
        fsc_rnmf(caseIter,runIter) = fscore;
        pur_rnmf(caseIter,runIter) = pur; 

        


    end
        
    fprintf('%d \n',k);


    acc=[mean(acc_rnmf(caseIter,:))];    
    nmi=[mean(nmi_rnmf(caseIter,:))];  
    pur=[mean(pur_rnmf(caseIter,:))];  
    fsc=[mean(fsc_rnmf(caseIter,:))];  
    ACC(caseIter,:) = acc;
    NMI(caseIter,:) = nmi;
    PUR(caseIter,:) = pur;
    FSC(caseIter,:) = fsc;


end
toc;
