clear;clc
%% ----------------------------------------------------------
addpath('./ScCNAA/');
load data.mat
% [n,p]=size(tr{1}.Xtr{1,1});
load options.mat
load group.mat
kfold=5;
t1=cputime;
opt.class=length(tr{1}.Xtr);
n_class=opt.class;
%% setting parameters
opt.beta1=1e1; % L1,1
opt.beta2=1e0; % graph connetivity constraint
opt.eta=1e-5; % association of QTs
opt.gamma=1e4; % projection latent feature representation

opt.alpha1=1e5; % G21
opt.alpha2=1e-5; % L1
opt.lambda=1e-5; % LPLACE
opt.h=10; % projection feature
opt.X_group = group_idx;

        for k=1:kfold
            
            % training set
            for kk=1:n_class
                itrain_set.X_c{kk,1} = tr{k}.Xtr{1,1};
                itrain_set.Y_c{kk,1} = tr{k}.Ytr{kk,1};
                
                % testing set
                itest_set.X{kk,1} = getNormalization(te{k}.Xte{1,1});
                itest_set.Y{kk,1} = getNormalization(te{k}.Yte{1,1});
            end
            itrain_set.X_r{1,1} = re{k}.Xtr{1,1}(51:end,:);
            itrain_set.X_r{2,1} = re{k}.Xtr{2,1}(1:50,:);
            itrain_set.Y_r{1,1} = re{k}.Ytr{1,1}(51:end,:);
            itrain_set.Y_r{2,1} = re{k}.Ytr{2,1}(1:50,:);
            %% multi-modal non-parameters regression; trainning model
            tic;
            [S1,Z1,P1,X_r,X_c,Y_r,Y_c] = ScCNAA(itrain_set,opt);
            toc;
            S{k}=S1;
            Z{k}=Z1;
            P{k}=P1;
            weight.S{k} = S1;
            weight.Z{k} = Z1;
            weight.P{k} = P1;
           
            test_Y11=itest_set.Y{1,1}*Z1(:,:,1)*P1(:, 1);
            test_Y21=itest_set.Y{2,1}*Z1(:,:,2)*P1(:, 2);
            
            pred_Y11= itest_set.X{1,1}*(S1(:, 1));
            pred_Y21= itest_set.X{2,1}*(S1(:, 2));
            %RMSE
            testRMSE1(k)=sqrt(mean(mean((test_Y11-pred_Y11).^2)));
            testRMSE2(k)=sqrt(mean(mean((test_Y21-pred_Y11).^2)));
            %CC
            testCC1(k)=corr(mean(test_Y11,2),mean(pred_Y11,2));
            testCC2(k)=corr(mean(test_Y21,2),mean(pred_Y11,2));
            
            WW1=w1+w2;
            [~,n]=find(WW1~=0);
            label=zeros(1,p);
            label(:,n)=1;
            [tpr,fpr,thresholds] =roc(label,mean(abs(S1), 2)');
            AUC.auc_S(k)=trapz(fpr,tpr);
            
            v=(v1+v2)/2;
            Z2=mean(abs(Z1),2);
            Z2=reshape(abs(Z2),q,2,1);
            Z3=mean(Z2,2);
            [~,n]=find(v~=0);
            label=zeros(1,q);
            label(:,n)=1;
            [tpr,fpr,thresholds] =roc(label,(Z3)');
            AUC.auc_V(k)=trapz(fpr,tpr);
            [~,n]=find(v1~=0);
            label=zeros(1,q);
            label(:,n)=1;
            
            [tpr,fpr,thresholds] =roc(label,abs(Z2(:,1))');
            AUC.auc_V1(k)=trapz(fpr,tpr);
            [~,n]=find(v2~=0);
            label=zeros(1,q);
            label(:,n)=1;
            
            [tpr,fpr,thresholds] =roc(label,abs(Z2(:,2))');
            AUC.auc_V2(k)=trapz(fpr,tpr);
        end
        % S12=((S{1})+(S{2})+(S{3})+(S{4})+(S{5}))/5;
        auc_S=[mean(AUC.auc_S) std(AUC.auc_S)];
        auc_Z=[mean(AUC.auc_V) std(AUC.auc_V)];
        testRMSE=(testRMSE1+testRMSE2)/n_class;
        RMSE=[mean( testRMSE) std( testRMSE)];
        RMSE1 = mean( testRMSE);
        testCC=abs((testCC1+testCC2)/n_class);
        CC=[mean(testCC) std(testCC)];
        CC1 = mean(testCC);
        %------------------Result
        Results.AUC = AUC;
        Results.aucS = auc_S;
        Results.aucZ = auc_Z;
        Results.RMSE = RMSE;
        Results.CC = CC;
        Results.weight = weight;



t=cputime-t1;
pathname = './result/';
filename = 'result.mat' ;
save([pathname,filename],'Results', 't')
