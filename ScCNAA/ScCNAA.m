function [S,Z,P,X_r,X_c,Y_r,Y_c] = ScCNAA(data,opt)
%%


% Setting
[n_img,q] = size(data.Y_c{1});
[n_snp,p] = size(data.X_c{1});
n_class = opt.class;
h=opt.h;% projection latent feature representation

% Set data
X = data.X_c;
Y_c = data.Y_c;
X_r = data.X_r;
Y_r = data.Y_r;
X_c = getNormalization(X{1,1});
% get normalization
for k=1:n_class
    
    Y_c{k,1} = getNormalization(Y_c{k,1})';
    X_r{k,1} = getNormalization(X_r{k,1});
    Y_r{k,1} = getNormalization(Y_r{k,1})';
    n_r(k) = size(X_r{k},1);
end
n_c = size(X_c,1);
for k=1:n_class
    XK{k}=[X_c;X_r{k}]; %每个模态对应的X
    YK{k} = [Y_c{k},Y_r{k}];
    KK{k} = Kernel_function(XK{k},opt);
end
% clear X_r
%% calculation L for imaging data
for k=1:n_class
    YY=[Y_c{k},Y_r{k}]'; %complete imaging data for each modality
    
    D=(pdist(YY).^2)';
    S{k}=squareform(exp(-D));
    l=sum(S{k},2);
    LY{k}=diag(l)-S{k};
    LX{k} = rand(p,p);
end
clear YY S
XX=X_c;
for k=1:n_class
    XX=[XX;X_r{k}];
end


%% % Img weights Z 
for k = 1:n_class
    S(:, k) = ones(p, 1);
    scale = norm(XK{k} * S(:, k));
    S(:, k) = S(:, k)./scale;
    A(:,k) = ones(size(XK{k},1),1);
    Z(:,:,k) = ones(q,h);
    scale = norm(Z(:,:,k)'*YK{k});
    Z(:,:,k)=Z(:,:,k)./scale;
    
end

%% % initialize H_c and H_r, H_c denote the complete multi-modality data; h*n_c H_r: h*n_r
H_c=rand(h,n_c);
% scale=norm(H_c);
% H_c=H_c./scale;

%initialize H_r, H_r denote the remaing data in m-modality exclude complete
% multi-modality data; h*n_r
for k=1:n_class
    H_r{k}=rand(h,n_r(k));
    Q{k} =zeros(h,n_c+n_r(k));
    P(:, k) = rand(h,1);
    scale = norm(P(:,k));
    P(:, k)  =P(:, k)./scale;
    %         scale=norm(H_r{k});
    %         H_r{k}=H_r{k}./scale;
end


%%  graph connectivity
for k=1:n_class
    PY{k} = get_connectivity(YK{k},2);
    
    E{k}=zeros(h,(n_c+n_r(k)));
end

tol=opt.tol;
maxiter=opt.maxiter;

% Set tuned parameters
beta1 = opt.beta1; % L2,1
beta2 = opt.beta2; % graph
gamma = opt.gamma; % projection latent representation

alpha1 = opt.alpha1; % L1,1
alpha2 = opt.alpha2; % L2,1
lambda = opt.lambda; % LPLACE
eta = opt.eta;
mu = 1e-4;%opt.mu;
pho =1.5;
max_mu = 1e6;
group_x=opt.X_group;

%% Iteration
t=0;
ts = inf;
tz = inf;
tp = inf;
iter = 0;
obj = [];
err = inf;
%%
for iter = 1:maxiter
    
    S_old = S;
    Z_old = Z;
    P_old = P;
    t=t+1;
    for k=1:n_class
        %% -------------------------------------Update Z
        D2g=updateD(Z(:,:,k));
        F1= lambda*YK{k}*LY{k}*YK{k}'+beta1*D2g+beta2*PY{k}+0.5*mu*YK{k}*YK{k}'+ eps;%
        b1= (mu/2)*YK{k}*([H_c,H_r{k}]+E{k}-Q{k}/mu)';
        Z(:,:,k) = F1\b1;
        scale = norm(Z(:,:,k)' *YK{k});
        Z(:,:,k) = Z(:,:,k) ./ scale;
        %%  ------------------------------------Update E
        F1 = Z(:,:,k)'*YK{k} - [H_c,H_r{k}] + Q{k}/mu ;
        b1 = gamma / mu ;
        E{k} = softth(F1,b1) ;
        
        %% -------------------------------------Update H_r
        F11 = P(:,k)*P(:,k)' + mu*eye(h) + eps;
        b11 = P(:,k)*(KK{k}((n_c+1):end, (n_c+1):end) * A(n_c+1:end, k))' + mu*(Z(:,:,k)' *Y_r{k} - E{k}(:,(n_c+1):end) + Q{k}(:,(n_c+1):end)/mu);
        H_r{k} = F11\b11;
        scale = norm(H_r{k});
        H_r{k} = H_r{k}./scale;
    end
    
    %% --------------------------------------Update H_c
    %     F1=P'*P;
    F1 = zeros(h,h);
    b1 = zeros(h,n_c);
    for k=1:n_class
        F1= F1 + P(:,k)'* P(:,k) + mu *eye(h) + eps;
        b1 = b1 + P(:,k) * (KK{k}(1:n_c, 1:n_c) * A(1:n_c, k))' +  mu * (Z(:,:,k)'*Y_c{k} -E{k}(:,1:n_c) +Q{k}(:,1:n_c)/mu);
    end
    
    H_c= F1\b1;
    scale=norm(H_c);
    H_c=H_c./scale;

    %% --------------------------------------Update A, S and LX 
    

    for k=1:n_class
        U{k} = [S(:,k), LX{k}];
        Dg1=updateD_group_Given2(U{k},group_x); %G21
        DDG = diag(Dg1);
        F1 = KK{k} + alpha1*eye(size(XK{k}, 1));
        b1 = [H_c,H_r{k}]'*P(:,k);
        A(:, k) = F1 \ b1;
        S(:, k) = inv(sqrt(DDG)) * XK{k}' * A(:, k);
        scale = norm(XK{k}*S(:, k));
        S(:, k) = S(:, k) ./ scale;
        D1=updateD(LX{k});
        HH = updateD((XK{k} - XK{k}*LX{k}));
        FF1 = (XK{k}'*HH*XK{k} + alpha1/(2*norm(XK{k} - XK{k}*LX{k},'fro')) * DDG + alpha2/(2*norm(XK{k} - XK{k}*LX{k},'fro')) * D1 + eps);
        bb1 = XK{k}'*HH *XK{k};
        LX{k} = FF1 \ bb1;
       
    %% --------------------------------------Update P  
   
        F1 = [H_c,H_r{k}]*[H_c,H_r{k}]'+ eta*eye(h) + eps;
        b1 = [H_c,H_r{k}]*KK{k} * A(:, k);
        P(:, k) = F1\b1;
        scale = norm(P(:, k));
        P(:, k) = P(:, k)./scale;
    %% --------------------------------------Update Q
   
        Q{k} = Q{k} + mu*(Z(:,:,k)'*YK{k} - [H_c,H_r{k}] - E{k});
    end
    %% --------------------------------------Update mu
    mu = min(pho*mu,max_mu);
    
    %% --------------------------------------Iterative termination
    for k = 1:n_class
        errp(k) = norm(abs([H_c,H_r{k}]-Z(:,:,k)'*[Y_c{k},Y_r{k}]),inf);
    end
    
    err(iter)=max(errp);
    tz = max(max(max(abs(Z - Z_old))));
    ts = max(max(abs(S - S_old)));
    tp = max(max(abs(P - P_old)));
    if err(iter) < tol || ( tz < tol || ts < tol )
        break;
    end
end

end