function K = Kernel_function(X, opt)

type = opt.ktype;
sigma = opt.scaler;
N = size(X, 1);

switch type
    
    case 'gauss'
        K = gauss_kernel(X,sigma, N);
        
    case 'linear'
        K = X*X';
        
    case 'ploy'
        K = ploy_kernel(X, dimention);
        
    case 'sigmoid'
        K = tanh_kernel(X, beta, theta);
      
end
end

function S = gauss_kernel(X, sigma, N)
dist2 = repmat(sum((X.^2)', 1), [N 1])' + ...
            repmat(sum((X.^2)',1), [N 1]) - ...
            2*X*(X');
S = exp(-dist2 / (2*sigma));
end

function S = ploy_kernel(X, d)
S = (1 + X*X').^d;
end    

function S = tanh_kernel(X, beta, theta)
S = tanh(beta* X*X' + theta);
end