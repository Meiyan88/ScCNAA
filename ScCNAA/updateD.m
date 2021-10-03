function D = updateD(W, type)

if nargin == 1
    % for L2,1-norm & L1,1-norm
    if size(W,3)> 1
        d = 1 ./ sqrt(sum(sum(W .^ 2, 3),2) + eps);
    else
        d = 1 ./ sqrt(sum(W .^ 2, 2) + eps);
        %     else
        %
    end


D = diag(0.5 * d);
