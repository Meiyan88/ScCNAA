function E = softth(F,lambda)
[h,n]=size(F);
lambda = repmat(lambda,h,n);
E = max(abs(F)-lambda,zeros(h,n));
% if F>lambda
%     E = F - lambda;
% else if F<-lambda
%         E=F+lambda;
%     else
%         E = zeros(h,n);
%     end
end