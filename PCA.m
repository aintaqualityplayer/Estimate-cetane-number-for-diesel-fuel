function [evec, eval, meanA] = PCA(A, maxVec)
%Principal Component Analysis
%A has th
[ rows, cols] = size(A);
if(nargin < 2)
maxVec = rows;
end

maxVec = min(maxVec, rows);

meanA = mean(A,2);
A = A-meanA*ones(1, cols);
%[evec eval] = eig(A’*A);
[evec eval] = eig(A*A');

[eval ind]  =  sort(-1*diag(eval));
%[eval ind]  =  sort(diag(eval), ”);
eval    = -1*eval(1:maxVec);
evec    = evec(:, ind(1:maxVec));