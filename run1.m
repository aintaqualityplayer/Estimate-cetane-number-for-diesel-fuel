clc
clear
close all

n_pca=20;
%%

A=load('cnDieselTrain.mat');


P=A.cnTrainX;
T=A.cnTrainY;
P_test=A.cnTestX;
y=T';

%%
[evec, eval, meanA] = PCA([P P_test]',n_pca);
p=evec';
p=p(:,1:133);
sP=size(p);
disp('PCA is done')
%%
%---------------------- Least Square--------------------------------
A=p';
x=(A'*A)^-1*A'*T';

yls=A*x;
residuals = (y-yls);
residuals_perc = (y-yls)./y*100;





figure(1)
plot(T,'--r')
hold on
plot(yls,'k','linewidth',2)
xlabel('Sample')
ylabel('Output')
title('LS')


figure(2)
stem(residuals)
xlabel('Sample');
ylabel('Residual Error');
title('LS')

figure(3)
stem(residuals_perc)
xlabel('Sample');
ylabel('Residual Error (%)');
title('LS')

MSE_LS=mse(y,yls)
RMS_Ls=rms(y-yls)
%%
%------------------------- PLS-------------------------------
n_pca=30;

X = P';
y = T';

[XL,yl,XS,YS,beta,PCTVAR] = plsregress(X,y,n_pca);

figure(4)
plot(1:n_pca,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

ypls = [ones(size(X,1),1) X]*beta;
residuals = (y-ypls);
residuals_perc = (y-ypls)./y*100;





figure(5)
plot(y,'--r')
hold on
plot(ypls,'k','linewidth',2)
xlabel('Sample')
ylabel('Output')
title('PLS')


figure(6)
stem(residuals)
xlabel('Sample');
ylabel('Residual Error');
title('PLS')

figure(7)
stem(residuals_perc)
xlabel('Sample');
ylabel('Residual Error (%)');
title('PLS')

MSE_LS=mse(y,ypls)
RMS_Ls=rms(y-ypls)
