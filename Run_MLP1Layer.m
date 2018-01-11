clc
clear
close all


n_pca=30;
%%
A=load('cnDieselTrain.mat');

P=A.cnTrainX;
T=A.cnTrainY;
P_test=A.cnTestX;
y=T;

%%
[evec, eval, meanA] = PCA([P P_test]',n_pca);
p=evec';
p=p(:,1:133);
sP=size(p);
disp('PCA is done')
%%
mygoal = 0;
opt_epoch =200;

net = newff(minmax(p), [10,1], {'tansig','purelin'}, 'trainlm', 'learngdm', 'mse');
net.trainParam.show = NaN;
net.trainParam.epochs = opt_epoch;
net.trainParam.goal = mygoal;

net = train(net, p, T);
    
ymlp = sim(net, p);
residuals = (y-ymlp);
residuals_perc = (y-ymlp)./y*100;


figure(1)
plot(y,'--r')
hold on
plot(ymlp,'k','linewidth',2)
xlabel('Sample')
ylabel('Output')
title('MLP')


figure(2)
stem(residuals)
xlabel('Sample');
ylabel('Residual Error');
title('MLP')

figure(3)
stem(residuals_perc)
xlabel('Sample');
ylabel('Residual Error (%)');
title('MLP')

MSE_LS=mse(y,ymlp)
RMS_Ls=rms(y-ymlp)
