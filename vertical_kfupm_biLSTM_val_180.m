clc
clear all
close all
%M=csvread("WS_10m.csv",1,3);
%M=csvread("WS_hr.csv",2,3,[2 3 3000 3]);
M=csvread("WS_KFUPM_10m_2015.csv",1,2);
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%M=M(:,[2 4 5 6 7 8 10 

days=360;
%numdat=6*24*days;
inputsize=4;
%M=M(1:(numdat),:);
N=length(M);

for k=1:11
for i=1:N  
    if M(i,k)> 20 %CLEAN THE DATA if 9999, then replace with previous value
        M(i,k)=M(i-1,k);
    elseif M(i,k)<= 0
    M(i,k)=M(i-1,k);
    end
end
end

M=fliplr(M);

ii=1;
for i=1:N
    diff0=M(i,2:11)-M(i,1:10);
    lt0=sum(find(diff0<0.1));
    if lt0==0 
        MN(ii,:)=M(i,:);
        ii=ii+1;
    end
end

mt15=find(MN(:,6)<=15);   
M=MN(mt15,:);
mt10=find(M(:,3)<=10);   
M=M(mt10,:);

M=[M(:,[1 2 3 4]) (M(:,4)+M(:,5))/2 M(:,5) (M(:,5)+M(:,6))/2 M(:,6) (M(:,6)+M(:,7))/2 M(:,7) (M(:,7)+M(:,8))/2 M(:,8) (M(:,8)+M(:,9))/2 M(:,9) (M(:,9)+M(:,10))/2 M(:,10) (M(:,10)+M(:,11))/2 M(:,11)];
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%

perc=100;
numdat=length(M);

%R=6; % Every 6 makes an hour
%mm=floor(N/R);
%for i=1:mm
%    j=(i-1)*R+1;
%    MD(i,1)=mean(M(j:j+R-1));
%end

trainingnum=floor(0.7*numdat); % Num of training samples
valnum=floor(0.1*numdat); % Num of training samples
maxx=max(max(M(1:trainingnum,1:inputsize)));
training=M(1:trainingnum,:);

series=training/maxx;
datasize=size(series);
nex=1;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

valdat=M((trainingnum+1):trainingnum+valnum,:);
seriesVal=valdat/maxx;
Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing




testing=M((trainingnum+valnum+1):end,:);
seriesT=testing/maxx;

%numdata=max(datasize)-(inputsize+ahead-1);
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN

%outval = netMLP(P);

trainingtargetmax=trainingtarget*maxx;

height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
% bilstm WSE

bilstmP50 = traininginput';
bilstmY50 = trainingtarget';

bilstmP50V = Valinput';
bilstmY50V = Valtarget';
bilstmtarget50V=bilstmY50V'*maxx;

bilstmPtest50 = testinginput';
bilstmYtest50 = testingtarget';
bilstmtestingtarget50=bilstmYtest50'*maxx;

numiter=200;
numhid=5;
miniBatchSize = 8000;
numFeatures=inputsize+(nex-1);
numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%layers = [ ...
%    sequenceInputLayer(numFeatures)
%    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%    fullyConnectedLayer(numHiddenUnits)
%    dropoutLayer(0.1)
%    fullyConnectedLayer(numResponses)
%   regressionLayer];

maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP50,bilstmY50,layers,options);
outval = predict(net,bilstmP50,'MiniBatchSize',1);

outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP50V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf50V=outvalV'*maxx;
bilstmmseV50=mse(outvalmaxV,bilstmtarget50V);
bilstmmapeV50=mape(outvalmaxV,bilstmtarget50V);
bilstmmbeV50=mbe(outvalmaxV,bilstmtarget50V);
bilstmr2V50=rsquare(outvalmaxV,bilstmtarget50V);
bilstmperf50V=[bilstmmseV50 bilstmmapeV50 bilstmmbeV50 bilstmr2V50];
[bilstmperf50V]
%
outvaltest = predict(net,bilstmPtest50,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
bilstmOutf50=outvaltestmax;
bilstmtestingtargetmax50=testingtarget50;
bilstmmsetest50=mse(bilstmOutf50,bilstmtestingtargetmax50);
bilstmmapetest50=mape(bilstmOutf50,bilstmtestingtargetmax50);
bilstmmbetest50=mbe(bilstmOutf50,bilstmtestingtargetmax50);
bilstmr2test50=rsquare(bilstmOutf50,bilstmtestingtargetmax50);

bilstmperf50=[bilstmmsetest50 bilstmmapetest50 bilstmmbetest50 bilstmr2test50];
bilstmOutf50train=outvalmax;
bilstmPtestMax50=bilstmPtest50'*maxx;
perfall=[mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanbilstm50=mean([bilstmPtestMax50'; bilstmOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanbilstm50,height50,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

figure
plot([bilstmPtestMax50'; bilstmOutf50' ],height50);

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[bilstmperf50]

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y60 = trainingtarget';
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;

testingtargetmax=testingtarget*maxx;
target60=testingtarget60;

%
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];
rl60=[1:mxr];
% bilstm WSE
%
bilstmP60 = [bilstmP50; bilstmOutf50train'/maxx];
bilstmY60 = trainingtarget';

bilstmP60V = [bilstmP50V; bilstmOutf50V'/maxx];
bilstmY60V = Valtarget';
bilstmtarget60V=bilstmY60V'*maxx;

bilstmPtest60 = [bilstmPtest50; bilstmOutf50'/maxx];
bilstmYtest60 = testingtarget';
bilstmtestingtarget60=bilstmYtest60'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP60,bilstmY60,layers,options);
outval = predict(net,bilstmP60,'MiniBatchSize',1);

bilstmOutf60train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP60V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf60V=outvalV'*maxx;
bilstmmseV60=mse(outvalmaxV,bilstmtarget60V);
bilstmmapeV60=mape(outvalmaxV,bilstmtarget60V);
bilstmmbeV60=mbe(outvalmaxV,bilstmtarget60V);
bilstmr2V60=rsquare(outvalmaxV,bilstmtarget60V);
bilstmperf60V=[bilstmmseV60 bilstmmapeV60 bilstmmbeV60 bilstmr2V60];
[bilstmperf60V]



outvaltest = predict(net,bilstmPtest60,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
bilstmOutf60=outvaltestmax;
bilstmmsetest60=mse(bilstmOutf60,testingtarget60);
bilstmmapetest60=mape(bilstmOutf60,testingtarget60);
bilstmmbetest60=mbe(bilstmOutf60,testingtarget60);
bilstmr2test60=rsquare(bilstmOutf60,testingtarget60);
bilstmperf60=[bilstmmsetest60 bilstmmapetest60 bilstmmbetest60 bilstmr2test60];
bilstmPtestMax60=bilstmPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanbilstm60=[meanbilstm50 mean(bilstmOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanbilstm60,height60,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax60'; bilstmOutf60' ],height60);
[bilstmperf60]

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y70 = trainingtarget';
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;

testingtargetmax=testingtarget*maxx;
target70=testingtarget70;

%
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];
rl70=[1:mxr];
% bilstm WSE
%
bilstmP70 = [bilstmP60; bilstmOutf60train'/maxx];
bilstmY70 = trainingtarget';

bilstmP70V = [bilstmP60V; bilstmOutf60V'/maxx];
bilstmY70V = Valtarget';
bilstmtarget70V=bilstmY70V'*maxx;

bilstmPtest70 = [bilstmPtest60; bilstmOutf60'/maxx];
bilstmYtest70 = testingtarget';
bilstmtestingtarget70=bilstmYtest70'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP70,bilstmY70,layers,options);
outval = predict(net,bilstmP70,'MiniBatchSize',1);
outvalmax=outval'*maxx;

outval = predict(net,bilstmP70,'MiniBatchSize',1);
bilstmOutf70train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP70V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf70V=outvalV'*maxx;
bilstmmseV70=mse(outvalmaxV,bilstmtarget70V);
bilstmmapeV70=mape(outvalmaxV,bilstmtarget70V);
bilstmmbeV70=mbe(outvalmaxV,bilstmtarget70V);
bilstmr2V70=rsquare(outvalmaxV,bilstmtarget70V);
bilstmperf70V=[bilstmmseV70 bilstmmapeV70 bilstmmbeV70 bilstmr2V70];
[bilstmperf70V]



outvaltest = predict(net,bilstmPtest70,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
bilstmOutf70=outvaltestmax;
bilstmmsetest70=mse(bilstmOutf70,testingtarget70);
bilstmmapetest70=mape(bilstmOutf70,testingtarget70);
bilstmmbetest70=mbe(bilstmOutf70,testingtarget70);
bilstmr2test70=rsquare(bilstmOutf70,testingtarget70);
bilstmperf70=[bilstmmsetest70 bilstmmapetest70 bilstmmbetest70 bilstmr2test70];
bilstmPtestMax70=bilstmPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanbilstm70=[meanbilstm60 mean(bilstmOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanbilstm70,height70,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax70'; bilstmOutf70' ],height70);
[bilstmperf70]

%% 80

nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y80 = trainingtarget';
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;

testingtargetmax=testingtarget*maxx;
target80=testingtarget80;

%
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];
rl80=[1:mxr];
% bilstm WSE
%
bilstmP80 = [bilstmP70; bilstmOutf70train'/maxx];
bilstmY80 = trainingtarget';

bilstmP80V = [bilstmP70V; bilstmOutf70V'/maxx];
bilstmY80V = Valtarget';
bilstmtarget80V=bilstmY80V'*maxx;

bilstmPtest80 = [bilstmPtest70; bilstmOutf70'/maxx];
bilstmYtest80 = testingtarget';
bilstmtestingtarget80=bilstmYtest80'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP80,bilstmY80,layers,options);
outval = predict(net,bilstmP80,'MiniBatchSize',1);
outvalmax=outval'*maxx;


bilstmOutf80train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP80V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf80V=outvalV'*maxx;
bilstmmseV80=mse(outvalmaxV,bilstmtarget80V);
bilstmmapeV80=mape(outvalmaxV,bilstmtarget80V);
bilstmmbeV80=mbe(outvalmaxV,bilstmtarget80V);
bilstmr2V80=rsquare(outvalmaxV,bilstmtarget80V);
bilstmperf80V=[bilstmmseV80 bilstmmapeV80 bilstmmbeV80 bilstmr2V80];
[bilstmperf80V]



outvaltest = predict(net,bilstmPtest80,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
bilstmOutf80=outvaltestmax;
bilstmmsetest80=mse(bilstmOutf80,testingtarget80);
bilstmmapetest80=mape(bilstmOutf80,testingtarget80);
bilstmmbetest80=mbe(bilstmOutf80,testingtarget80);
bilstmr2test80=rsquare(bilstmOutf80,testingtarget80);
bilstmperf80=[bilstmmsetest80 bilstmmapetest80 bilstmmbetest80 bilstmr2test80];
bilstmPtestMax80=bilstmPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanbilstm80=[meanbilstm70 mean(bilstmOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanbilstm80,height80,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax80'; bilstmOutf80' ],height80);
[bilstmperf80]

%% 90

nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y90 = trainingtarget';
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;

testingtargetmax=testingtarget*maxx;
target90=testingtarget90;

%
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];
rl90=[1:mxr];
% bilstm WSE
%
bilstmP90 = [bilstmP80; bilstmOutf80train'/maxx];
bilstmY90 = trainingtarget';

bilstmP90V = [bilstmP80V; bilstmOutf80V'/maxx];
bilstmY90V = Valtarget';
bilstmtarget90V=bilstmY90V'*maxx;

bilstmPtest90 = [bilstmPtest80; bilstmOutf80'/maxx];
bilstmYtest90 = testingtarget';
bilstmtestingtarget90=bilstmYtest90'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP90,bilstmY90,layers,options);
outval = predict(net,bilstmP90,'MiniBatchSize',1);
outvalmax=outval'*maxx;

bilstmOutf90train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP90V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf90V=outvalV'*maxx;
bilstmmseV90=mse(outvalmaxV,bilstmtarget90V);
bilstmmapeV90=mape(outvalmaxV,bilstmtarget90V);
bilstmmbeV90=mbe(outvalmaxV,bilstmtarget90V);
bilstmr2V90=rsquare(outvalmaxV,bilstmtarget90V);
bilstmperf90V=[bilstmmseV90 bilstmmapeV90 bilstmmbeV90 bilstmr2V90];
[bilstmperf90V]



outvaltest = predict(net,bilstmPtest90,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
bilstmOutf90=outvaltestmax;
bilstmmsetest90=mse(bilstmOutf90,testingtarget90);
bilstmmapetest90=mape(bilstmOutf90,testingtarget90);
bilstmmbetest90=mbe(bilstmOutf90,testingtarget90);
bilstmr2test90=rsquare(bilstmOutf90,testingtarget90);
bilstmperf90=[bilstmmsetest90 bilstmmapetest90 bilstmmbetest90 bilstmr2test90];
bilstmPtestMax90=bilstmPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanbilstm90=[meanbilstm80 mean(bilstmOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanbilstm90,height90,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax90'; bilstmOutf90' ],height90);
[bilstmperf90]

%% 100

nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y100 = trainingtarget';
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;

testingtargetmax=testingtarget*maxx;
target100=testingtarget100;

%
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];
rl100=[1:mxr];
% bilstm WSE
%
bilstmP100 = [bilstmP90; bilstmOutf90train'/maxx];
bilstmY100 = trainingtarget';

bilstmP100V = [bilstmP90V; bilstmOutf90V'/maxx];
bilstmY100V = Valtarget';
bilstmtarget100V=bilstmY100V'*maxx;

bilstmPtest100 = [bilstmPtest90; bilstmOutf90'/maxx];
bilstmYtest100 = testingtarget';
bilstmtestingtarget100=bilstmYtest100'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP100,bilstmY100,layers,options);

outval = predict(net,bilstmP100,'MiniBatchSize',1);
bilstmOutf100train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP100V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf100V=outvalV'*maxx;
bilstmmseV100=mse(outvalmaxV,bilstmtarget100V);
bilstmmapeV100=mape(outvalmaxV,bilstmtarget100V);
bilstmmbeV100=mbe(outvalmaxV,bilstmtarget100V);
bilstmr2V100=rsquare(outvalmaxV,bilstmtarget100V);
bilstmperf100V=[bilstmmseV100 bilstmmapeV100 bilstmmbeV100 bilstmr2V100];
[bilstmperf100V]



outvaltest = predict(net,bilstmPtest100,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
bilstmOutf100=outvaltestmax;
bilstmmsetest100=mse(bilstmOutf100,testingtarget100);
bilstmmapetest100=mape(bilstmOutf100,testingtarget100);
bilstmmbetest100=mbe(bilstmOutf100,testingtarget100);
bilstmr2test100=rsquare(bilstmOutf100,testingtarget100);
bilstmperf100=[bilstmmsetest100 bilstmmapetest100 bilstmmbetest100 bilstmr2test100];
bilstmPtestMax100=bilstmPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanbilstm100=[meanbilstm90 mean(bilstmOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanbilstm100,height100,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax100'; bilstmOutf100' ],height100);
[bilstmperf100]

%% 110

nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y110 = trainingtarget';
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;

testingtargetmax=testingtarget*maxx;
target110=testingtarget110;

%
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];
rl110=[1:mxr];
% bilstm WSE
%
bilstmP110 = [bilstmP100; bilstmOutf100train'/maxx];
bilstmY110 = trainingtarget';

bilstmP110V = [bilstmP100V; bilstmOutf100V'/maxx];
bilstmY110V = Valtarget';
bilstmtarget110V=bilstmY110V'*maxx;

bilstmPtest110 = [bilstmPtest100; bilstmOutf100'/maxx];
bilstmYtest110 = testingtarget';
bilstmtestingtarget110=bilstmYtest110'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP110,bilstmY110,layers,options);
outval = predict(net,bilstmP110,'MiniBatchSize',1);



bilstmOutf110train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP110V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf110V=outvalV'*maxx;
bilstmmseV110=mse(outvalmaxV,bilstmtarget110V);
bilstmmapeV110=mape(outvalmaxV,bilstmtarget110V);
bilstmmbeV110=mbe(outvalmaxV,bilstmtarget110V);
bilstmr2V110=rsquare(outvalmaxV,bilstmtarget110V);
bilstmperf110V=[bilstmmseV110 bilstmmapeV110 bilstmmbeV110 bilstmr2V110];
[bilstmperf110V]



outvaltest = predict(net,bilstmPtest110,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
bilstmOutf110=outvaltestmax;
bilstmmsetest110=mse(bilstmOutf110,testingtarget110);
bilstmmapetest110=mape(bilstmOutf110,testingtarget110);
bilstmmbetest110=mbe(bilstmOutf110,testingtarget110);
bilstmr2test110=rsquare(bilstmOutf110,testingtarget110);
bilstmperf110=[bilstmmsetest110 bilstmmapetest110 bilstmmbetest110 bilstmr2test110];
bilstmPtestMax110=bilstmPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanbilstm110=[meanbilstm100 mean(bilstmOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanbilstm110,height110,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax110'; bilstmOutf110' ],height110);
[bilstmperf110]

%% 120

nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y120 = trainingtarget';
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;

testingtargetmax=testingtarget*maxx;
target120=testingtarget120;

%
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];
rl120=[1:mxr];
% bilstm WSE
%
bilstmP120 = [bilstmP110; bilstmOutf110train'/maxx];
bilstmY120 = trainingtarget';

bilstmP120V = [bilstmP110V; bilstmOutf110V'/maxx];
bilstmY120V = Valtarget';
bilstmtarget120V=bilstmY120V'*maxx;

bilstmPtest120 = [bilstmPtest110; bilstmOutf110'/maxx];
bilstmYtest120 = testingtarget';
bilstmtestingtarget120=bilstmYtest120'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP120,bilstmY120,layers,options);
outval = predict(net,bilstmP120,'MiniBatchSize',1);

outval = predict(net,bilstmP120,'MiniBatchSize',1);
bilstmOutf120train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP120V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf120V=outvalV'*maxx;
bilstmmseV120=mse(outvalmaxV,bilstmtarget120V);
bilstmmapeV120=mape(outvalmaxV,bilstmtarget120V);
bilstmmbeV120=mbe(outvalmaxV,bilstmtarget120V);
bilstmr2V120=rsquare(outvalmaxV,bilstmtarget120V);
bilstmperf120V=[bilstmmseV120 bilstmmapeV120 bilstmmbeV120 bilstmr2V120];
[bilstmperf120V]



outvaltest = predict(net,bilstmPtest120,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
bilstmOutf120=outvaltestmax;
bilstmmsetest120=mse(bilstmOutf120,testingtarget120);
bilstmmapetest120=mape(bilstmOutf120,testingtarget120);
bilstmmbetest120=mbe(bilstmOutf120,testingtarget120);
bilstmr2test120=rsquare(bilstmOutf120,testingtarget120);
bilstmperf120=[bilstmmsetest120 bilstmmapetest120 bilstmmbetest120 bilstmr2test120];
bilstmPtestMax120=bilstmPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanbilstm120=[meanbilstm110 mean(bilstmOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanbilstm120,height120,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax120'; bilstmOutf120' ],height120);
[bilstmperf120]
%% 130

nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y130 = trainingtarget';
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;

testingtargetmax=testingtarget*maxx;
target130=testingtarget130;

%
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];
rl130=[1:mxr];
% bilstm WSE
%
bilstmP130 = [bilstmP120; bilstmOutf120train'/maxx];
bilstmY130 = trainingtarget';

bilstmP130V = [bilstmP120V; bilstmOutf120V'/maxx];
bilstmY130V = Valtarget';
bilstmtarget130V=bilstmY130V'*maxx;

bilstmPtest130 = [bilstmPtest120; bilstmOutf120'/maxx];
bilstmYtest130 = testingtarget';
bilstmtestingtarget130=bilstmYtest130'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP130,bilstmY130,layers,options);
outval = predict(net,bilstmP130,'MiniBatchSize',1);

outval = predict(net,bilstmP130,'MiniBatchSize',1);
bilstmOutf130train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP130V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf130V=outvalV'*maxx;
bilstmmseV130=mse(outvalmaxV,bilstmtarget130V);
bilstmmapeV130=mape(outvalmaxV,bilstmtarget130V);
bilstmmbeV130=mbe(outvalmaxV,bilstmtarget130V);
bilstmr2V130=rsquare(outvalmaxV,bilstmtarget130V);
bilstmperf130V=[bilstmmseV130 bilstmmapeV130 bilstmmbeV130 bilstmr2V130];
[bilstmperf130V]



outvaltest = predict(net,bilstmPtest130,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
bilstmOutf130=outvaltestmax;
bilstmmsetest130=mse(bilstmOutf130,testingtarget130);
bilstmmapetest130=mape(bilstmOutf130,testingtarget130);
bilstmmbetest130=mbe(bilstmOutf130,testingtarget130);
bilstmr2test130=rsquare(bilstmOutf130,testingtarget130);
bilstmperf130=[bilstmmsetest130 bilstmmapetest130 bilstmmbetest130 bilstmr2test130];
bilstmPtestMax130=bilstmPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanbilstm130=[meanbilstm120 mean(bilstmOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanbilstm130,height130,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax130'; bilstmOutf130' ],height130);
[bilstmperf130]
%% 140

nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y140 = trainingtarget';
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;

testingtargetmax=testingtarget*maxx;
target140=testingtarget140;

%
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];
rl140=[1:mxr];
% bilstm WSE
%
bilstmP140 = [bilstmP130; bilstmOutf130train'/maxx];
bilstmY140 = trainingtarget';

bilstmP140V = [bilstmP130V; bilstmOutf130V'/maxx];
bilstmY140V = Valtarget';
bilstmtarget140V=bilstmY140V'*maxx;

bilstmPtest140 = [bilstmPtest130; bilstmOutf130'/maxx];
bilstmYtest140 = testingtarget';
bilstmtestingtarget140=bilstmYtest140'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP140,bilstmY140,layers,options);
outval = predict(net,bilstmP140,'MiniBatchSize',1);

outval = predict(net,bilstmP140,'MiniBatchSize',1);
bilstmOutf140train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP140V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf140V=outvalV'*maxx;
bilstmmseV140=mse(outvalmaxV,bilstmtarget140V);
bilstmmapeV140=mape(outvalmaxV,bilstmtarget140V);
bilstmmbeV140=mbe(outvalmaxV,bilstmtarget140V);
bilstmr2V140=rsquare(outvalmaxV,bilstmtarget140V);
bilstmperf140V=[bilstmmseV140 bilstmmapeV140 bilstmmbeV140 bilstmr2V140];
[bilstmperf140V]



outvaltest = predict(net,bilstmPtest140,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
bilstmOutf140=outvaltestmax;
bilstmmsetest140=mse(bilstmOutf140,testingtarget140);
bilstmmapetest140=mape(bilstmOutf140,testingtarget140);
bilstmmbetest140=mbe(bilstmOutf140,testingtarget140);
bilstmr2test140=rsquare(bilstmOutf140,testingtarget140);
bilstmperf140=[bilstmmsetest140 bilstmmapetest140 bilstmmbetest140 bilstmr2test140];
bilstmPtestMax140=bilstmPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanbilstm140=[meanbilstm130 mean(bilstmOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanbilstm140,height140,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax140'; bilstmOutf140' ],height140);
[bilstmperf140]

%%
%% 150

nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y150 = trainingtarget';
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;

testingtargetmax=testingtarget*maxx;
target150=testingtarget150;

%
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];
rl150=[1:mxr];
% bilstm WSE
%
bilstmP150 = [bilstmP140; bilstmOutf140train'/maxx];
bilstmY150 = trainingtarget';

bilstmP150V = [bilstmP140V; bilstmOutf140V'/maxx];
bilstmY150V = Valtarget';
bilstmtarget150V=bilstmY150V'*maxx;

bilstmPtest150 = [bilstmPtest140; bilstmOutf140'/maxx];
bilstmYtest150 = testingtarget';
bilstmtestingtarget150=bilstmYtest150'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP150,bilstmY150,layers,options);
outval = predict(net,bilstmP150,'MiniBatchSize',1);

outval = predict(net,bilstmP150,'MiniBatchSize',1);
bilstmOutf150train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP150V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf150V=outvalV'*maxx;
bilstmmseV150=mse(outvalmaxV,bilstmtarget150V);
bilstmmapeV150=mape(outvalmaxV,bilstmtarget150V);
bilstmmbeV150=mbe(outvalmaxV,bilstmtarget150V);
bilstmr2V150=rsquare(outvalmaxV,bilstmtarget150V);
bilstmperf150V=[bilstmmseV150 bilstmmapeV150 bilstmmbeV150 bilstmr2V150];
[bilstmperf150V]



outvaltest = predict(net,bilstmPtest150,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
bilstmOutf150=outvaltestmax;
bilstmmsetest150=mse(bilstmOutf150,testingtarget150);
bilstmmapetest150=mape(bilstmOutf150,testingtarget150);
bilstmmbetest150=mbe(bilstmOutf150,testingtarget150);
bilstmr2test150=rsquare(bilstmOutf150,testingtarget150);
bilstmperf150=[bilstmmsetest150 bilstmmapetest150 bilstmmbetest150 bilstmr2test150];
bilstmPtestMax150=bilstmPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanbilstm150=[meanbilstm140 mean(bilstmOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanbilstm150,height150,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax150'; bilstmOutf150' ],height150);
[bilstmperf150]

%% 160

nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y160 = trainingtarget';
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;

testingtargetmax=testingtarget*maxx;
target160=testingtarget160;

%
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];
rl160=[1:mxr];
% bilstm WSE
%
bilstmP160 = [bilstmP150; bilstmOutf150train'/maxx];
bilstmY160 = trainingtarget';

bilstmP160V = [bilstmP150V; bilstmOutf150V'/maxx];
bilstmY160V = Valtarget';
bilstmtarget160V=bilstmY160V'*maxx;

bilstmPtest160 = [bilstmPtest150; bilstmOutf150'/maxx];
bilstmYtest160 = testingtarget';
bilstmtestingtarget160=bilstmYtest160'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP160,bilstmY160,layers,options);
outval = predict(net,bilstmP160,'MiniBatchSize',1);

outval = predict(net,bilstmP160,'MiniBatchSize',1);
bilstmOutf160train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP160V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf160V=outvalV'*maxx;
bilstmmseV160=mse(outvalmaxV,bilstmtarget160V);
bilstmmapeV160=mape(outvalmaxV,bilstmtarget160V);
bilstmmbeV160=mbe(outvalmaxV,bilstmtarget160V);
bilstmr2V160=rsquare(outvalmaxV,bilstmtarget160V);
bilstmperf160V=[bilstmmseV160 bilstmmapeV160 bilstmmbeV160 bilstmr2V160];
[bilstmperf160V]



outvaltest = predict(net,bilstmPtest160,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
bilstmOutf160=outvaltestmax;
bilstmmsetest160=mse(bilstmOutf160,testingtarget160);
bilstmmapetest160=mape(bilstmOutf160,testingtarget160);
bilstmmbetest160=mbe(bilstmOutf160,testingtarget160);
bilstmr2test160=rsquare(bilstmOutf160,testingtarget160);
bilstmperf160=[bilstmmsetest160 bilstmmapetest160 bilstmmbetest160 bilstmr2test160];
bilstmPtestMax160=bilstmPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanbilstm160=[meanbilstm150 mean(bilstmOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanbilstm160,height160,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax160'; bilstmOutf160' ],height160);
[bilstmperf160]

%%
%% 170

nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);                              

Y170 = trainingtarget';
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;

testingtargetmax=testingtarget*maxx;
target170=testingtarget170;

%
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];
rl170=[1:mxr];
% bilstm WSE
%
bilstmP170 = [bilstmP160; bilstmOutf160train'/maxx];
bilstmY170 = trainingtarget';

bilstmP170V = [bilstmP160V; bilstmOutf160V'/maxx];
bilstmY170V = Valtarget';
bilstmtarget170V=bilstmY170V'*maxx;

bilstmPtest170 = [bilstmPtest160; bilstmOutf160'/maxx];
bilstmYtest170 = testingtarget';
bilstmtestingtarget170=bilstmYtest170'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP170,bilstmY170,layers,options);
outval = predict(net,bilstmP170,'MiniBatchSize',1);

outval = predict(net,bilstmP170,'MiniBatchSize',1);
bilstmOutf170train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP170V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf170V=outvalV'*maxx;
bilstmmseV170=mse(outvalmaxV,bilstmtarget170V);
bilstmmapeV170=mape(outvalmaxV,bilstmtarget170V);
bilstmmbeV170=mbe(outvalmaxV,bilstmtarget170V);
bilstmr2V170=rsquare(outvalmaxV,bilstmtarget170V);
bilstmperf170V=[bilstmmseV170 bilstmmapeV170 bilstmmbeV170 bilstmr2V170];
[bilstmperf170V]



outvaltest = predict(net,bilstmPtest170,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
bilstmOutf170=outvaltestmax;
bilstmmsetest170=mse(bilstmOutf170,testingtarget170);
bilstmmapetest170=mape(bilstmOutf170,testingtarget170);
bilstmmbetest170=mbe(bilstmOutf170,testingtarget170);
bilstmr2test170=rsquare(bilstmOutf170,testingtarget170);
bilstmperf170=[bilstmmsetest170 bilstmmapetest170 bilstmmbetest170 bilstmr2test170];
bilstmPtestMax170=bilstmPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanbilstm170=[meanbilstm160 mean(bilstmOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanbilstm170,height170,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax170'; bilstmOutf170' ],height170);
[bilstmperf170]

%%
%% 180

nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);

Valinput=seriesVal(:,1:inputsize);
Valtarget=seriesVal(:,inputsize+nex);

%testing
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y180 = trainingtarget';
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;

testingtargetmax=testingtarget*maxx;
target180=testingtarget180;

%
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];
rl180=[1:mxr];
% bilstm WSE
%
bilstmP180 = [bilstmP170; bilstmOutf170train'/maxx];
bilstmY180 = trainingtarget';

bilstmP180V = [bilstmP170V; bilstmOutf170V'/maxx];
bilstmY180V = Valtarget';
bilstmtarget180V=bilstmY180V'*maxx;

bilstmPtest180 = [bilstmPtest170; bilstmOutf170'/maxx];
bilstmYtest180 = testingtarget';
bilstmtestingtarget180=bilstmYtest180'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = numiter;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(bilstmP180,bilstmY180,layers,options);
outval = predict(net,bilstmP180,'MiniBatchSize',1);

outval = predict(net,bilstmP180,'MiniBatchSize',1);
bilstmOutf180train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,bilstmP180V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
bilstmOutf180V=outvalV'*maxx;
bilstmmseV180=mse(outvalmaxV,bilstmtarget180V);
bilstmmapeV180=mape(outvalmaxV,bilstmtarget180V);
bilstmmbeV180=mbe(outvalmaxV,bilstmtarget180V);
bilstmr2V180=rsquare(outvalmaxV,bilstmtarget180V);
bilstmperf180V=[bilstmmseV180 bilstmmapeV180 bilstmmbeV180 bilstmr2V180];
[bilstmperf180V]



outvaltest = predict(net,bilstmPtest180,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
bilstmOutf180=outvaltestmax;
bilstmmsetest180=mse(bilstmOutf180,testingtarget180);
bilstmmapetest180=mape(bilstmOutf180,testingtarget180);
bilstmmbetest180=mbe(bilstmOutf180,testingtarget180);
bilstmr2test180=rsquare(bilstmOutf180,testingtarget180);
bilstmperf180=[bilstmmsetest180 bilstmmapetest180 bilstmmbetest180 bilstmr2test180];
bilstmPtestMax180=bilstmPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanbilstm180=[meanbilstm170 mean(bilstmOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanbilstm180,height180,'-.g');

hold off
title('average')
legend('measured','bilstm est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([bilstmPtestMax180'; bilstmOutf180' ],height180);
[bilstmperf180]


%%
perfallbilstm=[bilstmperf50; bilstmperf60; bilstmperf70; bilstmperf80; bilstmperf90; bilstmperf100; bilstmperf110; bilstmperf120; bilstmperf130; bilstmperf140; bilstmperf150; bilstmperf160; bilstmperf170; bilstmperf180]