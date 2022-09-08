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
% LSTM WSE

LSTMP50 = traininginput';
LSTMY50 = trainingtarget';

LSTMP50V = Valinput';
LSTMY50V = Valtarget';
LSTMtarget50V=LSTMY50V'*maxx;

LSTMPtest50 = testinginput';
LSTMYtest50 = testingtarget';
LSTMtestingtarget50=LSTMYtest50'*maxx;

numiter=200;
numhid=5;
miniBatchSize = 8000;
numFeatures=inputsize+(nex-1);
numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%layers = [ ...
%    sequenceInputLayer(numFeatures)
%    lstmLayer(numHiddenUnits,'OutputMode','sequence')
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

net = trainNetwork(LSTMP50,LSTMY50,layers,options);
outval = predict(net,LSTMP50,'MiniBatchSize',1);

outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP50V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf50V=outvalV'*maxx;
LSTMmseV50=mse(outvalmaxV,LSTMtarget50V);
LSTMmapeV50=mape(outvalmaxV,LSTMtarget50V);
LSTMmbeV50=mbe(outvalmaxV,LSTMtarget50V);
LSTMr2V50=rsquare(outvalmaxV,LSTMtarget50V);
LSTMperf50V=[LSTMmseV50 LSTMmapeV50 LSTMmbeV50 LSTMr2V50];
[LSTMperf50V]
%
outvaltest = predict(net,LSTMPtest50,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
LSTMOutf50=outvaltestmax;
LSTMtestingtargetmax50=testingtarget50;
LSTMmsetest50=mse(LSTMOutf50,LSTMtestingtargetmax50);
LSTMmapetest50=mape(LSTMOutf50,LSTMtestingtargetmax50);
LSTMmbetest50=mbe(LSTMOutf50,LSTMtestingtargetmax50);
LSTMr2test50=rsquare(LSTMOutf50,LSTMtestingtargetmax50);

LSTMperf50=[LSTMmsetest50 LSTMmapetest50 LSTMmbetest50 LSTMr2test50];
LSTMOutf50train=outvalmax;
LSTMPtestMax50=LSTMPtest50'*maxx;
perfall=[mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanLSTM50=mean([LSTMPtestMax50'; LSTMOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanLSTM50,height50,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

figure
plot([LSTMPtestMax50'; LSTMOutf50' ],height50);

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[LSTMperf50]

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
% LSTM WSE
%
LSTMP60 = [LSTMP50; LSTMOutf50train'/maxx];
LSTMY60 = trainingtarget';

LSTMP60V = [LSTMP50V; LSTMOutf50V'/maxx];
LSTMY60V = Valtarget';
LSTMtarget60V=LSTMY60V'*maxx;

LSTMPtest60 = [LSTMPtest50; LSTMOutf50'/maxx];
LSTMYtest60 = testingtarget';
LSTMtestingtarget60=LSTMYtest60'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP60,LSTMY60,layers,options);
outval = predict(net,LSTMP60,'MiniBatchSize',1);

LSTMOutf60train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP60V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf60V=outvalV'*maxx;
LSTMmseV60=mse(outvalmaxV,LSTMtarget60V);
LSTMmapeV60=mape(outvalmaxV,LSTMtarget60V);
LSTMmbeV60=mbe(outvalmaxV,LSTMtarget60V);
LSTMr2V60=rsquare(outvalmaxV,LSTMtarget60V);
LSTMperf60V=[LSTMmseV60 LSTMmapeV60 LSTMmbeV60 LSTMr2V60];
[LSTMperf60V]



outvaltest = predict(net,LSTMPtest60,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
LSTMOutf60=outvaltestmax;
LSTMmsetest60=mse(LSTMOutf60,testingtarget60);
LSTMmapetest60=mape(LSTMOutf60,testingtarget60);
LSTMmbetest60=mbe(LSTMOutf60,testingtarget60);
LSTMr2test60=rsquare(LSTMOutf60,testingtarget60);
LSTMperf60=[LSTMmsetest60 LSTMmapetest60 LSTMmbetest60 LSTMr2test60];
LSTMPtestMax60=LSTMPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanLSTM60=[meanLSTM50 mean(LSTMOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanLSTM60,height60,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax60'; LSTMOutf60' ],height60);
[LSTMperf60]

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
% LSTM WSE
%
LSTMP70 = [LSTMP60; LSTMOutf60train'/maxx];
LSTMY70 = trainingtarget';

LSTMP70V = [LSTMP60V; LSTMOutf60V'/maxx];
LSTMY70V = Valtarget';
LSTMtarget70V=LSTMY70V'*maxx;

LSTMPtest70 = [LSTMPtest60; LSTMOutf60'/maxx];
LSTMYtest70 = testingtarget';
LSTMtestingtarget70=LSTMYtest70'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP70,LSTMY70,layers,options);
outval = predict(net,LSTMP70,'MiniBatchSize',1);
outvalmax=outval'*maxx;

outval = predict(net,LSTMP70,'MiniBatchSize',1);
LSTMOutf70train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP70V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf70V=outvalV'*maxx;
LSTMmseV70=mse(outvalmaxV,LSTMtarget70V);
LSTMmapeV70=mape(outvalmaxV,LSTMtarget70V);
LSTMmbeV70=mbe(outvalmaxV,LSTMtarget70V);
LSTMr2V70=rsquare(outvalmaxV,LSTMtarget70V);
LSTMperf70V=[LSTMmseV70 LSTMmapeV70 LSTMmbeV70 LSTMr2V70];
[LSTMperf70V]



outvaltest = predict(net,LSTMPtest70,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
LSTMOutf70=outvaltestmax;
LSTMmsetest70=mse(LSTMOutf70,testingtarget70);
LSTMmapetest70=mape(LSTMOutf70,testingtarget70);
LSTMmbetest70=mbe(LSTMOutf70,testingtarget70);
LSTMr2test70=rsquare(LSTMOutf70,testingtarget70);
LSTMperf70=[LSTMmsetest70 LSTMmapetest70 LSTMmbetest70 LSTMr2test70];
LSTMPtestMax70=LSTMPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanLSTM70=[meanLSTM60 mean(LSTMOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanLSTM70,height70,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax70'; LSTMOutf70' ],height70);
[LSTMperf70]

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
% LSTM WSE
%
LSTMP80 = [LSTMP70; LSTMOutf70train'/maxx];
LSTMY80 = trainingtarget';

LSTMP80V = [LSTMP70V; LSTMOutf70V'/maxx];
LSTMY80V = Valtarget';
LSTMtarget80V=LSTMY80V'*maxx;

LSTMPtest80 = [LSTMPtest70; LSTMOutf70'/maxx];
LSTMYtest80 = testingtarget';
LSTMtestingtarget80=LSTMYtest80'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP80,LSTMY80,layers,options);
outval = predict(net,LSTMP80,'MiniBatchSize',1);
outvalmax=outval'*maxx;


LSTMOutf80train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP80V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf80V=outvalV'*maxx;
LSTMmseV80=mse(outvalmaxV,LSTMtarget80V);
LSTMmapeV80=mape(outvalmaxV,LSTMtarget80V);
LSTMmbeV80=mbe(outvalmaxV,LSTMtarget80V);
LSTMr2V80=rsquare(outvalmaxV,LSTMtarget80V);
LSTMperf80V=[LSTMmseV80 LSTMmapeV80 LSTMmbeV80 LSTMr2V80];
[LSTMperf80V]



outvaltest = predict(net,LSTMPtest80,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
LSTMOutf80=outvaltestmax;
LSTMmsetest80=mse(LSTMOutf80,testingtarget80);
LSTMmapetest80=mape(LSTMOutf80,testingtarget80);
LSTMmbetest80=mbe(LSTMOutf80,testingtarget80);
LSTMr2test80=rsquare(LSTMOutf80,testingtarget80);
LSTMperf80=[LSTMmsetest80 LSTMmapetest80 LSTMmbetest80 LSTMr2test80];
LSTMPtestMax80=LSTMPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanLSTM80=[meanLSTM70 mean(LSTMOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanLSTM80,height80,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax80'; LSTMOutf80' ],height80);
[LSTMperf80]

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
% LSTM WSE
%
LSTMP90 = [LSTMP80; LSTMOutf80train'/maxx];
LSTMY90 = trainingtarget';

LSTMP90V = [LSTMP80V; LSTMOutf80V'/maxx];
LSTMY90V = Valtarget';
LSTMtarget90V=LSTMY90V'*maxx;

LSTMPtest90 = [LSTMPtest80; LSTMOutf80'/maxx];
LSTMYtest90 = testingtarget';
LSTMtestingtarget90=LSTMYtest90'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP90,LSTMY90,layers,options);
outval = predict(net,LSTMP90,'MiniBatchSize',1);
outvalmax=outval'*maxx;

LSTMOutf90train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP90V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf90V=outvalV'*maxx;
LSTMmseV90=mse(outvalmaxV,LSTMtarget90V);
LSTMmapeV90=mape(outvalmaxV,LSTMtarget90V);
LSTMmbeV90=mbe(outvalmaxV,LSTMtarget90V);
LSTMr2V90=rsquare(outvalmaxV,LSTMtarget90V);
LSTMperf90V=[LSTMmseV90 LSTMmapeV90 LSTMmbeV90 LSTMr2V90];
[LSTMperf90V]



outvaltest = predict(net,LSTMPtest90,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
LSTMOutf90=outvaltestmax;
LSTMmsetest90=mse(LSTMOutf90,testingtarget90);
LSTMmapetest90=mape(LSTMOutf90,testingtarget90);
LSTMmbetest90=mbe(LSTMOutf90,testingtarget90);
LSTMr2test90=rsquare(LSTMOutf90,testingtarget90);
LSTMperf90=[LSTMmsetest90 LSTMmapetest90 LSTMmbetest90 LSTMr2test90];
LSTMPtestMax90=LSTMPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanLSTM90=[meanLSTM80 mean(LSTMOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanLSTM90,height90,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax90'; LSTMOutf90' ],height90);
[LSTMperf90]

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
% LSTM WSE
%
LSTMP100 = [LSTMP90; LSTMOutf90train'/maxx];
LSTMY100 = trainingtarget';

LSTMP100V = [LSTMP90V; LSTMOutf90V'/maxx];
LSTMY100V = Valtarget';
LSTMtarget100V=LSTMY100V'*maxx;

LSTMPtest100 = [LSTMPtest90; LSTMOutf90'/maxx];
LSTMYtest100 = testingtarget';
LSTMtestingtarget100=LSTMYtest100'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP100,LSTMY100,layers,options);

outval = predict(net,LSTMP100,'MiniBatchSize',1);
LSTMOutf100train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP100V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf100V=outvalV'*maxx;
LSTMmseV100=mse(outvalmaxV,LSTMtarget100V);
LSTMmapeV100=mape(outvalmaxV,LSTMtarget100V);
LSTMmbeV100=mbe(outvalmaxV,LSTMtarget100V);
LSTMr2V100=rsquare(outvalmaxV,LSTMtarget100V);
LSTMperf100V=[LSTMmseV100 LSTMmapeV100 LSTMmbeV100 LSTMr2V100];
[LSTMperf100V]



outvaltest = predict(net,LSTMPtest100,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
LSTMOutf100=outvaltestmax;
LSTMmsetest100=mse(LSTMOutf100,testingtarget100);
LSTMmapetest100=mape(LSTMOutf100,testingtarget100);
LSTMmbetest100=mbe(LSTMOutf100,testingtarget100);
LSTMr2test100=rsquare(LSTMOutf100,testingtarget100);
LSTMperf100=[LSTMmsetest100 LSTMmapetest100 LSTMmbetest100 LSTMr2test100];
LSTMPtestMax100=LSTMPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanLSTM100=[meanLSTM90 mean(LSTMOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanLSTM100,height100,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax100'; LSTMOutf100' ],height100);
[LSTMperf100]

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
% LSTM WSE
%
LSTMP110 = [LSTMP100; LSTMOutf100train'/maxx];
LSTMY110 = trainingtarget';

LSTMP110V = [LSTMP100V; LSTMOutf100V'/maxx];
LSTMY110V = Valtarget';
LSTMtarget110V=LSTMY110V'*maxx;

LSTMPtest110 = [LSTMPtest100; LSTMOutf100'/maxx];
LSTMYtest110 = testingtarget';
LSTMtestingtarget110=LSTMYtest110'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP110,LSTMY110,layers,options);
outval = predict(net,LSTMP110,'MiniBatchSize',1);



LSTMOutf110train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP110V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf110V=outvalV'*maxx;
LSTMmseV110=mse(outvalmaxV,LSTMtarget110V);
LSTMmapeV110=mape(outvalmaxV,LSTMtarget110V);
LSTMmbeV110=mbe(outvalmaxV,LSTMtarget110V);
LSTMr2V110=rsquare(outvalmaxV,LSTMtarget110V);
LSTMperf110V=[LSTMmseV110 LSTMmapeV110 LSTMmbeV110 LSTMr2V110];
[LSTMperf110V]



outvaltest = predict(net,LSTMPtest110,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
LSTMOutf110=outvaltestmax;
LSTMmsetest110=mse(LSTMOutf110,testingtarget110);
LSTMmapetest110=mape(LSTMOutf110,testingtarget110);
LSTMmbetest110=mbe(LSTMOutf110,testingtarget110);
LSTMr2test110=rsquare(LSTMOutf110,testingtarget110);
LSTMperf110=[LSTMmsetest110 LSTMmapetest110 LSTMmbetest110 LSTMr2test110];
LSTMPtestMax110=LSTMPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanLSTM110=[meanLSTM100 mean(LSTMOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanLSTM110,height110,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax110'; LSTMOutf110' ],height110);
[LSTMperf110]

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
% LSTM WSE
%
LSTMP120 = [LSTMP110; LSTMOutf110train'/maxx];
LSTMY120 = trainingtarget';

LSTMP120V = [LSTMP110V; LSTMOutf110V'/maxx];
LSTMY120V = Valtarget';
LSTMtarget120V=LSTMY120V'*maxx;

LSTMPtest120 = [LSTMPtest110; LSTMOutf110'/maxx];
LSTMYtest120 = testingtarget';
LSTMtestingtarget120=LSTMYtest120'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP120,LSTMY120,layers,options);
outval = predict(net,LSTMP120,'MiniBatchSize',1);

outval = predict(net,LSTMP120,'MiniBatchSize',1);
LSTMOutf120train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP120V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf120V=outvalV'*maxx;
LSTMmseV120=mse(outvalmaxV,LSTMtarget120V);
LSTMmapeV120=mape(outvalmaxV,LSTMtarget120V);
LSTMmbeV120=mbe(outvalmaxV,LSTMtarget120V);
LSTMr2V120=rsquare(outvalmaxV,LSTMtarget120V);
LSTMperf120V=[LSTMmseV120 LSTMmapeV120 LSTMmbeV120 LSTMr2V120];
[LSTMperf120V]



outvaltest = predict(net,LSTMPtest120,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
LSTMOutf120=outvaltestmax;
LSTMmsetest120=mse(LSTMOutf120,testingtarget120);
LSTMmapetest120=mape(LSTMOutf120,testingtarget120);
LSTMmbetest120=mbe(LSTMOutf120,testingtarget120);
LSTMr2test120=rsquare(LSTMOutf120,testingtarget120);
LSTMperf120=[LSTMmsetest120 LSTMmapetest120 LSTMmbetest120 LSTMr2test120];
LSTMPtestMax120=LSTMPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanLSTM120=[meanLSTM110 mean(LSTMOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanLSTM120,height120,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax120'; LSTMOutf120' ],height120);
[LSTMperf120]
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
% LSTM WSE
%
LSTMP130 = [LSTMP120; LSTMOutf120train'/maxx];
LSTMY130 = trainingtarget';

LSTMP130V = [LSTMP120V; LSTMOutf120V'/maxx];
LSTMY130V = Valtarget';
LSTMtarget130V=LSTMY130V'*maxx;

LSTMPtest130 = [LSTMPtest120; LSTMOutf120'/maxx];
LSTMYtest130 = testingtarget';
LSTMtestingtarget130=LSTMYtest130'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP130,LSTMY130,layers,options);
outval = predict(net,LSTMP130,'MiniBatchSize',1);

outval = predict(net,LSTMP130,'MiniBatchSize',1);
LSTMOutf130train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP130V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf130V=outvalV'*maxx;
LSTMmseV130=mse(outvalmaxV,LSTMtarget130V);
LSTMmapeV130=mape(outvalmaxV,LSTMtarget130V);
LSTMmbeV130=mbe(outvalmaxV,LSTMtarget130V);
LSTMr2V130=rsquare(outvalmaxV,LSTMtarget130V);
LSTMperf130V=[LSTMmseV130 LSTMmapeV130 LSTMmbeV130 LSTMr2V130];
[LSTMperf130V]



outvaltest = predict(net,LSTMPtest130,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
LSTMOutf130=outvaltestmax;
LSTMmsetest130=mse(LSTMOutf130,testingtarget130);
LSTMmapetest130=mape(LSTMOutf130,testingtarget130);
LSTMmbetest130=mbe(LSTMOutf130,testingtarget130);
LSTMr2test130=rsquare(LSTMOutf130,testingtarget130);
LSTMperf130=[LSTMmsetest130 LSTMmapetest130 LSTMmbetest130 LSTMr2test130];
LSTMPtestMax130=LSTMPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanLSTM130=[meanLSTM120 mean(LSTMOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanLSTM130,height130,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax130'; LSTMOutf130' ],height130);
[LSTMperf130]
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
% LSTM WSE
%
LSTMP140 = [LSTMP130; LSTMOutf130train'/maxx];
LSTMY140 = trainingtarget';

LSTMP140V = [LSTMP130V; LSTMOutf130V'/maxx];
LSTMY140V = Valtarget';
LSTMtarget140V=LSTMY140V'*maxx;

LSTMPtest140 = [LSTMPtest130; LSTMOutf130'/maxx];
LSTMYtest140 = testingtarget';
LSTMtestingtarget140=LSTMYtest140'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP140,LSTMY140,layers,options);
outval = predict(net,LSTMP140,'MiniBatchSize',1);

outval = predict(net,LSTMP140,'MiniBatchSize',1);
LSTMOutf140train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP140V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf140V=outvalV'*maxx;
LSTMmseV140=mse(outvalmaxV,LSTMtarget140V);
LSTMmapeV140=mape(outvalmaxV,LSTMtarget140V);
LSTMmbeV140=mbe(outvalmaxV,LSTMtarget140V);
LSTMr2V140=rsquare(outvalmaxV,LSTMtarget140V);
LSTMperf140V=[LSTMmseV140 LSTMmapeV140 LSTMmbeV140 LSTMr2V140];
[LSTMperf140V]



outvaltest = predict(net,LSTMPtest140,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
LSTMOutf140=outvaltestmax;
LSTMmsetest140=mse(LSTMOutf140,testingtarget140);
LSTMmapetest140=mape(LSTMOutf140,testingtarget140);
LSTMmbetest140=mbe(LSTMOutf140,testingtarget140);
LSTMr2test140=rsquare(LSTMOutf140,testingtarget140);
LSTMperf140=[LSTMmsetest140 LSTMmapetest140 LSTMmbetest140 LSTMr2test140];
LSTMPtestMax140=LSTMPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanLSTM140=[meanLSTM130 mean(LSTMOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanLSTM140,height140,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax140'; LSTMOutf140' ],height140);
[LSTMperf140]

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
% LSTM WSE
%
LSTMP150 = [LSTMP140; LSTMOutf140train'/maxx];
LSTMY150 = trainingtarget';

LSTMP150V = [LSTMP140V; LSTMOutf140V'/maxx];
LSTMY150V = Valtarget';
LSTMtarget150V=LSTMY150V'*maxx;

LSTMPtest150 = [LSTMPtest140; LSTMOutf140'/maxx];
LSTMYtest150 = testingtarget';
LSTMtestingtarget150=LSTMYtest150'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP150,LSTMY150,layers,options);
outval = predict(net,LSTMP150,'MiniBatchSize',1);

outval = predict(net,LSTMP150,'MiniBatchSize',1);
LSTMOutf150train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP150V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf150V=outvalV'*maxx;
LSTMmseV150=mse(outvalmaxV,LSTMtarget150V);
LSTMmapeV150=mape(outvalmaxV,LSTMtarget150V);
LSTMmbeV150=mbe(outvalmaxV,LSTMtarget150V);
LSTMr2V150=rsquare(outvalmaxV,LSTMtarget150V);
LSTMperf150V=[LSTMmseV150 LSTMmapeV150 LSTMmbeV150 LSTMr2V150];
[LSTMperf150V]



outvaltest = predict(net,LSTMPtest150,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
LSTMOutf150=outvaltestmax;
LSTMmsetest150=mse(LSTMOutf150,testingtarget150);
LSTMmapetest150=mape(LSTMOutf150,testingtarget150);
LSTMmbetest150=mbe(LSTMOutf150,testingtarget150);
LSTMr2test150=rsquare(LSTMOutf150,testingtarget150);
LSTMperf150=[LSTMmsetest150 LSTMmapetest150 LSTMmbetest150 LSTMr2test150];
LSTMPtestMax150=LSTMPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanLSTM150=[meanLSTM140 mean(LSTMOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanLSTM150,height150,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax150'; LSTMOutf150' ],height150);
[LSTMperf150]

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
% LSTM WSE
%
LSTMP160 = [LSTMP150; LSTMOutf150train'/maxx];
LSTMY160 = trainingtarget';

LSTMP160V = [LSTMP150V; LSTMOutf150V'/maxx];
LSTMY160V = Valtarget';
LSTMtarget160V=LSTMY160V'*maxx;

LSTMPtest160 = [LSTMPtest150; LSTMOutf150'/maxx];
LSTMYtest160 = testingtarget';
LSTMtestingtarget160=LSTMYtest160'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP160,LSTMY160,layers,options);
outval = predict(net,LSTMP160,'MiniBatchSize',1);

outval = predict(net,LSTMP160,'MiniBatchSize',1);
LSTMOutf160train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP160V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf160V=outvalV'*maxx;
LSTMmseV160=mse(outvalmaxV,LSTMtarget160V);
LSTMmapeV160=mape(outvalmaxV,LSTMtarget160V);
LSTMmbeV160=mbe(outvalmaxV,LSTMtarget160V);
LSTMr2V160=rsquare(outvalmaxV,LSTMtarget160V);
LSTMperf160V=[LSTMmseV160 LSTMmapeV160 LSTMmbeV160 LSTMr2V160];
[LSTMperf160V]



outvaltest = predict(net,LSTMPtest160,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
LSTMOutf160=outvaltestmax;
LSTMmsetest160=mse(LSTMOutf160,testingtarget160);
LSTMmapetest160=mape(LSTMOutf160,testingtarget160);
LSTMmbetest160=mbe(LSTMOutf160,testingtarget160);
LSTMr2test160=rsquare(LSTMOutf160,testingtarget160);
LSTMperf160=[LSTMmsetest160 LSTMmapetest160 LSTMmbetest160 LSTMr2test160];
LSTMPtestMax160=LSTMPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanLSTM160=[meanLSTM150 mean(LSTMOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanLSTM160,height160,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax160'; LSTMOutf160' ],height160);
[LSTMperf160]

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
% LSTM WSE
%
LSTMP170 = [LSTMP160; LSTMOutf160train'/maxx];
LSTMY170 = trainingtarget';

LSTMP170V = [LSTMP160V; LSTMOutf160V'/maxx];
LSTMY170V = Valtarget';
LSTMtarget170V=LSTMY170V'*maxx;

LSTMPtest170 = [LSTMPtest160; LSTMOutf160'/maxx];
LSTMYtest170 = testingtarget';
LSTMtestingtarget170=LSTMYtest170'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP170,LSTMY170,layers,options);
outval = predict(net,LSTMP170,'MiniBatchSize',1);

outval = predict(net,LSTMP170,'MiniBatchSize',1);
LSTMOutf170train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP170V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf170V=outvalV'*maxx;
LSTMmseV170=mse(outvalmaxV,LSTMtarget170V);
LSTMmapeV170=mape(outvalmaxV,LSTMtarget170V);
LSTMmbeV170=mbe(outvalmaxV,LSTMtarget170V);
LSTMr2V170=rsquare(outvalmaxV,LSTMtarget170V);
LSTMperf170V=[LSTMmseV170 LSTMmapeV170 LSTMmbeV170 LSTMr2V170];
[LSTMperf170V]



outvaltest = predict(net,LSTMPtest170,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
LSTMOutf170=outvaltestmax;
LSTMmsetest170=mse(LSTMOutf170,testingtarget170);
LSTMmapetest170=mape(LSTMOutf170,testingtarget170);
LSTMmbetest170=mbe(LSTMOutf170,testingtarget170);
LSTMr2test170=rsquare(LSTMOutf170,testingtarget170);
LSTMperf170=[LSTMmsetest170 LSTMmapetest170 LSTMmbetest170 LSTMr2test170];
LSTMPtestMax170=LSTMPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanLSTM170=[meanLSTM160 mean(LSTMOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanLSTM170,height170,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax170'; LSTMOutf170' ],height170);
[LSTMperf170]

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
% LSTM WSE
%
LSTMP180 = [LSTMP170; LSTMOutf170train'/maxx];
LSTMY180 = trainingtarget';

LSTMP180V = [LSTMP170V; LSTMOutf170V'/maxx];
LSTMY180V = Valtarget';
LSTMtarget180V=LSTMY180V'*maxx;

LSTMPtest180 = [LSTMPtest170; LSTMOutf170'/maxx];
LSTMYtest180 = testingtarget';
LSTMtestingtarget180=LSTMYtest180'*maxx;

numFeatures=inputsize+(nex-1);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
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

net = trainNetwork(LSTMP180,LSTMY180,layers,options);
outval = predict(net,LSTMP180,'MiniBatchSize',1);

outval = predict(net,LSTMP180,'MiniBatchSize',1);
LSTMOutf180train=outval'*maxx;
outvalmax=outval'*maxx;

trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvalV = predict(net,LSTMP180V,'MiniBatchSize',1);
outvalmaxV=outvalV'*maxx;
LSTMOutf180V=outvalV'*maxx;
LSTMmseV180=mse(outvalmaxV,LSTMtarget180V);
LSTMmapeV180=mape(outvalmaxV,LSTMtarget180V);
LSTMmbeV180=mbe(outvalmaxV,LSTMtarget180V);
LSTMr2V180=rsquare(outvalmaxV,LSTMtarget180V);
LSTMperf180V=[LSTMmseV180 LSTMmapeV180 LSTMmbeV180 LSTMr2V180];
[LSTMperf180V]



outvaltest = predict(net,LSTMPtest180,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
LSTMOutf180=outvaltestmax;
LSTMmsetest180=mse(LSTMOutf180,testingtarget180);
LSTMmapetest180=mape(LSTMOutf180,testingtarget180);
LSTMmbetest180=mbe(LSTMOutf180,testingtarget180);
LSTMr2test180=rsquare(LSTMOutf180,testingtarget180);
LSTMperf180=[LSTMmsetest180 LSTMmapetest180 LSTMmbetest180 LSTMr2test180];
LSTMPtestMax180=LSTMPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanLSTM180=[meanLSTM170 mean(LSTMOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanLSTM180,height180,'-.g');

hold off
title('average')
legend('measured','LSTM est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
figure
plot([LSTMPtestMax180'; LSTMOutf180' ],height180);
[LSTMperf180]


%%
perfallLSTM=[LSTMperf50; LSTMperf60; LSTMperf70; LSTMperf80; LSTMperf90; LSTMperf100; LSTMperf110; LSTMperf120; LSTMperf130; LSTMperf140; LSTMperf150; LSTMperf160; LSTMperf170; LSTMperf180]