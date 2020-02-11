function [g,buyquery,data,data_pred,tp,data_new_pred,tc,cost,train_data,Y_train_data,train_before]=softmax_cluster_K(class,label,k,K,g,buyquery,Center,data,data_pred,tp,data_new_pred,tc,train_data,Y_train_data,train_before)
%% ���������ǽ���softmax����
%% %%%%%����%%%%%
%%class:����������
%%label:ÿ�ض�Ӧԭʼ���ݼ��еı��
%%k:��ǩ����
%%K���������ܴ���
%%g:��ʼֵΪ1
%%buyquery:��¼����ı�ǩ���
%%Center:��������
%%data:ԭʼ���ݼ�
%% %%%%%���%%%%%
%%reblock2:softmaxԤ����Ӧ��ŵı�ǩ����
[n,c]=size(data);
X=data(:,1:c-1);
Y=data(:,c);
minDist_index=cell(K,1);
reblock1=zeros(n,1);
v=1;
vvv=1;
minDist_index1=[];
%class_data=zeros(n,c);
%% ��ѭ����������ÿ��ѡȡ��������������ݵ���Ϊѵ����
for h=1:K
data1=class{h,1};%��������
[z, d] = size(data1);  %���ݼ�n�У�dά
x= data1(:,1:d-1);  %%�ޱ�ǩ����
 %Y = data1(:,d);      %%��ʵ��ǩ
Dist=pdist2(x,Center(h,:),'euclidean');
%%  һ�����ݼ�ѡ�㷽ʽ%%%%
 minDist_index{h,1}=min(find(Dist==min(Dist)));%%�ҵ����������class�еı��
%%  ͼ�����ݼ�ѡ�㷽ʽ%%%%%
% [~,Index1]=sort(Dist,1,'ascend');
% minDist_index{h,1}=Index1(1:floor(0.35*z));
%minDist_index{h,1}=min(find(Dist==min(Dist)));
%%  
minDist_index{h,1}=label{h,1}(minDist_index{h,1},1);%%�ҵ��������ԭʼ���ݼ��еı��
%% ѭ���������ǽ�ÿ�ر�����ηŵ�ͬһ��������
for i=1:z
    %class_data(v,:)=class{h,1}(i,:);
     reblock1(v,1)=label{h,1}(i,1);
    v=v+1;
end
[vv,~]=size(minDist_index{h,1});
for ii=1:vv
    minDist_index1(vvv,1)=minDist_index{h,1}(ii,1);
    vvv=vvv+1;
end

end
%labels = []��
%X = class_data(:,1:d-1);
reblock1=sortrows(reblock1,1);%%�����������б�ţ�������ԭʼ���ݼ��ı�Ž���������1��2��3��������
AB=minDist_index1;

lambda = 1e-4;
%% ÿһ��label�����ͬ�ȴ�С��ȫΪ0��һ��
A=zeros(n,1);
reblock1=[reblock1,A];%%��reblock1�������1�У����ڴ��softmaxԤ�����Ӧλ�õı�ǩ
% Weight decay parameter
%% %%%%%%%%%%%%%%%%%%%%��ʼ��%%%%%%%%%%%%%%
isLabeled = zeros(1,n);
Labels = zeros(1,n);
%% %%%%%%%%%%%%%%%%%%%%���������%%%%%%%%%%%%%%%%%

%% %%%%%%%%5%%%ÿ����������һ��x0=1%%%%
oneArray = ones(1,n);
X = [oneArray; X']; %ÿ��������Ҫ����һ��x0=1;
%X = mapminmax(X);
%%   %%%%%%%%% ������ʼѵ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%ͨ��������ѡ��K�������%%%%% 
%     case 1
%  ratio = K;
%  trainingSet = desInd(1:1:ratio);
 %% ѡȡѵ����

 %% ���û����ͬ�ģ����Ծ������ĵ�����ĵ���Ϊѵ����������ͬ�ľ���֮ǰ�������Ϊѵ����

      p=AB;
     trainingSet=p;%%ѵ����
     q=size(p,1);
%% ѭ�������ǽ�ѵ�������η���buyquery��
     for i=1:q
         trainingnum=trainingSet(i,1);
      buyquery(g,:)=data(trainingnum,:);
%       train_data(g,:)=X(:,trainingSet(i,1));
     g=g+1;
     end
    reblock1(trainingSet,end)=Y(trainingSet);%%��ѵ�����ı�ǩ�ŵ�reblock1��Ӧ��ŵ����һ��
%  end
% buy_data_num=g-1;
%% ѵ����
% train_data=data(trainingSet,:);

%% %%%ѵ������
trainingInput = [X(:,trainingSet) train_data];     %%%%%%%ѵ��������ǰһ��ȥ���������Ĵ��е�ѵ����ȥ��������ʣ��ļӵ����ε�ѵ�����У�
labels =[Y(trainingSet);Y_train_data];              %%%%%%%ѵ������ǩ��ͬѵ����һ����������һ��ѵ������ǩ��
Labels(trainingSet) = Y(trainingSet);
Labels = Labels';
isLabeled(trainingSet) = 1;
%%  %%%%%%%%% ѵ��theta %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = 0.0005 * randn(d, k);     % ���������ʼtheta
[cost, grad] = softmax_regression_vec(theta,trainingInput,labels,lambda);
options.maxIter = 100;
softmaxModel = softmaxTrain(d, k, lambda, trainingInput,labels, options);
theta = softmaxModel.optTheta;  %���»��theta����

%% STEP : Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pred = zeros(1, size(data, 2));
%[~,pred]= max(theta'*TestInput);%theta'*data���������ͼ5��ĳһ������softmax���ֵ���������ĳһ�����
%ֵ�ǵȼ۵ģ���Ϊÿһ�г���ͬһ����ĸ�Ͳ�����һ���ģ�����exp��.����������������ֻ����������ֵ���ɡ�
% a=1;

cost=0;
while sum(isLabeled)~=n
    for i = 1:n
        if (isLabeled(i) == 1)            %%ʵ������Ѵ�����������ʵ��
            continue;
        end
         tempIncomData = X(:,i);        % �����걸���ݼ��ĵ���ʵ����ֵ��tempIncomData���Ӷ����ı�ԭ���걸���ݼ�
         
%          switch initial_num
%              case 1
        predProbability(:,i) = theta'*tempIncomData;                           %%����z1��z2��z3
        M=max( predProbability(:,i));
        predProbability2(:,i) = exp(predProbability(:,i)-M);                      %%����e^z1��e^z2��e^z3
        sumProb(:,i) = sum(predProbability2(:,i)); 
         completeP(:,i) = (predProbability2(:,i))./sumProb(i);
         cost=cost+2*(1-max(completeP(:,i)));
%          if (((1-completeP(i))*2)<1)%������С�ڹ��������ֱ��Ԥ��
%              [~,pred]= max(theta'*tempIncomData);

             [~,pred]= max(completeP(:,i)); 
             Labels(i) = pred;
             reblock1(i,end)=Labels(i);%��Ԥ��ı�ǩ����label���һ��
             isLabeled(i) = 1;
%
         
     end
end
buy_data_num=size(trainingInput,2);
cost=cost+buy_data_num;
reblock2=reblock1;
class_pred1=cell(K,1);
% class_pred2=cell(K,1);
purity=zeros(K,1);
% purity2=zeros(K,1);
tb=1;
train_data=[data(trainingSet,:);train_before];%%��ǰһ�ε�ѵ��������
% train_before=data(trainingSet,:);
%% KNN1(�����)�㷨
KNN_train_data=train_data(:,1:end-1);
KNN_train_label=train_data(:,end);
num_neighbors=1;
KNN_test_data=data(:,1:end-1);
Factor = ClassificationKNN.fit(KNN_train_data, KNN_train_label, 'NumNeighbors', num_neighbors);
KNN_predict_label = predict(Factor, KNN_test_data);
%% 
train_out=[];
data=[];
qq=1;
for h=1:K
    [z,~]=size(class{h,1});
    class_pred1{h,1}=reblock2(label{h,1},:);
   A=class_pred1{h,1}(:,end);
   b=1:max(A);
   max_class_num=max(histc((A),b));%%����Ԥ�����Ԥ���������һ�������
   purity(h,1)=max_class_num./z;%%����ÿһ�ص�Ԥ�ⴿ��
   %% ����KNN1������ڣ�ÿһ�صġ����ȡ�
   class_pred2{h,1}=KNN_predict_label(label{h,1},:);
   B=class_pred2{h,1}(:,end);
   c=1:max(B);
   KNN_max_class_num=max(histc((B),c));%%����Ԥ�����Ԥ���������һ�������
   purity2(h,1)=KNN_max_class_num./z;%%����ÿһ�ص�Ԥ�ⴿ��
   if (purity(h,1)==1)&&(purity2(h,1)==1)

      for i=1:z
           if A(i,:)==B(i,:)
         
           data_pred(tp,:)=[class{h,1}(i,:) A(i,:)];%%�����㴿�����������ݷŽ�һ�������У����ٵ���
           tp=tp+1;
           else
               data(tb,:)=class{h,1}(i,:);%%���е��������ݣ������������ݣ�
               data_new_pred{tc,1}(tb,:)=[class{h,1}(i,:) A(i,:)];%%����ÿһ�ε���ǰ���������ݣ��Ѿ�����softmaxԤ�⣬������Ԥ���ǩ��
               tb=tb+1;
          end
       end
       train_out(qq,1)=h;
       qq=qq+1;
       
   else
       for i=1:z
         data(tb,:)=class{h,1}(i,:);%%���е��������ݣ������������ݣ�
%          data_new(tb,:)=[class{h,1}(i,:) A(i,:)];%%��Ҫ���еĵ��������ݺ����һ�У���Ӧ��softmaxԤ��ı�ǩ���������ڱȽϴ��۵�ʱ���ҵ��������ʱȡ��
         data_new_pred{tc,1}(tb,:)=[class{h,1}(i,:) A(i,:)];%%����ÿһ�ε���ǰ���������ݣ��Ѿ�����softmaxԤ�⣬������Ԥ���ǩ��
         tb=tb+1;
         
       end
       
   end
end
%% ������Ҫ��Ĵ���ѡ��ѵ����ȥ��
if isempty(train_out)
else
train_data(train_out,:)=[];
end
tc=tc+1;
train_before=train_data;
Y_train_data=train_data(:,end);
pp=size(train_data,1);
train_data=[ones(1,pp);(train_data(:,1:end-1))'];%%��һ��softmax��Ҫ�����ѵ����

 