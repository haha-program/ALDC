function [g,buyquery,data,data_pred,tp,data_new_pred,tc,cost,train_data,Y_train_data,train_before]=softmax_cluster_K(class,label,k,K,g,buyquery,Center,data,data_pred,tp,data_new_pred,tc,train_data,Y_train_data,train_before)
%% 函数作用是进行softmax分类
%% %%%%%输入%%%%%
%%class:聚类块的数据
%%label:每簇对应原始数据集中的标号
%%k:标签总数
%%K：聚类后的总簇数
%%g:初始值为1
%%buyquery:记录购买的标签标号
%%Center:聚类中心
%%data:原始数据集
%% %%%%%输出%%%%%
%%reblock2:softmax预测后对应标号的标签数组
[n,c]=size(data);
X=data(:,1:c-1);
Y=data(:,c);
minDist_index=cell(K,1);
reblock1=zeros(n,1);
v=1;
vvv=1;
minDist_index1=[];
%class_data=zeros(n,c);
%% 此循环的作用是每簇选取离中心最近的数据点作为训练集
for h=1:K
data1=class{h,1};%输入数据
[z, d] = size(data1);  %数据集n行，d维
x= data1(:,1:d-1);  %%无标签数据
 %Y = data1(:,d);      %%真实标签
Dist=pdist2(x,Center(h,:),'euclidean');
%%  一般数据集选点方式%%%%
 minDist_index{h,1}=min(find(Dist==min(Dist)));%%找到最近距离在class中的标号
%%  图像数据集选点方式%%%%%
% [~,Index1]=sort(Dist,1,'ascend');
% minDist_index{h,1}=Index1(1:floor(0.35*z));
%minDist_index{h,1}=min(find(Dist==min(Dist)));
%%  
minDist_index{h,1}=label{h,1}(minDist_index{h,1},1);%%找到最近点在原始数据集中的标号
%% 循环的作用是将每簇标号依次放到同一个数组中
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
%labels = []；
%X = class_data(:,1:d-1);
reblock1=sortrows(reblock1,1);%%按照升序排列标号（即按照原始数据集的标号进行排列如1，2，3，，，）
AB=minDist_index1;

lambda = 1e-4;
%% 每一簇label后加入同等大小且全为0的一列
A=zeros(n,1);
reblock1=[reblock1,A];%%在reblock1后面加入1列，用于存放softmax预测的相应位置的标签
% Weight decay parameter
%% %%%%%%%%%%%%%%%%%%%%初始化%%%%%%%%%%%%%%
isLabeled = zeros(1,n);
Labels = zeros(1,n);
%% %%%%%%%%%%%%%%%%%%%%计算代表性%%%%%%%%%%%%%%%%%

%% %%%%%%%%5%%%每个样本增加一个x0=1%%%%
oneArray = ones(1,n);
X = [oneArray; X']; %每个样本都要增加一个x0=1;
%X = mapminmax(X);
%%   %%%%%%%%% 构建初始训练集 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%通过代表性选择K个代表点%%%%% 
%     case 1
%  ratio = K;
%  trainingSet = desInd(1:1:ratio);
 %% 选取训练集

 %% 如果没有相同的，就以距离中心点最近的点作为训练集；有相同的就以之前购买的作为训练集

      p=AB;
     trainingSet=p;%%训练集
     q=size(p,1);
%% 循环作用是将训练集依次放入buyquery中
     for i=1:q
         trainingnum=trainingSet(i,1);
      buyquery(g,:)=data(trainingnum,:);
%       train_data(g,:)=X(:,trainingSet(i,1));
     g=g+1;
     end
    reblock1(trainingSet,end)=Y(trainingSet);%%将训练集的标签放到reblock1对应标号的最后一列
%  end
% buy_data_num=g-1;
%% 训练集
% train_data=data(trainingSet,:);

%% %%%训练集数
trainingInput = [X(:,trainingSet) train_data];     %%%%%%%训练集（将前一次去掉‘纯’的簇中的训练集去掉，并把剩余的加到本次的训练集中）
labels =[Y(trainingSet);Y_train_data];              %%%%%%%训练集标签（同训练集一样，加上上一次训练集标签）
Labels(trainingSet) = Y(trainingSet);
Labels = Labels';
isLabeled(trainingSet) = 1;
%%  %%%%%%%%% 训练theta %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = 0.0005 * randn(d, k);     % 随机产生初始theta
[cost, grad] = softmax_regression_vec(theta,trainingInput,labels,lambda);
options.maxIter = 100;
softmaxModel = softmaxTrain(d, k, lambda, trainingInput,labels, options);
theta = softmaxModel.optTheta;  %重新获得theta矩阵

%% STEP : Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pred = zeros(1, size(data, 2));
%[~,pred]= max(theta'*TestInput);%theta'*data这个矩阵如图5，某一个样本softmax最大值与这个矩阵某一列最大
%值是等价的，因为每一列除以同一个分母和不除是一样的，并且exp（.）是增函数，所以只求里面的最大值即可。
% a=1;

cost=0;
while sum(isLabeled)~=n
    for i = 1:n
        if (isLabeled(i) == 1)            %%实例如果已处理，则跳过该实例
            continue;
        end
         tempIncomData = X(:,i);        % 将不完备数据集的单个实例赋值给tempIncomData，从而不改变原不完备数据集
         
%          switch initial_num
%              case 1
        predProbability(:,i) = theta'*tempIncomData;                           %%计算z1，z2，z3
        M=max( predProbability(:,i));
        predProbability2(:,i) = exp(predProbability(:,i)-M);                      %%计算e^z1，e^z2，e^z3
        sumProb(:,i) = sum(predProbability2(:,i)); 
         completeP(:,i) = (predProbability2(:,i))./sumProb(i);
         cost=cost+2*(1-max(completeP(:,i)));
%          if (((1-completeP(i))*2)<1)%误差代价小于购买代价则直接预测
%              [~,pred]= max(theta'*tempIncomData);

             [~,pred]= max(completeP(:,i)); 
             Labels(i) = pred;
             reblock1(i,end)=Labels(i);%将预测的标签放入label最后一列
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
train_data=[data(trainingSet,:);train_before];%%把前一次的训练集加上
% train_before=data(trainingSet,:);
%% KNN1(最近邻)算法
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
   max_class_num=max(histc((A),b));%%计算预测簇类预测类别最多的一类的数量
   purity(h,1)=max_class_num./z;%%计算每一簇的预测纯度
   %% 计算KNN1（最近邻）每一簇的‘纯度’
   class_pred2{h,1}=KNN_predict_label(label{h,1},:);
   B=class_pred2{h,1}(:,end);
   c=1:max(B);
   KNN_max_class_num=max(histc((B),c));%%计算预测簇类预测类别最多的一类的数量
   purity2(h,1)=KNN_max_class_num./z;%%计算每一簇的预测纯度
   if (purity(h,1)==1)&&(purity2(h,1)==1)

      for i=1:z
           if A(i,:)==B(i,:)
         
           data_pred(tp,:)=[class{h,1}(i,:) A(i,:)];%%将满足纯度条件的数据放进一个数组中，不再调用
           tp=tp+1;
           else
               data(tb,:)=class{h,1}(i,:);%%进行迭代的数据（即不纯的数据）
               data_new_pred{tc,1}(tb,:)=[class{h,1}(i,:) A(i,:)];%%保存每一次迭代前的输入数据（已经经过softmax预测，并带有预测标签）
               tb=tb+1;
          end
       end
       train_out(qq,1)=h;
       qq=qq+1;
       
   else
       for i=1:z
         data(tb,:)=class{h,1}(i,:);%%进行迭代的数据（即不纯的数据）
%          data_new(tb,:)=[class{h,1}(i,:) A(i,:)];%%将要进行的迭代的数据后面加一列（对应的softmax预测的标签），方便在比较代价的时候找到代价最低时取出
         data_new_pred{tc,1}(tb,:)=[class{h,1}(i,:) A(i,:)];%%保存每一次迭代前的输入数据（已经经过softmax预测，并带有预测标签）
         tb=tb+1;
         
       end
       
   end
end
%% 将满足要求的簇中选的训练集去掉
if isempty(train_out)
else
train_data(train_out,:)=[];
end
tc=tc+1;
train_before=train_data;
Y_train_data=train_data(:,end);
pp=size(train_data,1);
train_data=[ones(1,pp);(train_data(:,1:end-1))'];%%下一轮softmax将要加入的训练集

 