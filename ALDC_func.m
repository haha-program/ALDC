%% %%%测试函数%%%%%%%
%% 主要集成调用函数为KK1,KK2，softmax_cluster_K,和badblock函数
function [ave_F_measure,ave_cost]=ALDC_func(data,cluster_ratio)
dbstop if error
%data= 'C:\Users\浅陋\Desktop\所用的38个数据集\fourtyDecision.xlsx';
%[data,~] = xlsread(data);
 [n,c]=size(data); %求输入数据点的个数m
 %% 特征归一化%%%%%%
%[X_norm, mu, sigma] = featureNormalize(data(:,1:c-1));
%data=[X_norm data(:,c)];

num=n;
% RandIndex = randperm(n);
% data = data(RandIndex,:);
KNN_test=data;
buylabel=cell(10,1);
X=data(:,1:c-1);
Y=data(:,c);
Dists = pdist(X,'euclidean');
Max1=max(max(Dists));
k = numel(unique(Y)); 
buyquery1=cell(10,1);
buyquery_index=cell(10,1);
for j=1:10
g=1;
buyquery=[];
data_pred=[];
tp=1;
data_new_pred=cell(num,1);
tc=1;
a=1;
train_data=[];
Y_train_data=[];
train_before=[];
data=[X Y];
% Dists = pdist(X,'euclidean');
% Max=max(max(Dists));
 Max=Max1;
[class,label,K,Center]=KK1(data,Max,cluster_ratio);
[g,buyquery,data,data_pred,tp,data_new_pred,tc,cost,train_data,Y_train_data,train_before]=softmax_cluster_K(class,label,k,K,g,buyquery,Center,data,data_pred,tp,data_new_pred,tc,train_data,Y_train_data,train_before);
% num_query=numel(unique(Y(buyquery,:)));%购买的标签类别数量
cost_all(a,1)=cost;
a=a+1;
 if isempty(data)
     end_num=1;
 else
     end_num=0;
 end

 while (end_num==0)
%      initial_num=2;
Dists = pdist(data(:,1:end-1),'euclidean');
Max=max(max(Dists));
AA=data_new_pred{tc-1,1};
     [label,class,K,Center]=KK2(data,AA,Max);
     [g,buyquery,data,data_pred,tp,data_new_pred,tc,cost,train_data,Y_train_data,train_before]=softmax_cluster_K(class,label,k,K,g,buyquery,Center,data,data_pred,tp,data_new_pred,tc,train_data,Y_train_data,train_before);
%      num_query=numel(unique(Y(buyquery,:)));
     cost_all(a,1)=cost;
     [p,~]=size(cost_all);
     if (cost_all(p,1)>cost_all(p-1,1))||(isempty(data))
         end_num=1;
     else
       a=a+1;
       end_num=0;
     end
 end
 %% 找出最终预测标签%%%%%%
 if isempty(data)   
 else
 cost_p=size(data_new_pred{a-1,1},1)-size(data_new_pred{a,1},1);
 num_p=size(data_pred,1);
 data_pred(num_p-cost_p+1:num_p,:)=[];
 data_pred=[data_pred;data_new_pred{a-1}];
 end

% buylabel{j,1}=unique(buyquery);%%找到buyquery中全不相同的标号
%% 去掉训练集中相同的训练数据
buyquery1{j,1}=unique(buyquery,'rows');
buynum(j,1)=size(buyquery1{j,1},1);%%找到所有购买的数量
ratio(j,1)=buynum(j,1)./n;%%购买比率计算
%%  找到训练集在原始数据中的标号%%%%%%
R=KNN_test;
Q=buyquery1{j,1};
px=size(Q,1);
% R=[1 2 3;4 5 6;7 8 9];
% Q=[1 2 3];
source=[];
mnum=1;
for xx=1:px
Pz=ismember(R,Q(xx,:),'legacy');
find(all(Pz)==1);
PL=sum(Pz,2);
PPP=find(PL==c);
source(mnum,1)=min(PPP);
mnum=mnum+1;
end
buyquery_index{j,1}=source;%%将每一次的训练集标号依次存进cell中
NN=n-buynum(j,1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  SVM %%%%%%%%%%%%%%%%%%%%%%%%%
% SVM_data=KNN_test;
% SVM_traindata=buyquery1{j,1}(:,1:end);
% [X_norm, mu, sigma] = featureNormalize(SVM_traindata);%%归一化
% SVM_traindata=X_norm;
% SVM_trainlabel=buyquery1{j,1}(:,end);
% SVM_data(source,:)=[];
% SVM_testdata=SVM_data(:,1:end-1);
% % SVM_testdata=SVM_data(:,1:end-1);
% [X_norm, mu, sigma] = featureNormalize(SVM_testdata);
% SVM_testdata=X_norm;
% SVM_testlabel=SVM_data(:,end);
% model = svmtrain(SVM_trainlabel, SVM_traindata, '-c 1 -g 0.07');
% [predict_label,~, ~] = svmpredict(SVM_testlabel, SVM_testdata, model);
% % predict_label=predict_SVM;
% actual_label=SVM_testlabel;
% classes = [1:max(max(actual_label),max(predict_label))];
% [confus,precision,recall,F,F1]=compute_accuracy_F(actual_label,predict_label,classes);
% SVM_error=0;
% for i=1:NN
%     if(SVM_data(i,end))~=predict_label(i)
%        SVM_error=SVM_error+1;
%     end
% end
% SVM_errors(j)=SVM_error;
% accuary_SVM(j,1)=(NN-SVM_errors(j))./NN;%%精度测试
% cost_SVM(j,1)=(buynum(j,1)+2*SVM_errors(j))./n;%%代价测试
% F_measure_SVM(j,1)=mean(F);
% acc_SVM = mean(predict_SVM==y);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KNN1 算法%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KNN_error=0;
% KNN_train_data=buyquery1{j,1}(:,1:end-1);
% KNN_train_label=buyquery1{j,1}(:,end);
% num_neighbors=1;
% KNN_test_data=KNN_test(:,1:end-1);
% Factor = ClassificationKNN.fit(KNN_train_data, KNN_train_label, 'NumNeighbors', num_neighbors);
% KNN_predict_label = predict(Factor, KNN_test_data);
% actual_label=KNN_test(:,end);
% predict_label=KNN_predict_label;
% classes = [1:max(max(actual_label),max(predict_label))];
% [confus,precision,recall,F,F1]=compute_accuracy_F(actual_label,predict_label,classes);
% for i=1:n
%     if(KNN_test(i,end))~=KNN_predict_label(i)
%        KNN_error=KNN_error+1;
%     end
% end
% KNN_errors(j)=KNN_error;
% accuary_KNN(j,1)=(n-buynum(j,1)-KNN_errors(j))./(n-buynum(j,1));%%精度测试
% cost_KNN(j,1)=(buynum(j,1)+2*KNN_errors(j))./n;%%代价测试
% F_measure_KNN(j,1)=mean(F);
%%  %%%%%%%%%%%%%%%%%%%%%%%% KNN3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KNN3_test=KNN_test;
% KNN3_error=0;
% KNN3_train_data=buyquery1{j,1}(:,1:end-1);
% KNN3_train_label=buyquery1{j,1}(:,end);
% num_neighbors=3;
% KNN3_test(source,:)=[];
% KNN3_test_data=KNN3_test(:,1:end-1);
% % KNN3_test_data=KNN3_test(:,1:end-1);
% Factor = ClassificationKNN.fit(KNN3_train_data, KNN3_train_label, 'NumNeighbors', num_neighbors);
% KNN3_predict_label = predict(Factor, KNN3_test_data);
% actual_label=KNN3_test(:,end);
% predict_label=KNN3_predict_label;
% classes = [1:max(max(actual_label),max(predict_label))];
% [confus,precision,recall,F,F1]=compute_accuracy_F(actual_label,predict_label,classes);
% for i=1:NN
%     if(KNN3_test(i,end))~=KNN3_predict_label(i)
%        KNN3_error=KNN3_error+1;
%     end
% end
% KNN3_errors(j)=KNN3_error;
% accuary_KNN3(j,1)=(NN-KNN3_errors(j))./NN;%%精度测试
% cost_KNN3(j,1)=(buynum(j,1)+2*KNN3_errors(j))./n;%%代价测试
% F_measure_KNN3(j,1)=mean(F);
% %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%  C45   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % C_45_train=buyquery1{j,1};
% C_45_data=KNN_test;
% C_45_train=C_45_data(source,:);
% buynum_c45=buynum(j,1);
% C_45_data(source,:)=[];
% [test_targets,accuary_C45,cost_C45]=C4_5(C_45_train,C_45_data,buynum_c45,NN,n);
% actual_label=C_45_data(:,end);
% predict_label=test_targets;
% classes = [1:max(max(actual_label),max(predict_label))];
% [confus,precision,recall,F,F1]=compute_accuracy_F(actual_label,predict_label,classes);
% accuary_J48(j,1)=accuary_C45;%%精度测试
% cost_J48(j,1)=cost_C45;%%代价测试
% F_measure_J48(j,1)=mean(F);

%% 将对应的预测标签与真实标签进行比较，找到总的预测正确数
data_pred(source,:)=[];
predict_error=0;%a将预测错误数初值设为0
for i=1:NN
    if (data_pred(i,end-1)~=data_pred(i,end))
        predict_error=predict_error+1;
    end
end
actual_label=data_pred(:,end-1);
predict_label=data_pred(:,end);
classes = [1:max(max(actual_label),max(predict_label))];
[confus,precision,recall,F,F1]=compute_accuracy_F(actual_label,predict_label,classes);
predict_errors(j)=predict_error;
accuary(j,1)=(NN-predict_errors(j))./NN;%%精度测试
cost2(j,1)=(buynum(j,1)+2*predict_errors(j))./n;%%代价测试
F_measure(j,1)=mean(F);
end
%% 求购买个数和比率的均值及方差
ave_buynum=(sum(buynum))./10;%%求购买个数均值
var_buynum=var(buynum,1);%%求购买个数方差
ave_ratio=(sum(ratio))./10;%%求购买比率均值
var_ratio=var(ratio,1);%%求购买比率方差
all_buynum=[buynum;ave_buynum;var_buynum];
all_ratio=[ratio;ave_ratio;var_ratio];
%%  求算法精度、代价、F_measure的均值及方差
ave_accuary=(sum(accuary))./10;%%求精度均值
var_accuary=var(accuary,1);%%求精度方差
ave_cost=(sum(cost2))./10;%%求代价均值
var_cost=var(cost2,1);%%求代价方差
ave_F_measure=(sum(F_measure))./10;
var_F_measure=var(F_measure,1);
all_accuary=[accuary;ave_accuary;var_accuary];
all_cost=[cost2;ave_cost;var_cost];
all_F_measure=[F_measure;ave_F_measure;var_F_measure];
%% 求KNN1算法精度、代价、F_measure的均值及方差
% ave_accuary_KNN=(sum(accuary_KNN))./10;%%求精度均值
% var_accuary_KNN=var(accuary_KNN,1);%%求精度方差
% ave_cost_KNN=(sum(cost_KNN))./10;%%求代价均值
% var_cost_KNN=var(cost_KNN,1);%%求代价方差
% ave_F_measure_KNN=(sum(F_measure_KNN))./10;
% var_F_measure_KNN=var(F_measure_KNN,1);
%% 求KNN3算法精度、代价、F_measure的均值及方差
% ave_accuary_KNN3=(sum(accuary_KNN3))./10;%%求精度均值
% var_accuary_KNN3=var(accuary_KNN3,1);%%求精度方差
% ave_cost_KNN3=(sum(cost_KNN3))./10;%%求代价均值
% var_cost_KNN3=var(cost_KNN3,1);%%求代价方差
% ave_F_measure_KNN3=(sum(F_measure_KNN3))./10;
% var_F_measure_KNN3=var(F_measure_KNN3,1);
% all_accuary_KNN3=[accuary_KNN3;ave_accuary_KNN3;var_accuary_KNN3];
% all_cost_KNN3=[cost_KNN3;ave_cost_KNN3;var_cost_KNN3];
% all_F_measure_KNN3=[F_measure_KNN3;ave_F_measure_KNN3;var_F_measure_KNN3];
% %%  求J48算法精度、代价、F_measure的均值及方差
% ave_accuary_J48=(sum(accuary_J48))./10;%%求精度均值
% var_accuary_J48=var(accuary_J48,1);%%求精度方差
% ave_cost_J48=(sum(cost_J48))./10;%%求代价均值
% var_cost_J48=var(cost_J48,1);%%求代价方差
% ave_F_measure_J48=(sum(F_measure_J48))./10;
% var_F_measure_J48=var(F_measure_J48,1);
% all_accuary_J48=[accuary_J48;ave_accuary_J48;var_accuary_J48];
% all_cost_J48=[cost_J48;ave_cost_J48;var_cost_J48];
% all_F_measure_J48=[F_measure_J48;ave_F_measure_J48;var_F_measure_J48];
% %%  求SVM算法精度、代价、F_measure的均值及方差
% ave_accuary_SVM=(sum(accuary_SVM))./10;%%求精度均值
% var_accuary_SVM=var(accuary_SVM,1);%%求精度方差
% ave_cost_SVM=(sum(cost_SVM))./10;%%求代价均值
% var_cost_SVM=var(cost_SVM,1);%%求代价方差
% ave_F_measure_SVM=(sum(F_measure_SVM))./10;
% var_F_measure_SVM=var(F_measure_SVM,1);
% all_accuary_SVM=[accuary_SVM;ave_accuary_SVM;var_accuary_SVM];
% all_cost_SVM=[cost_SVM;ave_cost_SVM;var_cost_SVM];
% all_F_measure_SVM=[F_measure_SVM;ave_F_measure_SVM;var_F_measure_SVM];
% % % X = mapminmax(X);
% % XX=[X Y];
% 


 