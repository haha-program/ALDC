function [class,label,K,Center]=KK1(data,Max,cluster_ratio)
%% �����������Ǹ��ݵ�һ���Ĵ������ҵ���Ѿ������
%% %%%%%%%%����%%%%%%%%%%%%%%%%%%%%%%%
%%data:ԭʼ���ݼ�����
%%Max:���ݼ���Զ������֮��ľ��룬�����ݼ���ֱ��
[n,c]=size(data); %���������ݵ�ĸ���m
%data1=data;
X=data(:,1:c-1);
maxK=floor(cluster_ratio*n);%ȷ�����������������Ϊ10%�����ܸ���(�������ݼ���С����ѡ�����������������Ϊ0.1*n,0.05*n,sqrt(n))
cost=zeros(maxK-1,1);%����
%% �����ݼ���ֱ��
%  Dists = pdist(X,'euclidean');
%  Max=max(max(Dists));
 Center1=cell(maxK-1,1);%���ĵ�
 t=1;
%% �ҵ�������������С��Kֵ
for K=2:floor(cluster_ratio*n)%%%��maxK����һ��,���Ǿ����������
    maxdiameter=cell(K,1);
    %Q=zeros(K,c-1);
    class_distence=cell(K,1);
    class=cell(K,1);
    %% �ȼ��ѡȡ��ʼ�������ĵ�
% for i=1:K
%     %C(i,:)=X(randi(m,1),:);     %�����ʼ����������%randi(n,1)����һ��1��n��α�������
%     %b=randi(n,1);
%    b=floor((n.*i)./(K+1));% ѡȡ��ʼ���ĵ�ı�ţ�����100��۳�2�࣬��ѡȡ��33��66��������Ϊ��ʼ���ĵ㣩
%    Q(i,:)=X(b,:);
% end
% matrix=Q;%��ʼ���ľ���
% [idx,C]= kmeans(X,K,'start',matrix);%�������Դ���Kmeans�㷨�����нϿ�һЩ
%Center{t,1}=C;
[idx,C]= kmeans(X,K);
Center1{t,1}=C;
cluster_cost=0;%ÿһ��Ĵ��۳�ʼֵ��Ϊ0
%% ����������ΪKʱ���ܴ���
for i=1:K
    class{i,1}=data(idx==i,:);%���ڵ�i�������
    [p,~]=size(class{i,1});
    %% �����i���ֱ�����������Ӧ�����
    class_distence{i,1}=X(idx==i,:);
     Dists = pdist(class_distence{i,1},'euclidean');
     maxdiameter{i,1}=max(max(Dists));
     [phi]=badblock(maxdiameter,Max,K);%%�����Ϻ���
%       normdis=maxdiameter/Max;
%      phi=(-0.01641)*normdis.^3+(-0.1231)*normdis.^2+0.3322*normdis+0.009893;
%%  �����ֵ�i��ֻ��һ�����ݵ��ʱ�򣬼����Ϊ0
       if isempty(phi{i,1})
           phi{i,1}=0;  
       end
     cluster_cost=cluster_cost+p.*phi{i,1}; %ÿһ����۷ֱ����
end 
cost(t,1)=2*cluster_cost+K;%����ΪKʱ���ܴ���
t=t+1;
end
%costK=cost;
idk=find(cost==min(cost))+1;%�ҵ���С���۵ľ������K
K=idk;
% Q=zeros(K,c-1);
class=cell(K,1);
%class_diatence=cell(K,1);
label=cell(K,1);
% Center=cell(K,1);
% t=1;
%% ������Ϊ��Сʱ�ľ������K��Ϊ���ݼ��ľ������
% for i=1:K
%    % C(i,:)=X(randi(m,1),:);     %�����ʼ����������%randi(n,1)����һ��1��n��α�������
%    %b=randi(n,1);
%    b=floor((n.*i)./(K+1));
%    Q(i,:)=X(b,:);
% %    buyquery(g,1)=b;
% %    g=g+1;
% end
matrix=Center1{K-1,1};
[idx,C]= kmeans(X,K,'start',matrix);
% [idx,C]= kmeans(X,K);
 Center=C;
% t=t+1;
%cluster_cost=0;
%label_center=zeros(K,1);
for i=1:K
    class{i,1}=data(idx==i,:);
    label{i,1}=find(idx==i);%%�ҵ���i�صı��
    
end 
end
