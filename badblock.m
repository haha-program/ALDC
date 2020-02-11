function [phi]=badblock(maxdiameter,Max,K)
%%% 函数作用为计算每一簇的误差值%%%%%%%%
%%% 即将子簇直径与父簇直径的比值带入误差拟合函数来计算%%%%%%
phi=cell(K,1);
normdis=cell(K,1);
for i=1:K
    normdis{i,1}=maxdiameter{i,1}./Max;
   phi{i,1}=(-0.01641)*normdis{i,1}.^3+(-0.1231)*normdis{i,1}.^2+0.3322*normdis{i,1}+0.009893;
     %phi{i,1}=(-0.06378)*normdis{i,1}.^3+(-0.02442)*normdis{i,1}.^2+0.2343*normdis{i,1}-0.01133;
end
end
