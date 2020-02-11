function [confus,precision,recall,F,F1] = compute_accuracy_F (actual_label,pred_label,classes)
% GETCM : gets confusion matrices, precision, recall, and F scores
% [confus,numcorrect,precision,recall,F] = getcm (actual,pred,[classes])
%
% actual is a N-element vector representing the actual classes
% pred is a N-element vector representing the predicted classes
% classes is a vector with the numbers of the classes (by default, it is 1:k, where k is the
%    largest integer to appear in actual or pred.
%

% 
% if size(actual_label,1) ~= size(pred_label,1)
%     pred_label=pred_label';
% end
% if nargin < 3
%     classes = [1:max(max(actual_label),max(pred_label))];
% end

%%
%%

%  numcorrect = sum(actual_label==pred_label);
%  accuracy = numcorrect/length(actual_label);
for i=1:length(classes)
    % confus(i,:) = hist(pred,classes);
    a = classes(i);
    d = find(actual_label==a);     % d has indices of points with class a
    for j=1:length(classes)
        confus(i,j) = length(find(pred_label(d)==classes(j)));
    end
end

precision=[];
recall=[];
F=[];


for i=1:length(classes)
    S1= sum(confus(i,:));
    if nargout>=4
        if S1
            recall(i) = confus(i,i) / sum(confus(i,:));
            recall(i) = confus(i,i) /S1;
        else
            recall(i) = 0;
        end
    end
    S2=  sum(confus(:,i));
    if nargout>=3
        if S2
            precision(i) = confus(i,i) / S2;
        else
            precision(i) = 0;
        end
    end
    if nargout>=5
        if (precision(i)+recall(i))
            F(i) = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i));
        else
            F(i) = 0;
        end
    end
end
mean_precision=mean(precision);
mean_recall=mean(recall);
F1=2*mean_precision*mean_recall/(mean_precision+mean_recall);
end