function [av_precision] = compute_precision(yhat, y_test, k)

N = size(y_test,1);
total_pos_labels = 0;

for i = 1:N
    
    [val, index] = sortrows(yhat(i,:)',-1);
    no_pos_labels = sum(y_test(i,index(1:k)));
    total_pos_labels = total_pos_labels + no_pos_labels;      
    
end

    av_precision = (total_pos_labels) / (k*N) * 100;
    
end