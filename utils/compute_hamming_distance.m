function [av_HD] = compute_hamming_distance(yhat,y_test)

N = size(y_test,1);
L = size(y_test,2);
HD = 0;

for i = 1:N
    HD = HD + (sum(y_test(i,:) ~= ((yhat(i,:) > 0.5))));
end

av_HD = HD / (L * N) * 100;

end
