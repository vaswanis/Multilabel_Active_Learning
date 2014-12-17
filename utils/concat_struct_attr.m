function [A] = concat_struct_attr(W, attr_name)

K = size(W,2);
d = length(W(1).(attr_name));
A = zeros(K,d);
for k = 1:K
    A(k,:)  = W(k).(attr_name);
end

end