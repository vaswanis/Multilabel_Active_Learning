function [W] = train_mod(X,y,K,opts)

L = size(y,2); %#(true labels)

%noise parameters
chi = opts.chi;
small_sigma = opts.small_sigma;
%number of update iterations
maxiter = opts.maxiter;

G = X * X';
d = size(X,1);%#(num features)
N = size(X,2); %#(num examples)

%random projection matrix
phi = rand(K,L);

%obtaining the compressed labels
z = (phi * y')';

%prior on W
sigma_p = eye(d);
W(1:K) = struct('mu',zeros(d,1),'sigma',sigma_p);

A = ((small_sigma)^-2) * G + pinv(sigma_p);
A_inverse = pinv(A);

for i = 1:K
    W(i).mu = ((small_sigma)^-2) * A_inverse * X * y(:,i);
    W(i).sigma = A_inverse;
end

%testing on the train set
zhat = zeros(size(z));
for i = 1:K
    zhat(:,i) = X' * W(i).mu ;
end


end