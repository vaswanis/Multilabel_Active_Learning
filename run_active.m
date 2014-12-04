clear;
close all;
% clc;

addpath('arff');
%number of true labels
L = 19;
%number of compressed labels
K = 19;

%noise parameters
opts.chi = 1e-2;
opts.small_sigma = 1e-1;
%number of update iterations
opts.maxiter = 500;

phi = rand(K,L);

%extract train data
if ispc
    data_fname = '.\datasets\birds\birds-train.arff';
else
    data_fname = './datasets/birds/birds-train.arff';
end
[X_train,y_train]  = parse_data(data_fname ,L);
d = size(X_train,2);

%extract test data
if ispc
    data_fname = '.\datasets\birds\birds-test.arff';
else
    data_fname = './datasets/birds/birds-test.arff';
end
[X_test,y_test]  = parse_data(data_fname ,L);


%concatenate train, test to derive posterior on y
N = 100;
X = [X_train ; X_test];
y = [y_train; y_test];
% N = size(X,1);
X = X';
X = X(:,1:N);
y = y(1:N,:);

Y(1:N) = struct('mu',zeros(L,1),'sigma',eye(L));

G = X' * X;
K_const = G + (opts.small_sigma)^2 * eye(N,N);

sigma_Z_const = zeros(N*K, N*K);
for i = 1:N
    for j = 1:N
        Kij = eye(K,K) * K_const(i,j);
        sigma_Z_const((i-1)*K + 1:i * K, (j-1)*K + 1:j*K) = Kij;
    end
end
phi_tilde = kron(eye(N),phi);
sigma_Y_const_inverse = phi_tilde' * pinv( (opts.chi)^2 * eye(N*K,N*K) + sigma_Z_const ) * phi_tilde;

Y = struct('mu',zeros(N*L,1),'sigma',zeros(N*L, N*L));

in_a = ones(N*L,1) * 1e-6;
in_b = ones(N*L,1) * 1e-6;

a = in_a + 0.5;
b = in_b;

for t = 1:opts.maxiter
    
if mod(t,10) == 0
    t
end
E_alpha = a ./ b;    
Y.sigma = pinv( diag(E_alpha) + sigma_Y_const_inverse ) ;

b = in_b + 0.5 * (diag(Y.sigma));
    
end

q = 0.2 * N * L; %number of labels to be predicted
y = 2 * y - 1;
sigma_12 = Y.sigma(1:q,q+1:end);
sigma_22 = Y.sigma(q+1:end,q+1:end);
temp = y';
temp = temp(:);
a = temp(1:N*L - q);
yhat = sigma_12 * pinv(sigma_22) * a;










