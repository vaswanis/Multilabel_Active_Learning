clear;
close all;
% clc;

addpath('arff');
%number of true labels
L = 174;
%number of compressed labels
K = 174;

%number of update iterations
opts.maxiter = 100;
CV = 0;

%extract data
if ispc
    data_fname = '.\datasets\CAL500\CAL500.arff';
else
    data_fname = './datasets/CAL500/CAL500.arff';
end

[X,y]  = parse_data(data_fname ,L);
y = 2 * y - 1;
N = 300;
X = X(1:N,:);
y = y(1:N,:);
% N = size(X,1);


%n-fold cross-validation
if CV
    n = 5;
    chunk_size = floor(N/n);
    min_param_err = 100;
    for chi=[1e-4, 1e-3, 1e-2]
        for small_sigma = [1e-2, 1e-3, 1e-4]
            err = zeros(n,1);
            for test_num = 1:n
                Xtest = X((test_num-1)*chunk_size+1, :);
                Xtrain = X(setdiff(1:N,(test_num-1)*chunk_size+1),:);
                Ytest = y((test_num-1)*chunk_size+1,:);
                Ytrain = y(setdiff(1:N,(test_num-1)*chunk_size+1),:);
                
                opt.chi = chi;
                ops.small_sigma = small_sigma;
                
                [W,phi,opts] = train_mod(Xtrain,Ytrain,K,opts);
                
                Ytemp = test(Xtest,W,L,phi,opts);
                Yhat = concat_struct_attr(Ytemp,'mu');
                
                err(test_num) = compute_hamming_distance(Yhat, Ytest);
            end
            param_err = mean(err);
            if param_err < min_param_err
                best_small_sigma = small_sigma;
                best_chi = chi;
                min_param_err = param_err;
            end
            
        end
    end
else
    best_small_sigma = 1e-3;
    best_chi = 1e-2;
end

opts.chi = best_chi;
opts.small_sigma = best_small_sigma;

%train
t = clock;
% ------------------------------------------------- %
[W,phi,opts] = train_mod(X,y,K,opts);
% ------------------------------------------------- %
fprintf('Train time = %f\n', etime(clock,t));

%extract data
if ispc
    data_fname = '.\datasets\CAL500\CAL500.arff';
else
    data_fname = './datasets/CAL500/CAL500.arff';
end

[X_test,y_test]  = parse_data(data_fname ,L);
y_test = 2* y_test - 1;
N = 100;
X_test = X_test(end-N+1:end,:);
y_test = y_test(end-N+1:end,:);

%test
t = clock;
% ------------------------------------------------- %
Y = test(X_test,W,L,phi,opts);
yhat = concat_struct_attr(Y,'mu');
% ------------------------------------------------- %
fprintf('Test time = %f\n', etime(clock,t));

% calculating precision@k
k = 1;
precision = compute_precision(yhat, y_test,k);

% calculating average relative hamming distance
HD = compute_hamming_distance(yhat, y_test);

%Display Results
fprintf('Results: Number of labels = %d, Amount of compression = %f\n', L, (1 - (K/L))* 100);
fprintf('Precision@%d = %f\n', k, precision);
fprintf('Average Relative Hamming Distance = %f\n',HD);

%TODO:
% learn the noise parameters - Gaussian process regression
% just one data point and test = train
