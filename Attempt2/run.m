clear;
close all;
clc;

addpath('arff');
%number of true labels
L = 19;
%number of compressed labels
K = 10;

%noise parameters
opts.chi = 1e-2;
opts.small_sigma = 1e-2;
%number of update iterations
opts.maxiter = 1000;

%extract data
[X,y]  = parse_data('.\datasets\birds\birds-train.arff',L);
y = 2 * y - 1;

%train
t = clock;
% ------------------------------------------------- %
[W,phi,opts] = train_mod2(X,y,K,opts);
% ------------------------------------------------- %
fprintf('Train time = %f\n', etime(clock,t));

%extract data
[X_test,y_test]  = parse_data('.\datasets\birds\birds-test.arff',L);

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