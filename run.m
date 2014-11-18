addpath('arff');

%number of true labels
L = 1;

%number of compressed labels
K = 1;

%noise parameters
opts.chi = 2;
opts.small_sigma = 2;
%number of update iterations
opts.maxiter = 100;

%extract data
[X,y]  = parse_data('.\datasets\birds\birds-train.arff',L);

%train
W = train(X,y,K,opts);

%extract data
[X_test,y_test]  = parse_data('.\datasets\birds\birds-test.arff',L);

%test
Y = test(X_test,W,L,opts);
yhat = concat_struct_attr(Y,'mu');

% learn the noise parameters - Gaussian process regression
% think !
% debug - make the number of labels = 1
% no compression => K = L
% just one data point and test = train

