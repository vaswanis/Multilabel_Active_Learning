clear;
close all;
% clc;

addpath('arff');
%number of true labels
L = 174;
%number of compressed labels
percent_compression = 0.3;
K = floor((1 - percent_compression) * L);

%number of update iterations
opts.maxiter = 1000;
opts.max_rounds = 10;
train_fraction = 0.8;
CV = 0;

%extract data
if ispc
    data_fname = '.\datasets\CAL500\CAL500.arff';
else
    data_fname = './datasets/CAL500/CAL500.arff';
end
[X,y]  = parse_data(data_fname ,L);
disp('Read file');
N = size(X,1);
% y = 2 * y - 1;
N_train = floor(train_fraction * N);
X_train = X(1:N_train,:);
y_train = y(1:N_train,:);
N_test = size(X,1) - N_train;
N_train_initial = 50;
selection_batch_size = 2;

N_train_active = N_train - N_train_initial;
X_train_initial = X_train(1:N_train_initial, :);
y_train_initial = y_train(1:N_train_initial, :);
X_train_active = X_train(N_train_initial+1: end, :);
y_train_active = y_train(N_train_initial+1: end, :);
X_test = X(N_train + 1:end,:);
y_test = y(N_train+1:end,:);


%n-fold cross-validation
if CV
   cross_validation;
else
    best_small_sigma = 1e-2;
    best_chi = 1e-4;
end
opts.chi = best_chi;
opts.small_sigma = best_small_sigma;

%train
t = clock;
% ------------------------------------------------- %
%load 'phi_50'
phi = rand(K,L);
[W,phi,opts] = train_mod(X_train_initial,y_train_initial,K,opts,phi,1,[]);
% ------------------------------------------------- %
fprintf('Train time = %f\n', etime(clock,t));

    Y = test(X_test,W,L,phi,opts);
    yhat_initial = concat_struct_attr(Y,'mu');
    k = 5;
    precision = compute_precision(yhat_initial, y_test,k);
    fprintf('Initial Precision@%d = %f\n', k, precision);


    X_train_initial_uncertainty = X_train_initial;
    y_train_initial_uncertainty = y_train_initial;
    X_train_active_uncertainty = X_train_active;
    y_train_active_uncertainty = y_train_active;
    X_train_initial_rand = X_train_initial;
    y_train_initial_rand = y_train_initial;
    X_train_active_rand = X_train_active;
    y_train_active_rand = y_train_active;
    
    W_uncertainty = W;
    W_rand = W;
    
    precision_uncertainty = zeros(opts.max_rounds, 1);
    precision_rand = zeros(opts.max_rounds, 1);

for AL_round = 1:opts.max_rounds
%-------------------------------------------------- %  
    
    [X_train_initial_uncertainty, y_train_initial_uncertainty, X_train_active_uncertainty, y_train_active_uncertainty, W_uncertainty, phi, opts] = select_instance(X_train_initial_uncertainty, X_train_active_uncertainty, y_train_initial_uncertainty, y_train_active_uncertainty, W_uncertainty, L, opts, phi, K, 'uncertainty', selection_batch_size);
   
    
    [X_train_initial_rand, y_train_initial_rand, X_train_active_rand, y_train_active_rand, W_rand, phi, opts] = select_instance(X_train_initial_rand, X_train_active_rand, y_train_initial_rand, y_train_active_rand, W_rand, L, opts, phi, K, 'random', selection_batch_size);
	
	fprintf('End of active learning round %d\n', AL_round);

	%test at each round	
	Y = test(X_test,W_uncertainty,L,phi,opts);
	yhat_uncertainty = concat_struct_attr(Y,'mu');
    
	Y = test(X_test,W_rand,L,phi,opts);
	yhat_rand = concat_struct_attr(Y,'mu');
    

	% calculating precision@k
	k = 5;
	precision_uncertainty(AL_round) = compute_precision(yhat_uncertainty, y_test,k);
	precision_rand(AL_round) = compute_precision(yhat_rand, y_test,k);

	fprintf('Uncertainty precision = %f, Random precision = %f\n',precision_uncertainty(AL_round), precision_rand(AL_round));

% ------------------------------------------------- %
end
precision_uncertainty
precision_rand
%test
t = clock;
% ------------------------------------------------- %
Y = test(X_test,W,L,phi,opts);
yhat = concat_struct_attr(Y,'mu');
% ------------------------------------------------- %
fprintf('Test time = %f\n', etime(clock,t));

% calculating precision@k
k = 5;
precision = compute_precision(yhat, y_test,k);

% calculating average relative hamming distance
%HD = compute_hamming_distance(yhat, y_test);

%Display Results
fprintf('Results: Number of labels = %d, Amount of compression = %f\n', L, (1 - (K/L))* 100);
fprintf('Precision@%d = %f\n', k, precision);
%fprintf('Average Relative Hamming Distance = %f\n',HD);

%TODO:
% learn the noise parameters - Gaussian process regression
% just one data point and test = train
