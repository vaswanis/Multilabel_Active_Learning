clear;
close all;
% clc;

addpath('arff');
%number of true labels
L = 174;
%number of compressed labels
percent_compression = 0.9;
K = floor((1 - percent_compression) * L);

%number of update iterations
opts.maxiter = 1000;
opts.max_rounds = 3;
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
N_train_initial = 100;
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
[W,phi,opts] = train_mod(X_train_initial,y_train_initial,K,opts,phi);
% ------------------------------------------------- %
fprintf('Train time = %f\n', etime(clock,t));

for round = 1:opts.max_rounds
%-------------------------------------------------- %
	Y = test(X_train_active,W,L,phi,opts);
	H = zeros(N_train_active, 1);
	for sample=1:N_train_active
		H(sample) = logdet(Y(sample).sigma);
	end

	[val, ind] = max(H); %get point with max uncertainty

	X_train_initial = [X_train_initial ; X_train_active(ind,:)];
	y_train_initial = [y_train_initial ; y_train_active(ind,:)];
	X_train_active(ind,:) = [];
	y_train_active(ind,:) = [];

	N_train_active = N_train_active - 1;
	N_train_initial = N_train_initial + 1;

		
	[W,phi,opts] = train_mod(X_train_initial,y_train_initial,K,opts,phi);

	%test at each round	
	Y = test(X_test,W,L,phi,opts);
	yhat = concat_struct_attr(Y,'mu');

	% calculating precision@k
	k = 5;
	precision = compute_precision(yhat, y_test,k)

% ------------------------------------------------- %
end
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
