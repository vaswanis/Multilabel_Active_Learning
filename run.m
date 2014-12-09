function [precision] = run(L, percent_compression)

	addpath('arff');

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	%number of update iterations
	opts.maxiter = 100;
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


	N_train = floor(train_fraction * N);
	X_train = X(1:N_train,:);
	y_train = y(1:N_train,:);
	N_test = size(X,1) - N_train;
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
	phi = round(rand(K,L));
	%phi = rand(K,L);
	[W,phi,opts] = train_mod(X_train,y_train,K,opts,phi,1,[]);
	% ------------------------------------------------- %
	fprintf('Train time = %f\n', etime(clock,t));

	%test
	t = clock;
	% ------------------------------------------------- %
	Y = test_mod(X_test,W,L,phi,opts);
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

end

%TODO:
% learn the noise parameters - Gaussian process regression
