function [precision_uncertainty, precision_rand] = run_active(percent_compression, X, y, opts)

	%number of true labels
	L = size(y,2);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	N = size(X,1);
	N_train = floor(opts.train_fraction * N);
	X_train = X(1:N_train,:);
	y_train = y(1:N_train,:);

	N_test = size(X,1) - N_train;

	N_train_active = N_train - opts.N_train_initial;
	X_train_initial = X_train(1:opts.N_train_initial, :);
	y_train_initial = y_train(1:opts.N_train_initial, :);
	X_train_active = X_train(opts.N_train_initial+1: end, :);
	y_train_active = y_train(opts.N_train_initial+1: end, :);
	X_test = X(N_train + 1:end,:);
	y_test = y(N_train+1:end,:);

	%n-fold cross-validation
	if opts.CV
		%n-fold cross-validation
		cross_validation;
	else
		best_small_sigma = 1e-2;
		best_chi = 1e-4;
	end

	opts.chi = best_chi;
	opts.small_sigma = best_small_sigma;

	%initial train
	t = clock;
	phi = rand(K,L);
	[W,phi,opts] = train_mod(X_train_initial,y_train_initial,K,opts,phi,1,[]);
	fprintf('Train time = %f\n', etime(clock,t));

	%inital test
	Y = test_mod(X_test,W,L,phi,opts);
	yhat_initial = concat_struct_attr(Y,'mu');
	precision = compute_precision(yhat_initial, y_test,opts.k);
	fprintf('Initial Precision@%d = %f\n', opts.k, precision);

	
	%different matrices for random sampling and uncertainty sampling
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

		[X_train_initial_uncertainty, y_train_initial_uncertainty, X_train_active_uncertainty, y_train_active_uncertainty, W_uncertainty, phi, opts] = select_instance(X_train_initial_uncertainty, X_train_active_uncertainty, y_train_initial_uncertainty, y_train_active_uncertainty, W_uncertainty, L, opts, phi, K, 'uncertainty', opts.selection_batch_size);

		[X_train_initial_rand, y_train_initial_rand, X_train_active_rand, y_train_active_rand, W_rand, phi, opts] = select_instance(X_train_initial_rand, X_train_active_rand, y_train_initial_rand, y_train_active_rand, W_rand, L, opts, phi, K, 'random', opts.selection_batch_size);

		fprintf('End of active learning round %d\n', AL_round);

		%test at each round	
		Y = test_mod(X_test,W_uncertainty,L,phi,opts);
		yhat_uncertainty = concat_struct_attr(Y,'mu');

		Y = test_mod(X_test,W_rand,L,phi,opts);
		yhat_rand = concat_struct_attr(Y,'mu');

		% calculating precision@k
		precision_uncertainty(AL_round) = compute_precision(yhat_uncertainty, y_test,opts.k);
		precision_rand(AL_round) = compute_precision(yhat_rand, y_test,opts.k);

		fprintf('Uncertainty precision = %f, Random precision = %f\n',precision_uncertainty(AL_round), precision_rand(AL_round));


	end
end
