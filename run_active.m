function [precision_uncertainty, precision_rand] = run_active(percent_compression, X, y, opts)

	%number of true labels
	L = size(y,2);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	N = size(X,1);
	d = size(X,2);
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

	N_train_initial = opts.N_train_initial;

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

	opts.kernel_length_scale = 1;
	opts.kernel_sigma = 1;
	if opts.kernelize == 1
		kernel = zeros(N_train_initial,N_train_initial);
		for i = 1:N_train_initial
			for j = 1:N_train_initial
				temp_kernel = exp(- norm(X_train_initial(i,:) - X_train_initial(j,:)) ^ 2 / (2 * opts.kernel_length_scale^2));
				kernel(i,j) = temp_kernel;
			end
		end

		opts.kernel_sigma = sqrt( norm(X_train_initial' * X_train_initial) / norm(kernel) );
		kernel = kernel * opts.kernel_sigma ^ 2;

		G = X_train_initial' * pinv( kernel +  (opts.small_sigma)^(2) * eye(N_train_initial) ) ;
	else
		G = pinv(X_train_initial' * X_train_initial + (opts.small_sigma)^2 * eye(d)) * X_train_initial';
		
	end


	[W,phi,opts] = train_mod(X_train_initial,y_train_initial,K,opts,phi,1,[], G) ;
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

	if opts.kernelize == 1
		kernel_uncertainty = kernel;
		kernel_rand = kernel;
	else
		kernel_uncertainty = [];
		kernel_rand = [];
	end

	precision_uncertainty = zeros(opts.max_rounds, 1);
	precision_rand = zeros(opts.max_rounds, 1);

	for AL_round = 1:opts.max_rounds

		[X_train_initial_uncertainty, y_train_initial_uncertainty, X_train_active_uncertainty, y_train_active_uncertainty, W_uncertainty, phi, opts, kernel_uncertainty] = select_instance(X_train_initial_uncertainty, X_train_active_uncertainty, y_train_initial_uncertainty, y_train_active_uncertainty, W_uncertainty, L, opts, phi, K, 'uncertainty', opts.selection_batch_size, kernel_uncertainty );

		[X_train_initial_rand, y_train_initial_rand, X_train_active_rand, y_train_active_rand, W_rand, phi, opts, kernel_rand] = select_instance(X_train_initial_rand, X_train_active_rand, y_train_initial_rand, y_train_active_rand, W_rand, L, opts, phi, K, 'random', opts.selection_batch_size, kernel_rand );

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
