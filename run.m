function [precision,train_time, test_time] = run(percent_compression, X,y,opts)

	%number of true labels
	L = size(y,2);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	%number of examples
	N = size(X,1);

	N_train = floor(opts.train_fraction * N);

	X_train = X(1:N_train,:);
	y_train = y(1:N_train,:);
	N_test = size(X,1) - N_train;
	X_test = X(N_train + 1:end,:);
	y_test = y(N_train+1:end,:);

	if opts.CV
		%n-fold cross-validation
		cross_validation;
	else
		%default parameters	
		best_small_sigma = 1e-2;
		best_chi = 1e-4;
	end

	opts.chi = best_chi;
	opts.small_sigma = best_small_sigma;

	%train
	t = clock;
	%phi = rand(K,L);

	%phi = round(rand(K,L));

	%phi = zeros(K,L);

	phi = rand(K,L);
	for i = 1:(L-K)
		col = ceil(rand() * L);  
		phi(:,col) = zeros(K,1);
	end
	
        [W,phi,opts] = train_mod(X_train,y_train,K,opts,phi,1,[]);
	train_time = etime(clock,t); 
	fprintf('Train time = %f\n', train_time);

	%test
	t = clock;
	Y = test_mod(X_test,W,L,phi,opts);
	yhat = concat_struct_attr(Y,'mu');
	test_time = etime(clock,t);
	fprintf('Test time = %f\n', test_time);

	% calculating precision@k
	precision = compute_precision(yhat, y_test,opts.k);

	%Display Results
	fprintf('Results: Number of labels = %d, Amount of compression = %f\n', L, (1 - (K/L))* 100);
	fprintf('Precision@%d = %f\n', opts.k, precision);

end

