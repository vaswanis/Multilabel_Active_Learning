function [precision,train_time, test_time] = run(percent_compression, X,y,phi,opts)

	%number of true labels
	L = size(y,2);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	%number of examples
	N = size(X,1);
	d = size(X,2);

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
	
	%random binary projection  matrix
	%phi = round(rand(K,L));

	%all zeros projection matrix
	%phi = zeros(K,L);

	%dropping out labels 
	%phi = rand(K,L);
	%for i = 1:(L-K)
	%	col = ceil(rand() * L);  
	%	phi(:,col) = zeros(K,1);
	%end

	%squared exponential kernel parameters

	kernel_length_scale = 1;

	if opts.kernelize == 1
		kernel = zeros(N_train,N_train);
		for i = 1:N_train
			for j = 1:N_train
				temp_kernel = exp(- norm(X_train(i,:) - X_train(j,:)) ^ 2 / (2 * kernel_length_scale^2));
				kernel(i,j) = temp_kernel;
			end
		end

		kernel_sigma = sqrt( norm(X_train' * X_train) / norm(kernel) );
		kernel = kernel * kernel_sigma ^ 2;

		G = X_train' * pinv( kernel +  (opts.small_sigma)^(2) * eye(N_train) ) ;
	else
		G = pinv(X_train' * X_train + (opts.small_sigma)^2 * eye(d)) * X_train';
		
	end

	
        [W,phi,opts] = train_mod(X_train,y_train,K,opts,phi,1,[], G);
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

