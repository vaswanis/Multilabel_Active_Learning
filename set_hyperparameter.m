function [] = set_hyperparameter(percent_compression, X,y,small_sigma,chi)

	%number of true labels
	L = size(y,2);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	%number of examples
	N = size(X,1);
	d = size(X,2);

	N_train = floor(0.4 * N)

	X_train = X(1:N_train,:);
	y_train = y(1:N_train,:);
	N_test = size(X,1) - N_train;
	X_test = X(N_train + 1:end,:);
	y_test = y(N_train+1:end,:);

	temp = y_train';
	Y = temp(:);
		
	G = X_train * X_train';
	phi = rand(K,L);
	phi_tilde = kron(eye(N_train),phi);


	small_sigma_vals = logspace(-5,1,7)
	chi_vals = logspace(-5,1,7)
		

	for small_sigma = small_sigma_vals
		for chi = chi_vals
	
			K_const = G + (small_sigma)^2 * eye(N_train,N_train);
			sigma_Z_const = zeros(N_train*K, N_train*K);
			for i = 1:N_train
			    for j = 1:N_train
        			Kij = eye(K,K) * K_const(i,j);
		        	sigma_Z_const((i-1)*K + 1:i * K, (j-1)*K + 1:j*K) = Kij;
			    end
			end
		
			f =  Y' * phi_tilde' * pinv( (chi)^2 * eye(N_train*K,N_train*K) + sigma_Z_const ) * phi_tilde * Y ;
			fprintf('Chi: %f, Sigma: %f, likelihood: %f\n',chi,small_sigma,f)

		end
	end
	


end

