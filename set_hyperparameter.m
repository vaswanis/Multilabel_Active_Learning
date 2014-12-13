function [] = set_hyperparameter(percent_compression, X,y,opts)

	%number of true labels
	L = size(y,2);

	%number of compressed labels
	K = floor((1 - percent_compression) * L);

	%number of examples
	N = size(X,1);
	d = size(X,2);

	%X = X - repmat(mean(X,1),[N 1]);


	N_train = floor(0.2 * N)

	X_train = X(1:N_train,:);
	y_train = y(1:N_train,:);
	N_test = size(X,1) - N_train;
	X_test = X(N_train + 1:end,:);
	y_test = y(N_train+1:end,:);

	temp = y_train';
	Y = temp(:);
		
	phi = rand(K,L);
	phi_tilde = kron(eye(N_train),phi);


	small_sigma_vals = logspace(-5,0,6)
	chi_vals =  logspace(-5,0,6)
	kernel_length_scale_vals = [1,10,100,1000];
	kernel_sigma_vals = [0.01,0.1,1,10,100,1000];
	
	best_f = 0;

	opts.kernelize = 1;

	for small_sigma = small_sigma_vals

		for chi = chi_vals

			for kernel_length_scale = kernel_length_scale_vals
				
				for kernel_sigma = kernel_sigma_vals

				if opts.kernelize == 1
					kernel = zeros(N_train,N_train);
					for i = 1:N_train
						for j = 1:N_train
							%norm(X_train(i,:) - X_train(j,:)) 
							temp_kernel = kernel_sigma^2 * exp(- norm(X_train(i,:) - X_train(j,:)) ^ 2 / (2 * kernel_length_scale^2));
							kernel(i,j) = temp_kernel;
						end
					end
					
					
					G = kernel;
				else
	
					G = X_train * X_train'; 
		
				end

				
				K_const = G + (small_sigma)^2 * eye(N_train);
				sigma_Z_const = zeros(N_train*K, N_train*K);

				for i = 1:N_train

				    for j = 1:N_train

        				Kij = eye(K,K) * K_const(i,j);
			        	sigma_Z_const((i-1)*K + 1:i * K, (j-1)*K + 1:j*K) = Kij;

				    end

				end

				f =  Y' * phi_tilde' * (pinv( (chi)^2 * eye(N_train*K,N_train*K) + sigma_Z_const )) * phi_tilde * Y ;
	
				fprintf('Chi: %f, Sigma: %f, Kernel length scale: %f, Kernel Sigma: %f, likelihood: %f\n',chi,small_sigma,kernel_length_scale,kernel_sigma,f)

				if f > best_f

					best_small_sigma = small_sigma;
					best_chi = chi;
					best_kernel_sigma = kernel_sigma;
					best_kernel_length_scale = kernel_length_scale;
					best_f = f;

				end

				end
			end

		end
	end
	
	fprintf('Optimal parameters:Chi: %f, Sigma: %f, Kernel length scale: %f, Kernel Sigma: %f, likelihood: %f\n',best_chi,best_small_sigma,best_kernel_length_scale,best_kernel_sigma,best_f)

end

