function [new_X_train, new_y_train, new_X_train_active, new_y_train_active, new_W, new_phi, new_opts, new_kernel] = select_instance(X_train_initial, X_train_active, y_train_initial, y_train_active, W, L, opts, phi, K, criterion, batch_size, kernel)

    N_train_active = size(X_train_active, 1);
    d = size(X_train_active,2);

    if strcmp(criterion, 'uncertainty')

	  Y = test(X_train_active,W,L,phi,opts); 
	  H = zeros(N_train_active, 1);
    
          for i = 1:N_train_active
            %H(sample) = ((rcond(Y(sample).sigma)));
	    %H(i) = sum(abs(log(diag(Y(i).sigma))));
	    H(i) = sum(abs(diag(Y(i).sigma)));
          end
          [~, inds] = sort(H(:), 'descend');

    elseif strcmp(criterion, 'random')
        inds = randperm(N_train_active);
    else
        disp('invalid criterion');
    end
    
    selection_inds = inds(1:batch_size);

	N_train = size(X_train_initial,1);
    
	new_X_train = [X_train_initial ; X_train_active(selection_inds,:)];
	new_y_train = [y_train_initial ; y_train_active(selection_inds,:)];
	new_X_train_active = X_train_active;
	new_y_train_active = y_train_active;
	new_X_train_active(selection_inds,:) = [];
	new_y_train_active(selection_inds,:) = [];

	new_N_train = size(new_X_train,1);
    
    if opts.kernelize == 1
        
        new_kernel = zeros(new_N_train,new_N_train);
        new_kernel(1:N_train,1:N_train) = kernel;
        temp = zeros(new_N_train,batch_size);
        for i = 1:new_N_train
            for j = 1:batch_size
                temp_kernel = opts.kernel_sigma^2 * exp(- norm(new_X_train(i,:) - new_X_train(N_train + j,:)) ^ 2 / (2 * opts.kernel_length_scale^2));
                temp(i,j) = temp_kernel;
            end
        end
        new_kernel(:, N_train+1:end) = temp;
        temp2 = temp(1:N_train,:);
        new_kernel(N_train+1:end,1:N_train) = temp2';
        G = new_X_train' * pinv( new_kernel +  (opts.small_sigma)^(2) * eye(new_N_train) ) ;
        
    else
        
        new_kernel = [];
        G = pinv(new_X_train' * new_X_train + (opts.small_sigma)^2 * eye(d)) * new_X_train';
        
    end
	
 	[new_W,new_phi,new_opts] = train(new_X_train,new_y_train,K,opts,phi,1,W,G);

end

