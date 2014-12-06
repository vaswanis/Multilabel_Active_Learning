function [new_X_train, new_y_train, new_X_train_active, new_y_train_active, new_W, new_phi, new_opts] = select_instance(X_train_initial, X_train_active, y_train_initial, y_train_active, W, L, opts, phi, K, criterion, batch_size)

    N_train_active = size(X_train_active, 1);
    Y = test(X_train_active,W,L,phi,opts);
	H = zeros(N_train_active, 1);
	
    
    if strcmp(criterion, 'uncertainty')
          for sample=1:N_train_active
            H(sample) = abs(logdet(Y(sample).sigma));
          end
        [~, inds] = sort(H(:), 'descend');
    elseif strcmp(criterion, 'random')
        inds = randperm(N_train_active);
    else
        disp('unvalid criterion');
    end
    
    selection_inds = inds(1:batch_size);
    
	new_X_train = [X_train_initial ; X_train_active(selection_inds,:)];
	new_y_train = [y_train_initial ; y_train_active(selection_inds,:)];
	new_X_train_active = X_train_active;
	new_y_train_active = y_train_active;
	new_X_train_active(selection_inds,:) = [];
	new_y_train_active(selection_inds,:) = [];


	[new_W,new_phi,new_opts] = train_mod(new_X_train,new_y_train,K,opts,phi, 0, W);


end

