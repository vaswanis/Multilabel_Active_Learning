function [new_X_train, new_y_train, new_X_train_active, new_y_train_active, new_W, new_phi, new_opts] = select_instance(X_train_initial, X_train_active, y_train_initial, y_train_active, W, L, opts, phi, K, criterion)

    N_train_active = size(X_train_active, 1);
    Y = test(X_train_active,W,L,phi,opts);
	H = zeros(N_train_active, 1);
	
    
    
    if strcmp(criterion, 'uncertainty')
          for sample=1:N_train_active
            H(sample) = logdet(Y(sample).sigma);
         end
        [val, ind] = max(H); %get point with max uncertainty
    elseif strcmp(criterion, 'random')
        ind = randi([1 N_train_active], 1);
    else
        disp('unvalid criterion');
    end
    

	new_X_train = [X_train_initial ; X_train_active(ind,:)];
	new_y_train = [y_train_initial ; y_train_active(ind,:)];
    new_X_train_active = X_train_active;
    new_y_train_active = y_train_active;
	new_X_train_active(ind,:) = [];
	new_y_train_active(ind,:) = [];

% 	N_train_active = N_train_active - 1;
% 	N_train_initial = N_train_initial + 1;

		
	[new_W,new_phi,new_opts] = train_mod(X_train_initial,y_train_initial,K,opts,phi);


end

