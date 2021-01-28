function [x_hat,f_vals,time_vec,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params)

% Basic Implementation of ZORO with flexible sensing matrix. 
% ======================== INPUTS ================================= %
% function_handle .......... name of oracle function.
% function_params .......... any parameters required by function
% ZORO_params .............. Parameters required by ZORO.
% cosamp_params ............ Parameters required by the call to cosamp
%
% ======================== OUTPUTS =============================== %
% x_hat .................... final iterate.
% f_vals ................... vec containing f(x_k) for all k.
% time_vec ................. vec containing cumulative running time at each
% iteration.
% gradient_norm ............ vec containing ||g_k|| for all k.
% num_samples_vec .......... number of samples made at iteration k
% 
% Daniel McKenzie 2019, Yuchen Lou 2020
% 


x = ZORO_params.x0;
Type = ZORO_params.Type;
sparsity = ZORO_params.sparsity;   
num_iterations = ZORO_params.num_iterations;    
step_size = ZORO_params.step_size;
max_time = ZORO_params.max_time;
D = ZORO_params.D;


% =========== Initialize some vectors 
f_vals = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
num_samples_vec = zeros(num_iterations,1);

num_samples = 4*sparsity; 
cosamp_params.maxiterations = ZORO_params.cosamp_max_iter;
cosamp_params.tol = 0.25;
cosamp_params.sparsity = sparsity;
cosamp_params.delta = ZORO_params.delta;
oversampling_param = 1.1;

% ========== Initialize the sensing matrix

if (Type == "Full")
    % Usual Sensing
    Z = 2*(rand(num_samples,D) > 0.5) - 1;
elseif (Type == "FullCirculant")
    % Circulant Sensing Matrix
    z1 = 2*(rand(1,D) > 0.5) - 1;

    %F = dftmtx(D); % with FFT
    %Z1 = F*diag(F*z1(:))/F;

    Z1 = gallery('circul',z1); % without FFT
    SSet = datasample(1:D,num_samples,'Replace',false);
    Z = Z1(SSet,:);
else  % This handles block methods.
    if isempty(ZORO_params.num_blocks)
        error('Number of blocks not specified')
    else
        J = ZORO_params.num_blocks;
    end
    samples_per_block = ceil(oversampling_param*num_samples/J); block_size = D/J;
    if (Type == "FullBD")
        Z = zeros(num_samples,D);
        Z1 = 2*(rand(samples_per_block,block_size) > 0.5) - 1;
        for i = 0:(J-1)
            Z((samples_per_block*i+1):(samples_per_block*i+samples_per_block),(block_size*i+1):(block_size*i+block_size)) = Z1;
        end
    elseif (Type == "FullBC")
        % Full GD with block circulant sensing matrix
        z1 = 2*(rand(1,block_size) > 0.5) -1;
        Z1 = gallery('circul',z1);
        SSet = datasample(1:block_size,samples_per_block,'Replace',false);
        Z2 = Z1(SSet,:);
        Z = zeros(num_samples,D);
        for i = 0:(J-1)
            Z((samples_per_block*i+1):(samples_per_block*i+samples_per_block),(block_size*i+1):(block_size*i+block_size)) = Z2;
        end
    else  % This handles the block coordinate descent methods.
        sparsity = ceil(oversampling_param*sparsity/J); % upper bound on sparsity per block.
        cosamp_params.sparsity = sparsity;
    
        if (Type == "BCD")
            % Block Rademacher Coordinate Descent
            Z = 2*(rand(samples_per_block,block_size) > 0.5) - 1;

        elseif (Type == "BCCD")
            % Block Circulant Coordinate Descent
            z1 = 2*(rand(1,block_size) > 0.5) -1;
            Z1 = gallery('circul',z1);
            SSet = datasample(1:block_size,samples_per_block,'Replace',false);
            Z = Z1(SSet,:);
        end
    end
end

cosamp_params.Z = Z;
% ========== Now do ZORO
        
       
if (Type == "BCD") || (Type == "BCCD")  % block coordinate descent methods.
    
    for i = 1:num_iterations
        tic
        %i
        coord_index = randi(J);% randomly select a block
        block = [(coord_index-1)*block_size + 1:coord_index*block_size];
        cosamp_params.block = block;
        [f_est,grad_estimate] = BlockCosampGradEstimate(function_handle,x,cosamp_params,function_params);
        x = x - step_size*grad_estimate;
        f_vals(i) = f_est;
        num_samples_vec(i) = samples_per_block;
        if i==1
            time_vec(i) = toc;
        else
            time_vec(i) = time_vec(i-1) + toc;
        end
        if time_vec(i) >= max_time
            x_hat = x;
            % if max_time is reached, trim arrays by removing zeros
            f_vals = f_vals(f_vals ~= 0);
            time_vec = time_vec(time_vec ~=0);
            num_samples_vec = num_samples_vec(num_samples_vec~=0);
            disp('Max time reached!')
            return
        end
        if sparsity == 0
            break
        end
    end
    
else
    for i = 1:num_iterations
        tic
        %i
        [f_est,grad_estimate] = CosampGradEstimate(function_handle,x,cosamp_params,function_params);
        x = x - step_size*grad_estimate;
        f_vals(i) = f_est;
        num_samples_vec(i) = num_samples;
        if i==1
            time_vec(i) = toc;
        else
            time_vec(i) = time_vec(i-1) + toc;
        end
        if sparsity == 0
            break
        end
        if time_vec(i) >= max_time
            x_hat = x;
            % if max_time is reached, trim arrays by removing zeros
            f_vals = f_vals(f_vals ~= 0);
            time_vec = time_vec(time_vec ~=0);
            num_samples_vec = num_samples_vec(num_samples_vec~=0);
            disp('Max time reached!')
            return
        end
    end
    
end

x_hat = x;
end

