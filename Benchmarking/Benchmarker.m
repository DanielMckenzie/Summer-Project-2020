function [x_hat,f_vals,time_vec,num_samples_vec] = Benchmarker(function_handle,grad_handle,function_params,grad_params, ZORO_params)

% Flexible optimizer designed to be used with a variety of gradient
% estimation algorithms.
% INPUTS
% ========================================================================
% function_handle ..... Name of objective function
% grad_handle ......... Name of gradient estimation algorithm
% function_params ..... Struct to hold all parameters needed for objective
% function.
% grad_params .......... Struct to hold all parameters needed for grad.
% estimator.
% ZORO_params ......... Struct to hold various parameters, e.g. step_size 
%
% HanQin Cai, Yuchen Lou and Daniel McKenzie 2021
% 



x = ZORO_params.x0;
num_iterations = ZORO_params.num_iterations;    
step_size = ZORO_params.step_size;
max_time = ZORO_params.max_time;

% =========== Initialize some vectors 
f_vals = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
num_samples_vec = zeros(num_iterations,1);

for i = 1:num_iterations
    % The next line passes the relevant quantities to the function in the 
    % .m file "grad_handle.m"
    [f_est,grad_est, num_samples] = feval(grad_handle,function_handle,x,function_params,grad_params); 
    f_vals(i) = f_est;
    x = x - step_size*grad_est;
    num_samples_vec(i) = num_samples;
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
end
x_hat = x;       
end

