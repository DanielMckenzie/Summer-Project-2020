function [f_est,grad_est, num_samples] = SPSA(function_handle,x,function_params,grad_params)
% 
% Dummy function to test the benchmarking script. 
% This returns a true gradient as the gradient approximation.
% Use as a template for other grad estimators.
% 
% INPUTS
% ========================================================================
% function_handle .... Name of obj. function. Should be "SparseQuadric"
% x .................. point at which to evaluate gradient
% function_params .... Struct containing any params needed by obj. function
% grad_params ........ Struct containing any params needed by grad
% estimation function.
% 
% OUTPUTS 
% ========================================================================
% f_est ............. Value of objective function at x
% grad_est .......... approx. to gradient of objective function at x
% num_samples ....... number of function queries used.
%
% Daniel McKenzie January 2021.
%

num_samples = grad_params.num_samples;
sampling_radius = grad_params.sampling_radius;
D = function_params.D;

% == Generate averaged stochastic gradient
grad = 0;
f_est = 0;

% random perturbations
Z = mvnrnd(zeros(D,1),eye(D),num_samples)';

for i = 1:num_samples
    [val1,~] = feval(function_handle,x,function_params);
    [val2,~] = feval(function_handle,x + sampling_radius*Z(:,i),function_params);
    grad = grad + (val2 - val1)*Z(:,i);
    f_est = f_est + val1;
end

grad_est = grad/num_samples;
f_est = f_est/num_samples;
num_samples = 2*num_samples;

end


