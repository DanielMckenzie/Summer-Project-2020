function [f_est,grad_est, num_samples] = TestGradEst(function_handle,x,function_params,grad_params)
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

[val,grad] = feval(function_handle,x,function_params);
f_est = val;
grad_est = grad;
num_samples = length(x);

end


