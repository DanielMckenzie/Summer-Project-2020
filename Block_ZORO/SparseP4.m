function [val,grad] = SparseP4(x_in,function_params)
% Provides noisy evaluations of a sparse power 4 polynomial of the form
% \sum x_{j_i}^4
%
% =========================== INPUTS ================================= %
% x_in ...................... Point at which to evaluate
% S ......................... Suppose set of sparse quadric. Keep this the
% same
% D ......................... Ambient dimension
% sigma ..................... sigma/sqrt(D) is per component Gaussian noise level
%
% ========================== OUTPUTS ================================== %
% 
% val ...................... noisy function evaluation at x_in
% grad ..................... exact (ie no noise) gradient evaluation at
% x_in
%
% Daniel Mckenzie
% 26th June 2019
% Modified by Yuchen Lou
% August 2020
%

% =========== Unpack function_params 
sigma = function_params.sigma;
S = function_params.S;
D = function_params.D;

noise = sigma*randn(1)./sqrt(D);
val = sum(x_in(S).^4) + noise;
grad = zeros(D,1);
grad(S) = 4*x_in(S).^3;

end

