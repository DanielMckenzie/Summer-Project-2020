function val = AssetRisk(x_in,function_params)
%
% =========================== INPUTS ================================= %
% x_in ...................... Point at which to evaluate
%
% ========================== OUTPUTS ================================== %
% 
% val ...................... noisy function evaluation at x_in
%
% Yuchen Lou 2021
%
 
% =========== Unpack function_params 
C = function_params.C;
m = function_params.m;
D = function_params.D;
lambda = function_params.lambda;
r = function_params.r;

val = x_in'*C*x_in/(sum(x_in))^2/2 + lambda*(min(m'*x_in/sum(x_in)-r,0))^2;
end

