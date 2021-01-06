% ===================== Testing Block ZORO algorithm ================ %
% Test Block_ZORO, with a variety of different sensing matrix types.
% Daniel McKenzie & Hanqin Cai 2019
% Yuchen Lou 2020
% ================================================================== %

clear, close all, clc

% =================== Function and oracle parameters ================ %
load('port5.mat','Correlation','stdDev_return','mean_return')
C = Correlation .* (stdDev_return * stdDev_return');
nAssets = numel(mean_return); r = 0.002;

function_params.D = nAssets; % ambient dimension
function_params.C = C;
function_params.m = mean_return;
function_params.r = r;
function_params.lambda = 5000;

% ================================ ZORO Parameters ==================== %

s = 50;

ZORO_params.num_iterations = 50; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.sparsity = s;
ZORO_params.step_size = 0.01;% Step size
ZORO_params.x0 = randn(function_params.D,1);
%ZORO_params.init_grad_estimate = norm(4*ZORO_params.x0.^3);
ZORO_params.init_grad_estimate = 2*norm(ZORO_params.x0);
ZORO_params.max_time = 1e3;
ZORO_params.num_blocks = 5;
function_handle = "AssetRisk";

% ====================== Run Full ZORO ====================== %
%ZORO_params.Type = "Full";
%ZORO_params.cosamp_max_iter = ceil(4*log(function_params.D)); % ceil(4*log(function_params.D));
%[x_hat,f_vals,time_vec,gradient_norm,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);

% === Plot
%xx = cumsum(num_samples_vec);
%semilogy(xx, abs(f_vals),'r*')
%hold on

% ===================== Run Block ZORO ===================== %
ZORO_params.Type = "BCD";
ZORO_params.cosamp_max_iter = ceil(4*log(function_params.D/ZORO_params.num_blocks));
ZORO_params.num_iterations = ZORO_params.num_blocks*ZORO_params.num_iterations;  % should be num_blocks* previous number.
[x_hat_b,f_vals_b,time_vec_b,gradient_norm_b,num_samples_vec_b] = Block_ZORO(function_handle,function_params,ZORO_params);

% == Plot
xx = cumsum(num_samples_vec_b);
semilogy(xx,abs(f_vals_b),'b*')

legend({'Full Gradient ZORO', 'Block Coordinate Descent ZORO'})