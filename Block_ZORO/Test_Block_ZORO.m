% ===================== Testing Block ZORO algorithm ================ %
% Test Block_ZORO, with a variety of different sensing matrix types.
% Daniel McKenzie & Hanqin Cai 2019
% Yuchen Lou 2020
% ================================================================== %

clear, close all, clc

% =================== Function and oracle parameters ================ %
function_params.D = 2000; % ambient dimension
ZORO_params.D = function_params.D;
s = 200;
function_params.S = datasample(1:function_params.D,s,'Replace',false); % randomly choose support.
function_params.sigma = 0.001;  % noise level

% ================================ ZORO Parameters ==================== %

ZORO_params.num_iterations = 50; % number of iterations
ZORO_params.sparsity = s;
ZORO_params.step_size = 0.5;% Step size
ZORO_params.x0 = 100*randn(function_params.D,1);
%ZORO_params.init_grad_estimate = norm(4*ZORO_params.x0.^3);
%ZORO_params.init_grad_estimate = 2*norm(ZORO_params.x0(function_params.S));
ZORO_params.max_time = 1e3;
ZORO_params.num_blocks = 5;
function_handle = "SparseQuadric";

% === Initial
[f0,~] = SparseQuadric(ZORO_params.x0,function_params);

% ====================== Run Full ZORO ====================== %
ZORO_params.Type = "Full";
% == Compute delta
H = 2*s;
ZORO_params.delta = 2*sqrt(function_params.sigma/H);
% == 
ZORO_params.cosamp_max_iter = ceil(4*log(function_params.D));
[x_hat,f_vals,time_vec,num_samples_vec] = Block_ZORO(function_handle,function_params,ZORO_params);

% === Plot
xx = cumsum(num_samples_vec);
semilogy([0,xx(1:end-1)], f_vals,'r*')
hold on

% ===================== Run Block ZORO ===================== %
ZORO_params.Type = "BCD";
% == Compute delta
H = 2*s/ZORO_params.num_blocks;
ZORO_params.delta = 2*sqrt(function_params.sigma/H);
ZORO_params.cosamp_max_iter = ceil(4*log(function_params.D/ZORO_params.num_blocks));
ZORO_params.num_iterations = ZORO_params.num_blocks*ZORO_params.num_iterations;  % should be num_blocks* previous number.
[x_hat_b,f_vals_b,time_vec_b,num_samples_vec_b] = Block_ZORO(function_handle,function_params,ZORO_params);

% == Plot
xx = cumsum(num_samples_vec_b);
semilogy([0,xx(1:end-1)],f_vals_b,'k*')

% ===================== Run Circulant Block ZORO ===================== %
ZORO_params.Type = "BCCD";

[x_hat_c,f_vals_c,time_vec_c,num_samples_vec_c] = Block_ZORO(function_handle,function_params,ZORO_params);

% == Plot
xx = cumsum(num_samples_vec_c);
semilogy([0,xx(1:end-1)],f_vals_c,'b*')

set(gca,'FontSize',16)
legend({'ZORO', 'Block-ZORO-R', 'Block-ZORO-RC'})