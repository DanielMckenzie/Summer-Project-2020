% ================== Testing Block ZORO algorithm ================ %
% Testing Block_ZORO on Sparse Quadric, for problems of increasing size.
% Daniel Mckenzie
% August 2020
% ================================================================ %

clear, close all, clc

function_handle = "SparseQuadric";

% ================================ ZORO Parameters ==================== %

% == Params for ZORO
ZORO_params.num_iterations = 10; % number of iterations. Set this high so that max_time always reached
ZORO_params.step_size = 0.5;% Step size
ZORO_params.max_time = 300;

% == Params for Block-ZORO
Block_ZORO_params.num_iterations = 10; % number of iterations. Set this high so that max_time always reached
Block_ZORO_params.step_size = 0.5;% Step size
Block_ZORO_params.max_time = 300;

% ========================= Initialize arrays ====================== %
num_tests = 5;  % TEST value, should be closer to 30 for paper.
final_val_ZORO = zeros(num_tests,1);
final_val_fullBD = zeros(num_tests,1);
final_val_fullBC = zeros(num_tests,1);
final_val_ZO_BCD = zeros(num_tests,1);
final_val_ZO_BCD_Circ = zeros(num_tests,1);


% ==================== Vary problem size ====================== %
for i = 1:num_tests
    D = 1000 + 1000*(i-1);
    s = 0.1*D;
    
    function_params.D = D;
    ZORO_params.D = D;
    
    function_params.S = datasample(1:function_params.D,s,'Replace',false); % randomly choose support.
    function_params.sigma = 0.01;  % noise level
    
    ZORO_params.sparsity = s;
    ZORO_params.x0 = randn(function_params.D,1);
    
    % == Original ZORO
    ZORO_params.Type = "Full";
    
    % == Compute delta
    H = 2*s;
    ZORO_params.delta = 2*sqrt(function_params.sigma/H);

    ZORO_params.cosamp_max_iter = ceil(4*log(function_params.D));
    [~,f_vals,~,~] = Block_ZORO(function_handle,function_params,ZORO_params);
    final_val_ZORO(i) = f_vals(end);
    
    % ===================== Now do block methods
    Block_ZORO_params.num_blocks = 10;
    Block_ZORO_params.sparsity = s;
    Block_ZORO_params.D = D;
    Block_ZORO_params.x0 = randn(function_params.D,1);
    
    % == Compute delta
    Block_H = 2*s/Block_ZORO_params.num_blocks;
    Block_ZORO_params.delta = 2*sqrt(function_params.sigma/Block_H);
    Block_ZORO_params.cosamp_max_iter = ceil(4*log(function_params.D/Block_ZORO_params.num_blocks));
    Block_ZORO_params.num_iterations = Block_ZORO_params.num_blocks*ZORO_params.num_iterations;
    
    % ================================================================== %
    
    % == Block ZORO with Rademacher z
    Block_ZORO_params.Type = "BCD";
    [~,f_vals,~,~] = Block_ZORO(function_handle,function_params,Block_ZORO_params);
    final_val_ZO_BCD(i) = f_vals(end);
    
    % == Block ZORO with circulant z
    Block_ZORO_params.Type = "BCCD";
    [~,f_vals,~,~] = Block_ZORO(function_handle,function_params,Block_ZORO_params);
    final_val_ZO_BCD_Circ(i) = f_vals(end);
    
end

save('Time_Block_ZORO_results_4.mat')
num_completed_trials = num_tests;
Problem_sizes = 1000 + 1000*[1:num_completed_trials];


semilogy(Problem_sizes, final_val_ZORO(1:num_completed_trials),'rs')
hold on
semilogy(Problem_sizes, final_val_ZO_BCD(1:num_completed_trials),'ro')
semilogy(Problem_sizes, final_val_ZO_BCD_Circ(1:num_completed_trials),'bo')
legend({'Full','Block. Coord.','Block Circ. Coord.'})
%savefig('Time_Block_ZORO_plot_4')
    
    

