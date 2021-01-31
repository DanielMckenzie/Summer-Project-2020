% ======================== Testing CMA-ES ============================== %
% Just testing the CMA-ES script
% Daniel McKenzie
% January 2021
% ====================================================================== %

% =================== Function and oracle parameters ================ %
function_handle = "SparseQuadric";
function_params.D = 2000; % ambient dimension
ZORO_params.D = function_params.D;
s = 200;
function_params.S = datasample(1:function_params.D,s,'Replace',false); % randomly choose support.
function_params.sigma = 0.001;  % noise level
x0 = 100*randn(function_params.D,1);
OPTS.Noise.on = 1;
OPTS.MaxIter = 100;
OPTS.Popsize = 10;
SIGMA = 2;

% ==== Call optimizer

[xmin,fmin,counteval,stopflag,out,bestever] = cmaes('SparseQuadric',x0,SIGMA,OPTS,function_params);
