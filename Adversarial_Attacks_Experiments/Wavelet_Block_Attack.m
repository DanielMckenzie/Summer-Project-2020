% ====================== Wavelet Block Attack ========================= %
% This script attempts to attack a large collection of imagenet images, 
% using a wavelet attack. We want to determine the attack success rate.
% Daniel McKenzie 2020.8 and 2020.11
% Yuchen Lou 2020.8
% ===================================================================== %

clear, close all, clc;

% ============== Load the network and images ================= %
function_params.net = squeezenet;
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;

pictures = dir(fullfile('imgs_test', '*.JPEG'));
num_images = 3; % length(pictures);

% ======================= Choose the transform ======================== %
 function_params.transform = 'db9';
 level = 3;

 % ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 20; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 180;
ZORO_params.num_blocks = 100;
ZORO_params.Type = "BCD";
function_handle = "ImageEvaluate";

% ==== Set to untargeted attack
function_params.target_id = NaN;
%function_params.target_id = 964; % Target label id, test label "pizza"
 if isnan(function_params.target_id) == 0 
     function_params.target_label = function_params.net.Layers(end).ClassNames(function_params.target_id);
 end

% ============ Initialize vectors to keep track of success ============= %
True_Labels = zeros(num_images,1);
Final_Labels = zeros(num_images,1);
Attack_Success = zeros(num_images,1);
ell_2_difference = zeros(num_images,1);
ell_0_difference = zeros(num_images,1);

for i = 1:num_images
    target_image = imread(fullfile(cd,'imgs_test', pictures(i).name));
    target_image = imresize(target_image,sz(1:2));
    target_image = double(target_image)/255;
    function_params.target_image = target_image;
    % == Classify the unperturbed image.
    [label,scores] = classify(function_params.net,255*target_image);
    [~,idx] = sort(scores,'descend');
    function_params.true_id = idx(1);
    function_params.label = label;
    True_Labels(i) = label;
    disp(label);
    [c,shape] = wavedec2(target_image,level,function_params.transform);
    % ====== Additional Parameters
    function_params.shape = shape;
    function_params.epsilon = 5;
    function_params.D = length(c);
    ZORO_params.D = length(c);
    ZORO_params.sparsity = 0.05*ZORO_params.D;
    ZORO_params.step_size = 3;% Step size
    ZORO_params.x0 = zeros(function_params.D,1);
    % ====================== run ZORO Attack ======================= %
    [Attacking_Noise, Attacked_image, f_vals, iter, num_samples_vec, Success, final_label] = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);
    ell_2_difference(i) = norm(target_image(:) - Attacked_image(:),2);
    ell_0_difference(i) = nnz(target_image - Attacked_image);
    Final_Labels(i) = final_label;
    Attack_Success(i) = Success;
end







figure, imshow(Attacked_image)

% === Plot function value
xx = cumsum(num_samples_vec);
f_vals = f_vals(f_vals ~= 0);
figure();
semilogy(xx, f_vals,'r*')