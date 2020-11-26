% ====================== Wavelet Block Attack ========================= %
% This script attempts to attack a large collection of imagenet images, 
% using a wavelet attack. We want to determine the attack success rate.
% Daniel McKenzie 2020.8 and 2020.11
% Yuchen Lou 2020.8
% ===================================================================== %

clear, close all, clc;

% ============== Load the network and images ================= %
function_params.net = inceptionv3; %squeezenet;
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;
Classes = function_params.net.Layers(end).Classes; % list of all imagenet classes.

pictures = dir(fullfile('imgs', '*.jpg'));
num_images = length(pictures);

% ======================= Choose the transform ======================== %
 function_params.transform = 'db15';
 level = 3;

 % ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 20; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 360;
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
ell_2_difference_wavelet = zeros(num_images,1);
ell_0_difference_wavelet = zeros(num_images,1);
Samples_to_success = zeros(num_images,1);
%Attacked_Images_Cell = cell(num_images,3);

for i = 1:num_images
    disp(['Now attacking image number ',num2str(i)])
    target_image = imread(fullfile(cd,'imgs', pictures(i).name));
    % Next block of code deals with gray scale images by copying the gray
    % layer into the R,G and B layers.
    if length(size(target_image)) == 2
        I1 = [target_image;target_image;target_image];
        [r,c] = size(I1);
        target_image = permute(reshape(I1',[c,r/3,3]),[2,1,3]);
    end
    target_image = imresize(target_image,sz(1:2));
    target_image = double(target_image)/255;
    function_params.target_image = target_image;
    % == Store True Image
    %Attacked_Images_Cell{i,1} = target_image;
    % == Classify the unperturbed image.
    % Note that the test set of imagenet stores the true label of the
    % image in its name. This block of code extracts that.
    
    splitStr = regexp(pictures(i).name,'\.','split');
    true_idx = str2num(splitStr{1});
    function_params.true_id = true_idx;
    label = Classes(true_idx);
    function_params.label = label;
    True_Labels(i) = label;
    %[label,scores] = classify(function_params.net,255*target_image);
    %[~,idx] = sort(scores,'descend');
    %function_params.true_id = idx(1);
    %function_params.label = label;
    %True_Labels(i) = label;
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
    [Attacking_Noise, Attacked_image, f_vals, iter, num_samples_vec, Success, final_label,Wavelet_distortion_ell_0,Wavelet_distortion_ell_2] = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);
    ell_2_difference(i) = norm(target_image(:) - Attacked_image(:),2);
    ell_0_difference(i) = nnz(target_image - Attacked_image);
    Final_Labels(i) = final_label;
    Attack_Success(i) = Success;
    Samples_to_success(i) = sum(num_samples_vec);
    % == Store attacked image and noise
    %Attacked_Images_Cell{i,2} = Attacking_Noise;
    %Attacked_Images_Cell{i,3} = Attacked_image;
    % == Store distortion in wavelet domain
    ell_0_difference_wavelet(i) = Wavelet_distortion_ell_0;
    ell_2_difference_wavelet(i) = Wavelet_distortion_ell_2;
    
end

function_params.net = 'inceptionv3';  % clear this variable before saving
save([datestr(now), '.mat'])
