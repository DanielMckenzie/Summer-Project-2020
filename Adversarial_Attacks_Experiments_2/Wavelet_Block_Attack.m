% ====================== Wavelet Block Attack ========================= %
% This script attempts to attack a large collection of imagenet images, 
% using a wavelet attack. We want to determine the attack success rate.
% Daniel McKenzie 2020.8 and 2020.11 and 2020.12
% Yuchen Lou 2020.8
% ===================================================================== %

clear, close all, clc;

% ============== Load the network and images ================= %
function_params.net = inceptionv3; %squeezenet;
sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0;
Classes = function_params.net.Layers(end).Classes; % list of all imagenet classes.

directory = '../Adversarial_Attacks_Experiments/imgs'; % for Cottus
% directory = 'imgs' % use this for small tests on PC.
pictures = dir(fullfile(directory, '*.jpg'));
num_images = length(pictures);
Order_of_Attack = randperm(num_images);
num_attack = 1000;
num_attacked_images = 0; % Counter to keep track of how many images attacked.

% ======================= Choose the transform ======================== %
 function_params.transform = 'db15';
 level = 3;

 % ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 20; % number of iterations
ZORO_params.delta1 = 0.0005;
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 360;
ZORO_params.num_blocks = 500;
ZORO_params.Type = "BCD";
function_handle = "ImageEvaluate";

% ==== Set to untargeted attack
function_params.target_id = NaN;
%function_params.target_id = 964; % Target label id, test label "pizza"
 if isnan(function_params.target_id) == 0 
     function_params.target_label = function_params.net.Layers(end).ClassNames(function_params.target_id);
 end

% ============ Initialize vectors to keep track of success ============= %
True_Labels = zeros(num_attack,1);
Final_Labels = zeros(num_attack,1);
Attack_Success = zeros(num_attack,1);
ell_2_difference = zeros(num_attack,1);
ell_0_difference = zeros(num_attack,1);
ell_2_difference_wavelet = zeros(num_attack,1);
ell_0_difference_wavelet = zeros(num_attack,1);
Samples_to_success = zeros(num_attack,1);
Attacked_Images_Cell = cell(num_attack,3);
attacked_image_id = zeros(num_attack,1);

i = 1; % counter keeping track of which image we are currently considering.
while num_attacked_images <= 1000
    flag = 0;
    while flag == 0
        ii = Order_of_Attack(i);
        target_image = imread(fullfile(directory, pictures(ii).name));
        % Next block of code deals with gray scale images by copying the gray
    % layer into the R,G and B layers.
        if length(size(target_image)) == 2
            I1 = [target_image;target_image;target_image];
            [r,c] = size(I1);
            target_image = permute(reshape(I1',[c,r/3,3]),[2,1,3]);
        end
        target_image = imresize(target_image,sz(1:2));
        [pred_label,scores] = classify(function_params.net,target_image);
        [~,temp_idx] = sort(scores,'descend');
        pred_idx = temp_idx(1);
        splitStr = regexp(pictures(ii).name,'\.','split');
        true_idx = str2num(splitStr{1});
        true_label = Classes(true_idx);
        if pred_idx == true_idx
            flag = 1;
            disp(['Predicted label is ',pred_label])
            disp(['True label is ', true_label])
            disp('Commencing with attack')
        end
        i = i+1;
    end
    function_params.true_id = true_idx;
    Attacked_Images_Cell{i,1} = target_image; % store true image
    target_image = double(target_image)/255;
    function_params.target_image = target_image;
    function_params.label = true_label;
    True_Labels(i) = true_label;
    disp(['Now attacking image number ',num2str(ii)])
    
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
    outputs = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);
    ell_2_difference(i) = norm(outputs.Attacking_Noise(:),2);
    disp(['ell_2 norm of attacking noise in pixel domain is ',num2str(ell_2_difference(i))])
    ell_0_difference(i) = nnz(outputs.Attacking_Noise(:));
    disp(['ell_0 norm of attacking noise in pixel domain is ',num2str(ell_0_difference(i))])
    Final_Labels(i) = outputs.Final_Label;
    Attack_Success(i) = outputs.Success;
    Samples_to_success(i) = sum(outputs.num_samples_vec);
    % == Store attacked image and noise
    Attacked_Images_Cell{i,2} = outputs.Attacking_Noise;
    Attacked_Images_Cell{i,3} = outputs.Attacked_image;
    % == Store distortion in wavelet domain
    ell_0_difference_wavelet(i) = outputs.Wavelet_distortion_ell_0;
    ell_2_difference_wavelet(i) = outputs.Wavelet_distortion_ell_2;
    
    % don't forget to increment!
    num_attacked_images = num_attacked_images + 1
    
end

function_params.net = 'inceptionv3';  % clear this variable before saving
save([datestr(now), '.mat'])
