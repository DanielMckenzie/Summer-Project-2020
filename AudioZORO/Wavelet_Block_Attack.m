% ====================== Wavelet Block Attack ========================= %
% This script attacks audio samples in a wavelet domain.
% Attack a large number of samples in order to determine the 
% Attack Success Rate.
% Daniel McKenzie 2020.8 and 2020.11 and 2020.12 and 2021.1
% Yuchen Lou 2020.8
% ===================================================================== %

clear, close all, clc;

% ============== Load the network and images ================= %
load('commandNet.mat')
function_params.net = trainedNet;

sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0.1;
Classes = function_params.net.Layers(end).Classes; % list of all imagenet classes.

%directory = '../validation'; % for Cottus
%directory = 'testSounds'; % use this for small tests on PC.
directory = 'Sounds'; % use this for small tests on PC.
sounds = dir(fullfile(directory, '**/*.wav'));
num_sounds = length(sounds);
rng(1,'twister')
Order_of_Attack = randperm(num_sounds);
num_attack = 500;
num_attacked_sounds = 1; % Counter to keep track of how many images attacked.

 % ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 30; % number of iterations
ZORO_params.delta1 = 0.001;
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 180;
ZORO_params.num_blocks = 2000;
ZORO_params.Type = "BCD";
function_handle = "AudioEvaluate";

% ==== Set to untargeted attack
function_params.target_id = NaN;
%function_params.target_id = 964; % Target label id, test label "pizza"
 if isnan(function_params.target_id) == 0 
     function_params.target_label = function_params.net.Layers(end).ClassNames(function_params.target_id);
 end

% ============ Initialize vectors to keep track of success ============= %
True_Labels = cell(num_attack,1);
Final_Labels = zeros(num_attack,1);
Attack_Success = zeros(num_attack,1);
Attack_Volume = zeros(num_attack,1);
ell_2_difference_wavelet = zeros(num_attack,1);
ell_0_difference_wavelet = zeros(num_attack,1);
Samples_to_success = zeros(num_attack,1);
Attacked_Sounds_Cell = cell(num_attack,3);
attacked_sounds_id = zeros(num_attack,1);

% ============ TO DO:
%   2/ Measure the loudness of the noise in (relative decibels):
% rel_db = 20*log(max_i{Attacking_noise_i}) - 20*log(max_i{Orig_Audio_i})
%   record this in a vector.
%   3. make sure the attack code below aligns with that in the
%   Basic_Block_Wavelet_Attack.m

i = 1; % counter keeping track of which image we are currently considering.
while num_attacked_sounds <= num_attack
    flag = 0;
    while flag == 0
        ii = Order_of_Attack(i);
        [target_audio,fs] = audioread(fullfile(sounds(ii).folder, sounds(ii).name));
        
        % == Convert to spectrogram
        AuditorySpect = helperExtractAuditoryFeatures(target_audio,fs);
        
        [pred_label,scores] = classify(function_params.net,AuditorySpect);
        [~,temp_idx] = sort(scores,'descend');
        pred_idx = temp_idx(1);
        splitStr = regexp(sounds(ii).folder,'/','split');
        true_label = splitStr{end};
        true_idx = find(Classes == true_label);
        if pred_idx == true_idx
            flag = 1;
            disp(['Predicted label is ',pred_label])
            disp(['True label is ', true_label])
            disp('Commencing with attack')
        end
        i = i + 1;
    end
    function_params.true_id = true_idx;
    Attacked_Sounds_Cell{num_attacked_sounds,1} = target_audio; % store true sound
    [target_audio_wavelet,~] = cwt(target_audio,'morse');
    function_params.target_audio_wavelet = target_audio_wavelet;
    function_params.label = true_label;
    True_Labels{num_attacked_sounds} = true_label;
    disp(['Now attacking audio clip number ',num2str(ii)])
    
    % ====== Additional Parameters
    function_params.fs = fs;
    function_params.epsilon = 5; % Box Constraint params
    function_params.D = length(target_audio_wavelet(:));
    function_params.shape = size(target_audio_wavelet);
    ZORO_params.D = function_params.D;
    ZORO_params.sparsity = 0.025*ZORO_params.D;
    ZORO_params.step_size = 0.05; % Step size. 3e-4 is value used by Kaidi Xu
    ZORO_params.x0 = zeros(function_params.D,1);
    % ====================== run ZORO Attack ======================= %
    outputs = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);
    
    % == Store attacked image and noise
    Attacked_Sounds_Cell{num_attacked_sounds,2} = outputs.Attacking_Noise;
    Attacked_Sounds_Cell{num_attacked_sounds,3} = outputs.Attacked_Audio;
    % == Store distortion in wavelet domain
    ell_0_difference_wavelet(num_attacked_sounds) = outputs.Wavelet_distortion_ell_0;
    ell_2_difference_wavelet(num_attacked_sounds) = outputs.Wavelet_distortion_ell_2;
    % == Compute and store attack loudness
    Attack_Volume(num_attacked_sounds) = 20*log(max(abs(outputs.Attacking_Noise))) - 20*log(max(abs(target_audio)));
    % == Store Attack success parameters
    Attack_Success(num_attacked_sounds) = outputs.Success;
    Final_Labels(num_attacked_sounds) = outputs.Final_Label;
    Samples_to_success(num_attacked_sounds) = sum(outputs.num_samples_vec);
    % don't forget to increment!
    num_attacked_sounds = num_attacked_sounds + 1
    
end

function_params.net = 'blank';  % clear this variable before saving
save([datestr(now), '.mat'])
