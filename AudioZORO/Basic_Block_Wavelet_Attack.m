% ====================== Basic Wavelet Block Audio Attack =================== %
% Yuchen Lou 2021.1
% ===================================================================== %

 clear,close all, clc;

% ============== Load the network, choose the image ================= %
load('commandNet.mat')
function_params.net = trainedNet;

sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0.1;

[sound_proto,fs] = audioread('testSounds/down/0c40e715_nohash_0.wav');
% Perform wavelet transform
[target_audio_wavelet,~] = cwt(sound_proto,'morse');
function_params.target_audio_wavelet = target_audio_wavelet;


% == Classify the unperturbed audio.
AuditorySpect = helperExtractAuditoryFeatures(sound_proto,fs);
[label,scores] = classify(trainedNet,AuditorySpect);

[~,idx] = sort(scores,'descend');
function_params.fs = fs;
function_params.true_id = idx(1);
function_params.label = label;
function_params.target_id = NaN;

if isnan(function_params.target_id) == 0
    function_params.target_label = function_params.net.Layers(end).ClassNames(function_params.target_id);
end
disp(label);


function_params.D = length(target_audio_wavelet(:));
ZORO_params.D = function_params.D;
% ================================ ZORO Parameters ==================== %
ZORO_params.num_iterations = 25; % number of iterations
ZORO_params.delta1 = 0.001;
ZORO_params.init_grad_estimate = 100;
ZORO_params.max_time = 180;
ZORO_params.num_blocks = 2000;
ZORO_params.Type = "BCD";
function_handle = "AudioEvaluate";

function_params.shape = size(target_audio_wavelet);
function_params.epsilon = 5; % Box Constraint params
ZORO_params.sparsity = 0.05*ZORO_params.D; % 0.05*ZORO_params.D
ZORO_params.step_size = 0.01; % Step size
ZORO_params.x0 = zeros(function_params.D,1);


% ====================== run ZORO Attack ======================= %
outputs = BCD_ZORO_Adversarial_Attacks(function_handle,function_params,ZORO_params);

% Play attacked sound
Attacked_Audio = outputs.Attacked_Audio;
sound(Attacked_Audio,fs);