%% ===========  Using Audio Classifier for command words ===== %
% This interactive script will show you the basics of using a
% built in Matlab NN to classifier command words. The data set 
% is the speech_commands data set from Google 
% https://www.tensorflow.org/datasets/catalog/speech_commands
% =============================================================
%% Load model and a test clip
load('commandNet.mat')
[x,fs] = audioread('testSounds/0ea0e2f4_nohash_0.wav');
% fs = 16 000 = frames per second.
% x is a 16 000 dimensional vector.
sound(x,fs)  % listen to the sound. Should say "Yes"

%%
% The classifier takes a 2D spectrum as input.
% We can generate it using the helperExtractAuditoryFeatures
auditorySpect = helperExtractAuditoryFeatures(x,fs);
% plot waveform and spectrogram
subplot(2,1,1)
plot(x)
% note that this specturm is a bark spectrum, which is related to, but 
% different from, the mel spectrum.

subplot(2,1,2)
pcolor(auditorySpect')
command = classify(trainedNet,auditorySpect)

%% 
% Lets test out some wavelet transforms.
[WaveletCoefficients,f] = cwt(x,'morse');
time = 0:1/fs:1;
figure
imagesc(time,f,abs(WaveletCoefficients));
axis xy;

%% 
% A simple experiment: Lets add random noise in the wavelet domain,
% transform back to signal domain, and then see if signal is still
% correctly classified.

% Will add noise at 500 randomly chosen coefficients
RandomIndices = randi(111*16000,500,1);

Noise = sqrt(0.25/2)*randn(500,1) + sqrt(0.25/2)*1i*randn(500,1); % complex noise.
PerturbedSignal = WaveletCoefficients;
PerturbedSignal(RandomIndices) = PerturbedSignal(RandomIndices) + Noise;
figure, imagesc(time,f,abs(PerturbedSignal));
axis xy

ReconstructedSignal = icwt(PerturbedSignal,'morse');
sound(ReconstructedSignal,fs)
figure
plot(ReconstructedSignal)

%%
% Try classify perturbed signal.
AttackedAuditorySpect = helperExtractAuditoryFeatures(ReconstructedSignal',fs);
Attacked_command = classify(trainedNet,AttackedAuditorySpect)

%% Some Remarks
%  - The classifier seems to be robust to sparse random perturbations to the 
%    wavelet coefficients. It is worth pointing out that the actual
%    classification occurs in a frequency domain (Auditory Spectrum).
%    Perhaps we should reverse our reasoning, and attack in the time
%    domain?
%  - Note that the CWT transform is invertible, while the transform
%  implemented by helperExtractAuditoryFeatures is not, I believe.

%% Obtaining the logits.
% We can also obtain the logits from the classifier. These are the
% probabilities that the input signal belongs to each of the twelve classes
% considered.
[command,prob] = classify(trainedNet,auditorySpect);
prob

[command,prob] = classify(trainedNet,AttackedAuditorySpect);
prob

