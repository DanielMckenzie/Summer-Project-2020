function [val,label] = AudioEvaluate(x,function_params)

% x -- a complex noise?

target_audio_wavelet = function_params.target_audio_wavelet;
x_reshaped = reshape(x, function_params.shape);
perturbated_wavelet = target_audio_wavelet + x_reshaped;
perturbated_audio = icwt(perturbated_wavelet, 'morse');

AttackedAuditorySpect = helperExtractAuditoryFeatures(perturbated_audio',function_params.fs);
[label,scores] = classify(function_params.net,AttackedAuditorySpect);

[~,idx] = sort(scores,'descend');
f_tru = scores(function_params.true_id);
if isnan(function_params.target_id)
    if (idx(1) == function_params.true_id)
        f_Ntru = scores(idx(2));
    else
        f_Ntru = scores(idx(1));
    end
else
    f_Ntru = scores(function_params.target_id);
end
%val = -log(f_Ntru); % Targeted attack objective version
%val = max(-function_params.kappa, log(f_tru) - log(f_Ntru));
val = max(-function_params.kappa, f_tru - f_Ntru);

end