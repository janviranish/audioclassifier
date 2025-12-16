% Load and split data
ads = audioDatastore("C:\Users\xxxxx\environmental_sounds", ...
    "IncludeSubfolders", true, "LabelSource", "foldernames");
[adsTrain, adsTest] = splitEachLabel(ads, 0.8, "randomized");

% Extract training features
nTrain = numel(adsTrain.Files);
featDim = 64;  % mel bands
trainFeatures = zeros(nTrain, featDim);
trainLabels = adsTrain.Labels;

for i = 1:nTrain
    [x, fs] = audioread(adsTrain.Files{i});
    trainFeatures(i,:) = extract_features(x, fs);
end

mdl = fitcknn(trainFeatures, trainLabels, "NumNeighbors", 5, "Standardize", true);

% Extract test features
nTest = numel(adsTest.Files);
testFeatures = zeros(nTest, featDim);

for i = 1:nTest
    [x, fs] = audioread(adsTest.Files{i});
    testFeatures(i,:) = extract_features(x, fs);
end

% Visualise performance
predictions = predict(mdl, testFeatures);
accuracy = sum(predictions == adsTest.Labels) / nTest * 100;
fprintf("Accuracy: %.1f%%\n", accuracy);
confusionchart(adsTest.Labels, predictions);

% Feature extraction
function feat = extract_features(x, fs)
    if size(x,2) > 1, x = mean(x,2); end
    x = x / (max(abs(x)) + 1e-6);
    
    S = melSpectrogram(x, fs, ...
        "Window", hamming(round(0.025*fs),"periodic"), ...
        "OverlapLength", round(0.015*fs), ...
        "NumBands", 64);
    
    feat = mean(log10(S + 1e-6), 2).';

end
