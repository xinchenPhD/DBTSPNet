% An example to get the BCI competition IV datasets 2b, only for reference
% Data from: http://www.bbci.de/competition/iv/
% using open-source toolbox Biosig on MATLAB: http://biosig.sourceforge.net/
% Just an example, you should change as you need.

% get processed T data
% function data = process(subject_index)
%% BioSig Get the data 
% T data
subject_index = 9; %1-9
session_type = 'E';
dir = ['H:\EEG_data analysis\MI-EEG public dataset\BCICIV_2b\B0',num2str(subject_index),'05',session_type,'.gdf'];
[s, HDR] = sload(dir);


% % Label 
% % classlabel = HDR.Classlabel;
% % filename = sprintf('H:\\EEG_data analysis\\MI-EEG public dataset\\2b_truelable\\B0%d04%s.mat', subject_index, session_type);
% % save(filename, 'classlabel');

labeldir = ['E:\model_updating-------\EEGNet\EEG\true_labels\B0',num2str(subject_index),'05',session_type,'.mat'];
load(labeldir);
label = classlabel;

Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000,3,120);   
for j = 1:length(Typ)
    if Typ(j) == 768
         k = k+1;
         data_1(:,:,k) = s((Pos(j)+750):(Pos(j)+1749),1:3);
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;

data = data_1;
pindex = randperm(120);
data = data(:, :, pindex);
label = label(pindex);

% 4-40 Hz
fc = 250;
fb_data = zeros(1000,3,120);

Wl = 4; Wh = 40; 
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:120
     fb_data(:,:,j) = filtfilt(b,a,data(:,:,j));
end
% 
% %z-score
eeg_mean = mean(fb_data,3);
eeg_std = std(fb_data,1,3); 
% % fb_data = (fb_data-eeg_mean)./eeg_std;
for j = 1:120
    fb_data(:,:,j) = (fb_data(:,:,j) - eeg_mean) ./ eeg_std;
end

data = fb_data;

saveDir = ['E:\model_updating-------\EEGNet\EEG\standard_BCICIV_2b_data\B0',num2str(subject_index),'05E.mat'];
save(saveDir,'data','label');

% end


