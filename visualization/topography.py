# ----------------------------2a---------------------------------
import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
save_dir = r'C:\Users\Admin\Desktop\test\\'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mat = scipy.io.loadmat(r'E:\model_updating-------\EEGNet\EEG\standard_BCICIV_2a_data\A04E.mat')
data = mat['data']  # data - (samples, channels, trials)   [1000,22,288]
label = mat['label']  # label -  (label, 1)     [288,1]

data = np.transpose(data, (2, 1, 0))  # [288,22,1000]
label = np.squeeze(np.transpose(label))  # (288,)

labels_of_interest = [1, 2, 3, 4]
label_dict = {1: 'Left Hand', 2: 'Right Hand', 3: 'Both Feet', 4: 'Tongue'}

biosemi_montage = mne.channels.make_standard_montage('biosemi64')
index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, label_id in enumerate(labels_of_interest):
    idx = np.where(label == label_id)[0]

    if len(idx) == 0:
        print(f'标签 {label_id} ({label_dict.get(label_id, "Unknown")}) 没有对应的试验，跳过绘图。')
        continue

    data_draw = data[idx]
    mean_trial = np.mean(data_draw, axis=0)  # [22,1000]
    mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)
    mean_ch = np.mean(mean_trial, axis=1)  # [22,]
    # -----------------------------
    window_size = 250
    step_size = window_size // 2
    mean_ch_list = []
    for ch_idx in range(mean_trial.shape[0]):
        channel_data = mean_trial[ch_idx, :]

        channel_means = []

        for start in range(0, len(channel_data) - window_size + 1, step_size):
            window_data = channel_data[start:start + window_size]
            channel_means.append(np.mean(window_data))

        mean_ch_list.append(np.mean(channel_means))

    mean_ch = np.array(mean_ch_list)
    # -----------------------
    evoked = mne.EvokedArray(mean_trial, info)
    evoked.set_montage(biosemi_montage)

    im, cn = mne.viz.plot_topomap(mean_ch, evoked.info, axes=axes[i], show=False)
    axes[i].set_title(f'{label_dict.get(label_id, "Label " + str(label_id))}')
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = os.path.join(save_dir, 'topo_2a_sub9.png')
plt.savefig(save_path)
plt.show()
print('所有拓扑图绘制完毕并已保存。')

#
# # ---------------------2b------------------------------
# import mne
# import numpy as np
# import scipy.io
# import matplotlib.pyplot as plt
# import os
# save_dir = r'C:\Users\Admin\Desktop\2b_raw_tp\\'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# mat = scipy.io.loadmat(r'E:\model_updating-------\EEGNet\EEG\standard_BCICIV_2b_data\B0902T.mat')
# data = mat['data']  # data - (samples, channels, trials)   [1000,22,288]
# label = mat['label']  # label -  (label, 1)     [288,1]
#
# data = np.transpose(data, (2, 1, 0))  # [288,22,1000]
# label = np.squeeze(np.transpose(label))  # (288,)
#
# labels_of_interest = [1, 2]
# label_dict = {1: 'Left Hand', 2: 'Right Hand'}
#
# biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# # channel_names = biosemi_montage.ch_names
# # c3_index = channel_names.index('C3')
# # c4_index = channel_names.index('C4')
# # cz_index = channel_names.index('Cz')
# index = [12,  47, 49]
# biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
# biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
# info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes = axes.flatten()
# for i, label_id in enumerate(labels_of_interest):
#     idx = np.where(label == label_id)[0]
#
#     if len(idx) == 0:
#         print(f'标签 {label_id} ({label_dict.get(label_id, "Unknown")}) 没有对应的试验，跳过绘图。')
#         continue
#
#     data_draw = data[idx]
#     mean_trial = np.mean(data_draw, axis=0)  # [22,1000]
#     mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)
#     mean_ch = np.mean(mean_trial, axis=1)  # [22,]
#
#     evoked = mne.EvokedArray(mean_trial, info)
#     evoked.set_montage(biosemi_montage)
#
#     im, cn = mne.viz.plot_topomap(mean_ch, evoked.info, axes=axes[i], show=False)
#     axes[i].set_title(f'{label_dict.get(label_id, "Label " + str(label_id))}')
#     fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
#
# plt.tight_layout()
# save_path = os.path.join(save_dir, 'topo_2b_sub9.png')
# plt.savefig(save_path)
# plt.show()
# print('所有拓扑图绘制完毕并已保存。')


