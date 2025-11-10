import random
import matplotlib.pyplot as plt
import numpy as np
from lib.metrics import discrete_score, extract_peaks_from_distance_transform, min_distance_from_pred_to_true
import torch # type: ignore

from lib.dataset_utils import augment_segment, create_segments_nonoverlapping


def visualize_ecg_and_peaks(ss, data_parsed, data_preprocessed, duration_sec=10):
    """Visualize ECG, pulse train, and R-peaks for parsed and preprocessed data side-by-side."""
    
    # Determine which datasets are available
    datasets = []
    if data_parsed is not None:
        datasets.append((data_parsed, 'Parsed'))
    if data_preprocessed is not None:
        datasets.append((data_preprocessed, 'Preprocessed'))

    n_plots = min(len(datasets), 2)  # Up to 2 plots side by side
    _fig, axes = plt.subplots(1, n_plots, figsize=(20, 5))
    
    # Ensure axes is iterable even if n_plots == 1
    if n_plots == 1:
        axes = [axes]

    for idx, (data, title_prefix) in enumerate(datasets):
        ECG = data['ECG']
        ECG_pulse = data['ECG_pulse']
        R_peaks = data['R_peaks']
        time = data['time']
        freq = data['freq']
        EEG = data['EEG']
        
        # Random segment
        start_idx = np.random.randint(0, len(time) - int(duration_sec * freq))
        end_idx = start_idx + int(duration_sec * freq)
        R_seg = R_peaks[(R_peaks >= start_idx) & (R_peaks < end_idx)]
        
        ax = axes[idx]
        ax.plot(time[start_idx:end_idx], ECG[start_idx:end_idx], label='ECG')
        ax.plot(time[start_idx:end_idx], ECG_pulse[start_idx:end_idx], label='Pulses')
        ax.scatter(time[R_seg], ECG[R_seg], s=20, c='red', label='R_peaks')
        
        for rp in R_seg:
            ax.axvline(time[rp], color='k', linestyle='--', alpha=0.8, linewidth=1)
        
        ax.set_title(f'{title_prefix} - Subject {ss} ({freq} Hz, {len(ECG)/freq:.1f}s, {EEG.shape[0]} ch, {len(R_peaks)} peaks)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_augmentation_example(data_preprocessed, demo_subj, aug_seg_len, seg_len, warp_factor_range, downsampled_frequency):
    # Build one larger segment
    X_demo, y_demo, ecg_demo = create_segments_nonoverlapping(
        data_preprocessed[demo_subj]['EEG'],
        data_preprocessed[demo_subj]['ECG_pulse'],
        data_preprocessed[demo_subj]['ECG'],
        aug_seg_len
    )
    
    # Extract center window (original)
    center_offset = (aug_seg_len - seg_len) // 2
    
    # Create 5 augmented versions
    augs = augment_segment(X_demo[0], y_demo[0], ecg_demo[0], seg_len, warp_factor_range, 5)
    
    # Plot
    t = np.arange(seg_len) / float(downsampled_frequency)
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    
    axes[0].plot(t, ecg_demo[0, center_offset:center_offset + seg_len], label='ECG', color='k', alpha=0.3, linewidth=1)
    axes[0].plot(t, y_demo[0, center_offset:center_offset + seg_len], label='Pulse', linewidth=1.5, color='C0')
    axes[0].set_title(f'Original - Subject {demo_subj}')
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc='upper right')
    
    for i, aug in enumerate(augs):
        axes[i+1].plot(t, aug['ecg'], color='k', alpha=0.3, linewidth=1)
        axes[i+1].plot(t, aug['pulse'], linewidth=1.5, color='C1')
        axes[i+1].set_title(f'Augmented #{i+1}')
        axes[i+1].grid(True, alpha=0.2)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_train_hystory(train_mae_hist, val_mae_hist, test_mae_hist, best_epoch):
    plt.figure(figsize=(10, 4))
    epochs_axis = np.arange(1, len(train_mae_hist) + 1)
    plt.plot(epochs_axis, train_mae_hist, label='Train MAE')
    plt.plot(epochs_axis, val_mae_hist, label='Val MAE')
    plt.plot(epochs_axis, test_mae_hist, label='Test MAE')
    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_predictions(name, X, y, ecg, model, seg_len, downsampled_frequency, device, subject_ids=None, k=5):
    """Visualize predictions on random segments."""
    if len(X) == 0:
        print(f"{name}: no segments")
        return
    
    k = min(k, len(X))
    idxs = np.random.choice(len(X), size=k, replace=False)
    t = np.arange(seg_len) / float(downsampled_frequency)
    
    fig, axes = plt.subplots(k, 1, figsize=(12, 2.2 * k), sharex=True)
    if k == 1:
        axes = [axes]
    
    model.eval()
    for ax, idx in zip(axes, idxs):
        xb = torch.from_numpy(X[idx:idx+1].astype(np.float32)).to(device)
        with torch.inference_mode():
            pred = model(xb).cpu().numpy().squeeze()
        
        # Plot ECG, true pulse, and prediction
        ax.plot(t, ecg[idx], label='ECG', color='k', alpha=0.25, linewidth=1, zorder=0)
        ax.plot(t, y[idx], label='True', linewidth=1.5, zorder=1)
        ax.plot(t, pred, label='Pred', alpha=0.85, zorder=2)
        
        subj_str = f" (subj={int(subject_ids[idx])})" if subject_ids is not None else ""
        ax.set_title(f"{name} segment {int(idx)}{subj_str}")
        ax.grid(True, alpha=0.2)
    
    axes[-1].set_xlabel('Time (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_filter_result(data_preprocessed, duration_sec, plot_subj):
    pre_ds = data_preprocessed[plot_subj].get('EEG_unfilt', None)
    post_ds = data_preprocessed[plot_subj]['EEG']
    fs_ds = data_preprocessed[plot_subj]['freq']
    rpeaks_ds = data_preprocessed[plot_subj]['R_peaks']

    n_channels = pre_ds.shape[0]
    n_samp = min(int(duration_sec * fs_ds), pre_ds.shape[1])  # plot up to duration_sec
    t = np.arange(pre_ds.shape[1]) / fs_ds

    # Select 5 channels evenly across the montage
    n_plot = 5
    chan_idx = np.linspace(0, n_channels - 1, n_plot, dtype=int)
    fig, axes = plt.subplots(n_plot, 1, figsize=(24, 2.6 * n_plot), sharex=True)

    for i, ch in enumerate(chan_idx):
        ax = axes[i]
        y0 = pre_ds[ch, :n_samp]
        y1 = post_ds[ch, :n_samp]

        ax.plot(t[:n_samp], y0, label='EEG pre-filter', color='C0', alpha=0.7)
        ax.plot(t[:n_samp], y1, label='EEG post-filter', color='C1', alpha=0.9)

    # R-peak vertical lines (downsampled indices)
        for rp in rpeaks_ds:
            if 0 <= rp < n_samp:
                ax.axvline(t[rp], color='r', linestyle='--', alpha=0.35, linewidth=1)
        ax.set_ylabel(f"Ch {ch}")
        if i == 0:
            ax.legend(loc='upper right', frameon=False)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Subject {plot_subj} — EEG pre/post HR-band filter with R-peaks', y=1.02)
    plt.tight_layout()
    plt.show()



def _rr_ms(peaks, fs):
    return np.diff(np.sort(peaks)) / float(fs) * 1000.0

def _sdnn_rmssd_from_peaks(peaks, fs):
    rr = _rr_ms(peaks, fs)
    sdnn = np.std(rr, ddof=1)  # use sample std
    rmssd = np.sqrt(np.mean(np.diff(rr)**2)) 
    return sdnn, rmssd

def visualize_peak_detection(dataset_loader, model, device, n_samples=10, dataset_name="Dataset", fs=500):
    """
    Visualize peak detection on random samples and show MAE + % errors (SDNN, RMSSD) above each plot.
    """
    model.eval()
    dataset = dataset_loader.dataset
    ds_len = len(dataset)
    rand_idxs = random.sample(range(ds_len), min(n_samples, ds_len))
    
    x_list, y_list, ecg_list = [], [], []
    for idx in rand_idxs:
        item = dataset[idx]
        x_list.append(item[0])
        y_list.append(item[1])
        ecg_list.append(item[2])

    
    xb = torch.stack(x_list, dim=0)
    ecg = torch.stack(ecg_list, dim=0)
    yb = torch.stack(y_list, dim=0)
    
    with torch.no_grad():
        xb_gpu = xb.to(device)
        pred = model(xb_gpu).cpu().numpy()
        y_true = yb.cpu().numpy()
        ecg = ecg.cpu().numpy()
    
    for i, ds_idx in enumerate(rand_idxs):
        pred_dist = pred[i, 0, :]
        true_dist = y_true[i, 0, :]
        ecg_signal = ecg[i, :]
        
        # Extract peaks
        pred_peaks = extract_peaks_from_distance_transform(pred_dist, min_distance=200, height_threshold=-0.4, prominence=0.02)
        pred_peaks = [elem for elem in pred_peaks if 50 < elem < 4900]

        true_peaks = extract_peaks_from_distance_transform(true_dist, min_distance=200, height_threshold=-0.4, prominence=0.02)
        true_peaks = [elem for elem in true_peaks if 50 < elem < 4900]

        # Match peaks -> distances in samples
        distances = min_distance_from_pred_to_true(pred_peaks, true_peaks)
        mae_samples = np.mean(distances) if len(distances) > 0 else np.nan
        mae_ms = mae_samples / float(fs) * 1000.0 if not np.isnan(mae_samples) else np.nan
        
        # Discrete metrics (tolerance-based)
        TP, FP, FN, disc_recall, disc_precision, disc_f1 = discrete_score(pred_peaks, true_peaks, fs=fs, tol_ms=75)
        
        # HRV metrics for this sample
        sdnn_t, rmssd_t = _sdnn_rmssd_from_peaks(true_peaks, fs)
        sdnn_p, rmssd_p = _sdnn_rmssd_from_peaks(pred_peaks, fs)

        # percent errors (absolute)
        pct_sdnn = (abs(sdnn_p - sdnn_t) / abs(sdnn_t) * 100.0) if sdnn_t != 0 else np.nan
        pct_rmssd = (abs(rmssd_p - rmssd_t) / abs(rmssd_t) * 100.0) if rmssd_t != 0 else np.nan
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
        
        # Top: ECG + peaks
        axes[0].plot(ecg_signal, linewidth=0.8, alpha=0.8, label='ECG Signal')
        axes[0].scatter(true_peaks, ecg_signal[true_peaks], marker='o', s=80, label='GT Peaks', zorder=5)
        axes[0].scatter(pred_peaks, ecg_signal[pred_peaks], marker='x', s=60, label='Pred Peaks', zorder=5)
        axes[0].set_ylabel('ECG Amplitude')
        axes[0].legend(loc='upper right')
        axes[0].grid(alpha=0.3)
        
        # Bottom: distance transforms
        axes[1].plot(true_dist, linestyle='-', linewidth=1.2, label='GT Dist Transform')
        axes[1].plot(pred_dist, linestyle='-', linewidth=1.0, label='Pred Dist Transform')
        axes[1].scatter(true_peaks, true_dist[true_peaks], marker='o', s=80, zorder=5)
        axes[1].scatter(pred_peaks, pred_dist[pred_peaks], marker='x', s=60, zorder=5)
        axes[1].set_xlabel('Samples')
        axes[1].set_ylabel('Distance Transform')
        axes[1].legend(loc='upper right')
        axes[1].grid(alpha=0.3)
        
        # Suptitle with MAE and percent errors
        suptitle = (
            f"{dataset_name} - Index {ds_idx}    "
            f"GT={len(true_peaks)} Pred={len(pred_peaks)} (TP={TP}, FP={FP}, FN={FN})\n"
            f"MAE: {mae_samples:.1f} samples ({mae_ms:.1f} ms) | "
            f"Discrete P: {disc_precision*100:.1f}%, R: {disc_recall*100:.1f}%, F1: {disc_f1*100:.1f}%\n"
            f"SDNN: true={sdnn_t:.1f}ms, pred={sdnn_p:.1f}ms, error={pct_sdnn:.1f}% | "
            f"RMSSD: true={rmssd_t:.1f}ms, pred={rmssd_p:.1f}ms, error={pct_rmssd:.1f}%"
        )
        fig.suptitle(suptitle, fontsize=10, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()