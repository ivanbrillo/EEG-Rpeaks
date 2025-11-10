from collections import defaultdict
import numpy as np
import torch
import scipy
import pandas as pd


def min_distance_from_pred_to_true(pred_peaks, true_peaks, window_len=5000):
    """
    Match predicted peaks to nearest ground truth peaks.
    Returns list of distances for matched peaks.
    """
    if len(true_peaks) == 0:
        return np.array([])

    distances = []
    for pp in pred_peaks:
        if (pp < window_len / 100 or pp > (window_len - window_len / 100)):   # useful when the true peak is cut from windowing (+/- 1%)
            continue

        dists_to_true = np.abs(true_peaks - pp)
        distances.append(np.min(dists_to_true))
    
    return np.array(distances)


def min_distance_from_true_to_pred(true_peaks, pred_peaks, max_distance, window_len = 5000):
    if len(pred_peaks) == 0:
        return np.full(len(true_peaks), max_distance)  # Approximate to max_distance, useful in first epochs when no pred are detected
    
    min_dists = []
    for tp in true_peaks:
        if (tp < window_len / 100 or tp > (window_len - window_len / 100)):   # useful when the true peak is in border region of the window
            continue

        dists = np.abs(pred_peaks - tp)
        min_dists.append(np.min(dists))

    return np.clip(min_dists, None, max_distance)  # clip useful in early stages, when no pred peaks are detected


def clip(value):
    if np.isnan(value):
        return 0

    return np.clip(value, 0, 1)


def evaluate(pred_peaks, true_peaks, f=500, window_len=5000):
    pred_peaks = np.array(pred_peaks)
    true_peaks = np.array(true_peaks)

    if len(pred_peaks) > 0 and len(true_peaks) > 0:
        diffs = np.diff(np.array(true_peaks))
        max_consecutive_diff = np.max(diffs)
        max_recall = np.mean(np.abs(diffs) / 2)

        distances = min_distance_from_pred_to_true(pred_peaks, true_peaks, window_len=window_len)
        recall_dists = min_distance_from_true_to_pred(true_peaks, pred_peaks, max_distance=max_recall, window_len=window_len)

        recall = clip(1 - np.mean(np.abs(recall_dists)) / (max_recall + 1e-4))
        precision = clip(1 - np.mean(distances) / (0.5 * max_consecutive_diff + 1e-4))
        f1 = 2 * (precision * recall) / (precision + recall + 1e-4)
        mae = clip(np.mean(distances) / f)  # MAE in seconds, clipped in 0-1

        # --- mRR (mean RR in samples) ---
        mrr_pred = np.mean(np.diff(pred_peaks))
        mrr_true = np.mean(np.diff(true_peaks))
        mrr_error = 100.0 * abs(mrr_pred - mrr_true) / (mrr_true + 1e-4)

        # --- RR intervals in samples ---
        rr_pred = np.diff(pred_peaks)
        rr_true = np.diff(true_peaks)

        # --- pRR50: percent of successive RR interval changes > 50ms ---
        thr_samples = 0.05 * f  # 50 ms in samples
        # Successive differences of RR intervals:
        succ_diff_pred = np.diff(rr_pred)  # RR[i+1] - RR[i]
        succ_diff_true = np.diff(rr_true)
        
        # Count how many absolute differences exceed threshold
        prr50_pred = 100.0 * np.sum(np.abs(succ_diff_pred) > thr_samples) / max(len(succ_diff_pred), 1)
        prr50_true = 100.0 * np.sum(np.abs(succ_diff_true) > thr_samples) / max(len(succ_diff_true), 1)
        #prr50_error = 100.0 * abs(prr50_pred - prr50_true) / (prr50_true + 1e-4)
        prr50_error = abs(prr50_pred - prr50_true)  # Absolute Error

        # --- SDNN/SDRR: standard deviation of RR intervals  ---
        sdrr_pred = np.std(rr_pred, ddof=1)  # use sample std
        sdrr_true = np.std(rr_true, ddof=1)
        sdrr_error = 100.0 * abs(sdrr_pred - sdrr_true) / (sdrr_true + 1e-4)

        # --- RMSSD: root mean square of successive differences ---
        rmssd_pred = np.sqrt(np.mean(np.diff(rr_pred)**2))
        rmssd_true = np.sqrt(np.mean(np.diff(rr_true)**2))
        rmssd_error = 100.0 * abs(rmssd_pred - rmssd_true) / (rmssd_true + 1e-4)



        return {
            "mae": mae,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # mRR
            "mrr_pred": mrr_pred,
            "mrr_true": mrr_true,
            "mrr_error": mrr_error,
            # pRR50
            "prr50_pred": prr50_pred,
            "prr50_true": prr50_true,
            "prr50_error": prr50_error,
            # SDRR
            "sdrr_pred": sdrr_pred,
            "sdrr_true": sdrr_true,
            "sdrr_error": sdrr_error,
            # RMSSD
            "rmssd_pred": rmssd_pred,
            "rmssd_true": rmssd_true,
            "rmssd_error": rmssd_error,
        }

    # fallback when no peaks: return zeros 
    return {
        "mae": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "mrr_pred": 0, "mrr_true": 0, "mrr_error": 0,
        "prr50_pred": 0, "prr50_true": 0, "prr50_error": 0,
        "sdrr_pred": 0, "sdrr_true": 0, "sdrr_error": 0,
        "rmssd_pred": 0, "rmssd_true": 0, "rmssd_error": 0,
    }



def discrete_score(pred_peaks, true_peaks, fs=500, tol_ms=75):
    thr = tol_ms / 1000  # convert ms to seconds
    tol_samples = thr * fs

    pred_peaks = np.array(pred_peaks)
    true_peaks = np.array(true_peaks)

    TP, FP, FN = 0, 0, 0

    for j in range(len(true_peaks)):
        loc = np.where(np.abs(pred_peaks - true_peaks[j]) <= tol_samples)[0]

        # false positives between R peaks
        if j == 0:
            err = np.where((pred_peaks >= 0.5 * fs + tol_samples) &
                           (pred_peaks <= true_peaks[j] - tol_samples))[0]
        elif j == len(true_peaks) - 1:
            err = np.where((pred_peaks >= true_peaks[j] + tol_samples) &
                           (pred_peaks <= 9.5 * fs - tol_samples))[0]
        else:
            err = np.where((pred_peaks >= true_peaks[j] + tol_samples) &
                           (pred_peaks <= true_peaks[j + 1] - tol_samples))[0]

        FP += len(err)

        if len(loc) >= 1:
            TP += 1
            FP += len(loc) - 1
        else:
            FN += 1

    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return TP, FP, FN, recall, precision, f1




def extract_peaks_from_distance_transform(dist_transform, min_distance=300, height_threshold=-0.4, prominence=0.035):
    """
    Extract peak locations from distance transform signal.
    Returns indices of peaks (valleys in the inverted distance transform).
    """
    peaks, properties = scipy.signal.find_peaks(
        -dist_transform, 
        distance=min_distance,
        height=height_threshold,
        prominence=prominence
    )
    return peaks


def compute_peak_mae_on_loader(dataset_loader, model, device, height_threshold=-0.2):
    """
    Compute peak-based MAE (and HRV errors) for a given data loader.
    """
    model.eval()
    all_mae = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_mrr_err = []
    all_prr50_err = []
    all_sdrr_err = []
    all_rmssd_err = []
    all_disc_recall = []
    all_disc_prec = []
    all_disc_f1 = []

    with torch.no_grad():
        for x, y, _, _ in dataset_loader:
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            y_true = y.cpu().numpy()

            for i in range(y_pred.shape[0]):
                pred_dist = y_pred[i, 0, :]
                true_dist = y_true[i, 0, :]

                pred_peaks = extract_peaks_from_distance_transform(pred_dist, min_distance=200, height_threshold=height_threshold)
                true_peaks = extract_peaks_from_distance_transform(true_dist, min_distance=200, height_threshold=height_threshold)

                res = evaluate(pred_peaks, true_peaks)
                _TP, _FP, _FN, recall, precision, f1 = discrete_score(pred_peaks, true_peaks, fs=500, tol_ms=75)

                all_mae.append(res["mae"])
                all_precision.append(res["precision"])
                all_recall.append(res["recall"])
                all_f1.append(res["f1"])
                all_mrr_err.append(res["mrr_error"])
                all_prr50_err.append(res["prr50_error"])
                all_sdrr_err.append(res["sdrr_error"])
                all_rmssd_err.append(res["rmssd_error"])

                all_disc_recall.append(recall)
                all_disc_prec.append(precision)
                all_disc_f1.append(f1)

    return (
        np.mean(all_mae),
        len(all_mae),
        np.mean(all_precision),
        np.mean(all_recall),
        np.mean(all_f1),
        np.mean(all_mrr_err),
        np.mean(all_prr50_err),
        np.mean(all_sdrr_err),
        np.mean(all_rmssd_err),
        np.mean(all_disc_prec),
        np.mean(all_disc_recall),
        np.mean(all_disc_f1)
    )

def compute_per_subject_metrics(dataset_loader, model, device, min_distance=200, height_threshold=-0.3, fs=500):
    """
    Compute peak-based metrics grouped by subject ID using existing metrics functions.
    """
    model.eval()
    
    # Store data per subject
    subject_data = defaultdict(lambda: {
        'metrics': [],
        'disc_metrics': [],
        'num_segments': 0
    })
    
    with torch.no_grad():
        for x, y, _ecg, subj_ids in dataset_loader:
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            y_true = y.cpu().numpy()
            subj_ids = subj_ids.cpu().numpy()
            
            for i in range(y_pred.shape[0]):
                subj_id = int(subj_ids[i])
                pred_dist = y_pred[i, 0, :]
                true_dist = y_true[i, 0, :]
                
                pred_peaks = extract_peaks_from_distance_transform(
                    pred_dist, min_distance=min_distance, height_threshold=height_threshold
                )
                true_peaks = extract_peaks_from_distance_transform(
                    true_dist, min_distance=min_distance, height_threshold=height_threshold
                )
                
                # Use existing evaluate function
                metrics = evaluate(pred_peaks, true_peaks, f=fs)
                subject_data[subj_id]['metrics'].append(metrics)
                
                # Use existing discrete_score function
                TP, FP, FN, recall, precision, f1 = discrete_score(pred_peaks, true_peaks, fs=fs, tol_ms=75)
                subject_data[subj_id]['disc_metrics'].append({'precision': precision, 'recall': recall, 'f1': f1})
                subject_data[subj_id]['num_segments'] += 1
    
    # Aggregate metrics per subject
    results = {}
    for subj_id, data in subject_data.items():
        if data['num_segments'] == 0:
            continue
        
        metrics_list = data['metrics']
        disc_list = data['disc_metrics']
        
        results[subj_id] = {
            'Subject_ID': subj_id,
            'Num_Segments': data['num_segments'],
            'MAE_s': np.mean([m['mae'] for m in metrics_list]),
            'Precision': np.mean([m['precision'] for m in metrics_list]),
            'Recall': np.mean([m['recall'] for m in metrics_list]),
            'F1': np.mean([m['f1'] for m in metrics_list]),
            'Disc_P_%': np.mean([m['precision'] for m in disc_list]) * 100,
            'Disc_R_%': np.mean([m['recall'] for m in disc_list]) * 100,
            'Disc_F1_%': np.mean([m['f1'] for m in disc_list]) * 100,
            'mRR_err_%': np.mean([m['mrr_error'] for m in metrics_list]),
            'pRR50_err_%': np.mean([m['prr50_error'] for m in metrics_list]),
            'SDRR_err_%': np.mean([m['sdrr_error'] for m in metrics_list]),
            'RMSSD_err_%': np.mean([m['rmssd_error'] for m in metrics_list]),
        }
    
    return results




def summary_per_subject(model, testloader, trainloader, valloader, device):
    # Compute per-subject metrics for all sets
    print("Computing per-subject metrics for all datasets...")
    test_per_subject = compute_per_subject_metrics(testloader, model, device)
    train_per_subject = compute_per_subject_metrics(trainloader, model, device)
    val_per_subject = compute_per_subject_metrics(valloader, model, device)

    # Convert to DataFrames
    df_test = pd.DataFrame.from_dict(test_per_subject, orient='index').sort_values('Subject_ID')
    df_train = pd.DataFrame.from_dict(train_per_subject, orient='index').sort_values('Subject_ID')
    df_val = pd.DataFrame.from_dict(val_per_subject, orient='index').sort_values('Subject_ID')

    # Display per-subject results
    print("\n" + "="*120)
    print("PER-SUBJECT METRICS - TEST SET")
    print("="*120)
    print(df_test.to_string(index=False))

    print("\n" + "="*120)
    print("PER-SUBJECT METRICS - TRAIN SET")
    print("="*120)
    print(df_train.to_string(index=False))

    print("\n" + "="*120)
    print("PER-SUBJECT METRICS - VALIDATION SET")
    print("="*120)
    print(df_val.to_string(index=False))

    # Comprehensive summary statistics for all datasets
    print("\n" + "="*120)
    print("SUMMARY STATISTICS - ALL DATASETS")
    print("="*120)

    datasets = [('TRAIN', df_train), ('VALIDATION', df_val), ('TEST', df_test)]

    for dataset_name, df in datasets:
        print(f"\n{dataset_name} SET (n={len(df)} subjects):")
        print(f"  MAE:              {df['MAE_s'].mean():.4f} s (std: {df['MAE_s'].std():.4f})")
        print(f"  Precision:        {df['Precision'].mean():.4f} (std: {df['Precision'].std():.4f})")
        print(f"  Recall:           {df['Recall'].mean():.4f} (std: {df['Recall'].std():.4f})")
        print(f"  F1:               {df['F1'].mean():.4f} (std: {df['F1'].std():.4f})")
        print(f"  Discrete P:       {df['Disc_P_%'].mean():.2f}% (std: {df['Disc_P_%'].std():.2f}%)")
        print(f"  Discrete R:       {df['Disc_R_%'].mean():.2f}% (std: {df['Disc_R_%'].std():.2f}%)")
        print(f"  Discrete F1:      {df['Disc_F1_%'].mean():.2f}% (std: {df['Disc_F1_%'].std():.2f}%)")
        print(f"  mRR error:        {df['mRR_err_%'].mean():.2f}% (std: {df['mRR_err_%'].std():.2f}%)")
        print(f"  pRR50 error:      {df['pRR50_err_%'].mean():.2f}% (std: {df['pRR50_err_%'].std():.2f}%)")
        print(f"  SDRR error:       {df['SDRR_err_%'].mean():.2f}% (std: {df['SDRR_err_%'].std():.2f}%)")
        print(f"  RMSSD error:      {df['RMSSD_err_%'].mean():.2f}% (std: {df['RMSSD_err_%'].std():.2f})")

    print("="*120)
