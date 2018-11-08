"""
preproc.py - functions for preprocessing the raw EKG signal,
  e.g. de-trending EKG leads using gaussian smoothing, squashing the range into 
  [-1, 1]preprocess EKG signal functions
"""
from scipy import ndimage
import numpy as np
import pandas as pd
import pyprind
from biosppy.signals import ecg as becg


def preproc_ekg_mat(X, tgrid):
    """ Preprocess (detrend and clamp to [-1, 1]) EKG matrix
    Args:
      - X     : Num_ekgs x Num_channels x T_length EKG array
      - tgrid : T_length array of EKG time (in seconds)
    """
    N, C, _ = X.shape
    xmin, xmax = np.zeros((N,C)), np.zeros((N,C))
    for n in pyprind.prog_bar(range(N)):
        for c in range(C):
            X[n, c, :], (lo, hi) = preproc_raw(X[n, c, :], tgrid)
            xmin[n,c] = lo
            xmax[n,c] = hi
    return X, xmin, xmax


def detrend_raw(Yraw, ts):
    """ univariate detrend """
    sampfreq = 1. / (ts[1] - ts[0])

    # detrend w/ a gaussian kernel
    Ykern = ndimage.gaussian_filter1d(Yraw, sigma=sampfreq/4.)
    Y = (Yraw - Ykern) + Ykern.mean()
    return Y


def detrend_raw_multi_lead(Yraw, ts):
    return np.array([detrend_raw(Yraw[i], ts) for i in range(Yraw.shape[0])])


def preproc_raw(Yraw, ts):
    """ preproc a univariate example """
    # detrend w/ a gaussian kernel
    Y = detrend_raw(Yraw, ts)

    # re-scale so that the total range is between [-1 and 1]
    ymax, ymin = Y.max(), Y.min()
    Yproc = 2*(Y - ymin)/(ymax - ymin) - 1
    return Yproc, (ymin, ymax)


def preproc_raw_multi_lead(Yraw, ts):
    """ preproc a C-lead example """
    return np.array([preproc_raw(Yraw[i], ts)[0] for i in range(Yraw.shape[0])])


def standardize_channels(X):
    """ takes N x Channel x D data matrix, and for each row,channel
    (n \in N and c \in Channel), divides X[n,c] by std(X[n,c])
    """
    Xstd          = X.std(axis=2)
    Xstandardized = X / Xstd[:,:,None]
    return Xstandardized


def squash_range(X):
    """ N x Channel x D signal, make it so each observation,
    channel are between 0 and 1 """
    Xmax    = X.max(axis=2)
    Xmin    = X.min(axis=2)
    Xrange  = Xmax - Xmin
    Xsquash = (X - Xmin[:,:,None]) / Xrange[:,:,None]
    return Xsquash


def create_beat_dataset_fixed(metadf, Xmat, tgrid, do_parallel=True, detrend=True):
    if do_parallel:
        from joblib import Parallel, delayed
        bl_list = Parallel(n_jobs=30, verbose=5)(
            delayed(segment_beat)(Xmat[idx], tgrid, alg="christov-aligned", detrend=detrend)
                for idx in range(Xmat.shape[0]))
    else:
        bl_list = []
        for idx in range(Xmat.shape[0]):
            bl_list.append(segment_beat(Xmat[idx], tgrid, alg="christov-aligned", detrend=detrend))

    # go through and determine bad idx (bad splits)
    beat_list = [b for b, _ in bl_list]
    len_list  = [l for _, l in bl_list]
    idx_bad  = np.array([ b.shape[-1] != 100 for b in beat_list ])
    idx_good = np.where(~idx_bad)[0]

    # go through each beat and construct a beat dataframe
    beat_meta, beat_lens = [], []
    for idx in idx_good:
        beat_meta += [metadf.iloc[idx]]*len(beat_list[idx])
        beat_lens.append(len_list[idx])
    beat_list = [beat_list[i] for i in idx_good]

    # stack in to dataframe + data matrix
    beat_metadf = pd.DataFrame(beat_meta)
    beat_metadf.reset_index(inplace=True)
    beat_metadf['beat_len'] = np.concatenate(beat_lens)
    beat_mat = np.row_stack(beat_list)
    beat_mat = np.rollaxis(beat_mat, 0) # transpose s.t. Nbeat x Nchannel x Nsamp
    return beat_metadf, beat_mat


def segment_beat(X, tgrid, alg="christov-aligned", grid_len=100, detrend=True):
    #X = preproc_raw_multi_lead(X, tgrid)
    if alg == "christov":
        samp_rate = 1. / (tgrid[1]-tgrid[0])
        rpeaks = becg.christov_segmenter(X[0], samp_rate)
        bmat = np.dstack([
            becg.extract_heartbeats(Xr, rpeaks=rpeaks['rpeaks'],
                sampling_rate=samp_rate, before=.3, after=.4)['templates']
            for Xr in X
            ])
        return bmat

    elif alg == "christov-aligned":
        # first detect R peaks (using preprocessed first lead)
        samp_rate = 1. / (tgrid[1]-tgrid[0])
        Xfix   = preproc_raw(X[0], tgrid)[0]
        rpeaks = becg.christov_segmenter(Xfix, samp_rate)

        # then extract irregularly lengthed beats and resample
        if detrend:
            Xdet = detrend_raw_multi_lead(X, tgrid)
        else:
            Xdet = X

        # actually extract beats
        bmat, lens = extract_irregular_beats(Xdet,
            rpeaks=rpeaks['rpeaks'], grid_len=grid_len)
        return bmat, lens

    elif alg == "gen-conv":
        raise NotImplementedError
        fit_dict = bcm.fit(X, tgrid, global_params, verbose=True)
        edbest = fit_dict['elbo_best']
        beat_starts = bcm.segment_sparse_zgrid(
            edbest['pzgrid'], tgrid, edbest['filter_time'])

        dt = tgrid[1] - tgrid[0]
        beat_width = np.int(edbest['filter_time'] / dt)
        beat_idx = np.array([ np.where(bs < tgrid)[0][0] for bs in beat_starts])
        templates = [ X[bi:(bi+beat_width)] for bi in beat_idx]

        lens = np.array([ len(t) for t in templates ])
        too_short = lens != np.max(lens)
        templates = np.row_stack([templates[i] for ts in too_short if not ts])
        return templates


def extract_irregular_beats(X, rpeaks, grid_len):
    # start points are 1/3 cycle before the rpeak, ignore first one
    lens = np.diff(rpeaks)
    if len(lens) == 0:
        return np.array([]), np.array([])
    starts = rpeaks[:-1] + np.floor((2./3.)*lens).astype(int)
    ends   = starts + lens
    if ends[-1] > X.shape[1]:
        starts, ends = starts[:-1], ends[:-1]

    # segment each beat and interpolate to a fixed grid
    bgrid   = np.linspace(0, 1, grid_len)
    beatmat = np.zeros((len(starts), X.shape[0], grid_len))
    for n, (s, e) in enumerate(zip(starts, ends)):
        beat = X[:, s:e]
        bg = np.linspace(0, 1, beat.shape[1])
        for c in range(X.shape[0]):
            beatmat[n, c, :] = np.interp(bgrid, bg, beat[c])

    return beatmat, ends-starts

