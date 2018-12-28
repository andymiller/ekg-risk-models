import os, sys, feather
import numpy as np
import pandas as pd
import numpy as np
sys.path.append('../common/')
from ekgmodels import mlp, resnet, misc
from ekgmodels.mlp import logit
import experiment_data as ed


############################
# Simple feature sets      #
############################

def run_logistic(outcome="mace", features="simple"):
    Xdf, Ydf, encdf = \
        ed.make_dataset(outcome=outcome, features=features, do_split=False)

    # remove missing values
    resdict = {}
    for c in Ydf.columns:
        print("  working on outcome %s "%c)

        # split and purge nan
        Yc = Ydf[c]
        idx_nan = pd.isnull(Yc)
        Xdata, Ydata, mdata = misc.split_data(Xdf[~idx_nan].values,
            Yc[~idx_nan].values, encdf.split[~idx_nan], encdf[~idx_nan])

        cs = np.logspace(-2.5, 1, 20)
        res = misc.run_logistic(Xdata, Ydata,
            Xdf.columns.tolist(), penalty="l2", cs=cs)

        # make sure ztrain/ zval/ ztest are linked to an ed_enc_idx
        res['ztrain'] = pd.Series(logit(np.exp(res['lnp_train'])),
                                  index=mdata[0].index, name='ztrain')
        res['zval']   = pd.Series(logit(np.exp(res['lnp_val'])),
                                  index=mdata[1].index, name='zval')
        res['ztest']  = pd.Series(logit(np.exp(res['lnp_test'])),
                                  index=mdata[2].index, name='ztest')
        resdict[c] = res

    return resdict


#############################
# Raw EKG waveform data     #
#############################

remark_feats = ['has_bbb', 'has_st', 'has_lvh', 'has_normal_ecg',
                'has_normal_sinus', 'has_depress', 'has_st_eleva', 'has_twave',
                'has_aberran_bbb', 'has_jpoint_repol', 'has_jpoint_eleva',
                'has_twave_inver', 'has_prolonged_qt']


def run_beat_mlp(outcome='mace', use_features=False, features='simple'):
    # load outcome and ekg remarks
    Xdf, Ydf, encdf = ed.make_dataset(outcome=outcome, features='remarks', do_split=False)
    outcome_names = Ydf.columns
    Ydf = pd.concat([Ydf, Xdf[remark_feats]], axis=1)
    Xsimp, _, _ = ed.make_dataset(outcome=outcome, features=features, do_split=False)
    assert np.all(Xsimp.index == Ydf.index)

    # load outcome data and encounter dataframe
    # beat data ---- beatdf now has 'index' as a column (ed_enc_idx) 
    beatdf, beatmat = ed.load_beat_data(encdf=encdf, constrain_range=True)

    # expand outcome and simple features
    Ybeat      = Ydf.loc[ beatdf['index'] ].values
    Xsimp_beat = Xsimp.loc[ beatdf['index'] ].values

    # now split the data
    Xdata, Ydata, mdata = misc.split_data(beatmat, Ybeat, beatdf.split, beatdf)
    Wdata, _, _ = misc.split_data(Xsimp_beat, Ybeat, beatdf.split, beatdf)
    Xtrain, Xval, Xtest = Xdata
    _, n_channels, n_samples = Xtrain.shape
    ntrain, nval, ntest = (len(x) for x in Xdata)

    # isolate feature columns
    if use_features == True:
        print("Adding %s features to EKG in last layer!"%features)
        Xtrain = np.column_stack([Xtrain.reshape(ntrain, -1), Wdata[0]])
        Xval   = np.column_stack([Xval.reshape(nval, -1)    , Wdata[1]])
        Xtest  = np.column_stack([Xtest.reshape(ntest, -1) ,  Wdata[2]])
        # isolate feature columns
        model = mlp.BeatDeepWideMlpClassifier(
                data_dim = n_channels * n_samples,
                total_n_outputs = Ydata[0].shape[1],
                verbose = False,
                hdims   = [500, 500],
                h_dim   = 100,
                dim_wide = Wdata[0].shape[1])
    else:
        Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
        Xval   = Xval.reshape((Xval.shape[0], -1))
        Xtest  = Xtest.reshape((Xtest.shape[0], -1))
        model  = mlp.BeatMlpClassifier(data_dim = n_channels*n_samples,
                                       n_outputs=Ydata[0].shape[1],
                                       verbose=False,
                                       hdims=[500, 500])

    # Run optimization
    model.fit((Xtrain, Xval, Xtest), Ydata,
              lr_reduce_interval=20,
              epochs=80,
              learning_rate=1e-3,
              class_weights=False)

    # contract z values back down to encounter-level, save classification data
    for oi, outcome in enumerate(outcome_names):
        for si, split in enumerate(['train', 'val', 'test']):
            zs = pd.Series(model.fit_res['z%s'%split].numpy()[:,oi],
                           index = mdata[si]['index'])
            zs_mu = zs.groupby(level=0, sort=False).mean()
            yt    = Ydf.loc[zs_mu.index][outcome]
            model.fit_res['z%s-enc-%s'%(split, outcome)] = zs_mu
            model.fit_res['y%s-enc-%s'%(split, outcome)] = yt
            nan_idx = pd.isnull(yt)
            model.fit_res['%sdf-%s'%(split, outcome)] = \
                misc.classification_stats(yt[~nan_idx].values, zs_mu[~nan_idx].values)

    return model


def run_ekg_mlp(outcome='mace', use_features=False, features='simple'):
    # load outcome and ekg remarks
    Xdf, Ydf, encdf = ed.make_dataset(outcome=outcome, features='remarks', do_split=False)
    outcome_names = Ydf.columns
    Ydf = pd.concat([Ydf, Xdf[remark_feats]], axis=1)
    Xsimp, _, _ = ed.make_dataset(outcome=outcome, features=features, do_split=False)
    assert np.all(Xsimp.index == Ydf.index)

    # load outcome data and encounter dataframe
    # beat data ---- beatdf now has 'index' as a column (ed_enc_idx) 
    Xmat, _, tgrid = ed.load_ekg_data(encdf=encdf, constrain_range=True)

    # now split the data
    Xdata, Ydata, mdata = misc.split_data(Xmat, Ydf.values, encdf.split, encdf)
    Wdata, _, _ = misc.split_data(Xsimp, Ydf.values, encdf.split, encdf)
    Xtrain, Xval, Xtest = Xdata
    _, n_channels, n_samples = Xtrain.shape
    ntrain, nval, ntest = (len(x) for x in Xdata)

    # now split the data
    Xtrain, Xval, Xtest = (X[:,:,:-1] for X in Xdata) # convent size
    Ytrain, Yval, Ytest = (Yy for Yy in Ydata)

    if use_features:
        # if we put phatdict in here, add to 
        ntrain, nval, ntest = (len(x) for x in Xdata)
        Xtrain = np.column_stack([Xtrain.reshape(ntrain, -1), Wdata[0]])
        Xval   = np.column_stack([Xval.reshape(nval, -1)  ,   Wdata[1]])
        Xtest  = np.column_stack([Xtest.reshape(ntest, -1) ,  Wdata[2]])
        model = resnet.EKGDeepWideResNetClassifier(n_channels=3,
                                           n_samples=1000-1,
                                           total_n_outputs=Ytrain.shape[1],
                                           num_rep_blocks=8,
                                           kernel_size=16,
                                           h_dim = 100,
                                           dim_wide = Wdata[0].shape[1],
                                           verbose=False)
    else:
        model = resnet.EKGResNetClassifier(n_channels=3,
                                           n_samples=1000-1,
                                           n_outputs=Ytrain.shape[1],
                                           num_rep_blocks=8,
                                           kernel_size=16,
                                           verbose=False)
    model.fit((Xtrain, Xval, Xtest), (Ytrain, Yval, Ytest),
              epochs=80,
              learning_rate=1e-3,
              lr_reduce_interval=10,
              class_weights=False)

    # contract z values back down to encounter-level, save classification data
    for oi, outcome in enumerate(outcome_names):
        for si, split in enumerate(['train', 'val', 'test']):
            zs  = model.fit_res['z%s'%split].numpy()[:, oi]
            yt  = Ydata[si][:, oi]
            nan_idx = pd.isnull(yt)
            model.fit_res['%sdf-%s'%(split, outcome)] = \
                misc.classification_stats(yt[~nan_idx], zs[~nan_idx], num_samples=1000)
            model.fit_res['z%s-enc-%s'%(split, outcome)] = \
                pd.Series(zs, index=mdata[oi].index)
            model.fit_res['y%s-enc-%s'%(split, outcome)] = \
                pd.Series(yt, index=mdata[oi].index)

    return model


