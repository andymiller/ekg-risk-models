import os, sys, feather, pyprind
import numpy as np
import pandas as pd
import numpy as np
from ekgmodels import preproc
from paths import PROCESSED_EHR_DATA_DIR, PROCESSED_EKG_DATA_DIR


def make_dataset(outcome="mace", features="simple", do_split=True):
    """ Load fully processed dataset --- main exposed function
    Creates three data frames of size <num_ed_encounters> x <num_features>.
    Each row is a unique ED encounter.

    Returns:
      Xdf : Covariate data for each ED encounter (e.g. vitals, labs, demographics)
      Ydf : Outcome data for each ED encounter (e.g. future troponin results)
      mdf : Side information for each ED encounter (e.g. ekg-features, complaint, etc)
    """
    # cohort information
    encdf = load_cohort()

    # construct features from different sets
    flist = []
    if "simple" in features:
        flist.append(load_simple_features(encdf))

    if "remark" in features:
        flist.append(load_remark_features(encdf))

    if "ehr" in features:
        flist.append(load_ehr_features(encdf))

    if "vae" in features:
        flist.append(load_vae_features(encdf))

    # stack feats
    X = np.column_stack([f.values for f in flist])
    Xnames = np.concatenate([ f.columns.values for f in flist])

    # set up outcome
    outcomes = [outcome]
    if outcome == "mace":
        Y = 1.*( encdf[['has_mace']].values==1. )
    elif outcome == "stroke_6m":
        Y = 1.*( encdf[['has_stroke_6m']].values==1. )
    elif outcome == "future_afib":
        Y = 1.*( (encdf[['has_afib_future']].values==1.) | 
                 (encdf[['has_afib_past']].values==1.) )
    elif outcome == "troponin":
        outcomes = ['trop_3d', 'trop_7d', 'trop_30d', 'trop_180d']
        Y = encdf[outcomes].copy()
        for c in outcomes:
            pos_idx = (Y[c] > .04)
            neg_idx = (Y[c] <= .04)
            Y[c][pos_idx] = 1.
            Y[c][neg_idx] = 0.
        Y = Y.values
    else:
        raise NotImplementedError("no %s"%outcome)

    if not do_split:
        Y = pd.DataFrame(Y, columns=outcomes, index=encdf.index)
        X = pd.DataFrame(X, columns=Xnames, index=encdf.index)
        return X, Y, encdf

    # construct data splits from indicator
    Xdata, Ydata, mdata = split_data(X, Y, encdf.split, encdf)
    return Xdata, Ydata, mdata, Xnames, encdf


def report_class_ratio(Ydata, outcome):
    print("------------------\n outcome %s stats "%outcome)
    print("   num_train (num_pos): %d (%d) "%(np.sum(~np.isnan(Ydata[0])), np.nansum(Ydata[0]) ))
    print("   num_val   (num_pos): %d (%d) "%(np.sum(~np.isnan(Ydata[1])), np.nansum(Ydata[1]) ))
    print("   num_test  (num_pos): %d (%d) "%(np.sum(~np.isnan(Ydata[2])), np.nansum(Ydata[2]) ))
    print("--------------------")


def make_dataset_outcome_with_remarks(outcome, features='remarks'):
    X, Y, Xnames, encdf = make_dataset(
        outcome=outcome, features=features, do_split=False)
    Y = pd.Series(Y, index=encdf.index, name=outcome)
    X = pd.DataFrame(X, columns=Xnames, index=encdf.index)

    if features=='remarks':
        Xfeats = ['has_bbb', 'has_st', 'has_lvh', 'has_normal_ecg',
                  'has_normal_sinus', 'has_depress', 'has_st_eleva', 'has_twave',
                  'has_aberran_bbb', 'has_jpoint_repol', 'has_jpoint_eleva',
                  'has_twave_inver', 'has_prolonged_qt']
        Ydf = pd.concat([ Y, X[Xfeats] ], axis=1)
    else:
        Ydf = pd.concat([ Y, X ], axis=1)
    return Ydf, encdf


#########################################################################
# Load main cohort --- includes information about ED encounter and 
# the value of certain outcomes
########################################################################

def load_cohort(name="mace"):
    """ Main Cohort --- each "unit" is an ED Encounter corresponding to a
    patient and a date.  The outcome is either Troponin with mulitple
    windows, MACE, or afib within a fixed window
    """
    encdf = pd.read_pickle(
        os.path.join(PROCESSED_EHR_DATA_DIR, "enc_future_afib_df.pkl"))
    encdf.set_index("ed_enc_idx", inplace=True)

    # we care only about normal looking EKGs
    normal_idx = ( (encdf['has_normal']       == 1.) | \
                   (encdf['has_normal_sinus'] == 1.) | \
                   (encdf['has_normal_ecg']   == 1.) )
    normal_idx = encdf['has_normal']==1.
    to_keep = normal_idx & \
              (encdf['has_afib']==0.) & \
              (encdf['has_poor_or_quality']==0.) & \
              (encdf['has_pacemaker']==0.)
    encdf = encdf[ to_keep ].copy()

    # assign each patient to train/val/test group (deterministically random)
    encdf = split_by_id(encdf, id_field="ptid", frac_train=.6, frac_val=.15)
    return encdf


#######################################################
# Full EKG and Beat Specific Loading Functions        #
#######################################################

def load_beat_data(encdf=None, constrain_range=False,
                   drop_poor_or_quality=False, keep_normal=False):
    Xmat, Xmeta, tgrid = load_ekg_data(encdf=encdf,
        constrain_range=constrain_range,
        drop_poor_or_quality=drop_poor_or_quality,
        keep_normal=keep_normal)

    # chunk into beats, subtract off median and return --- we already
    # detrended in the above step, so set that flag to false
    beatdf, beatmat = preproc.create_beat_dataset_fixed(
        Xmeta, Xmat, tgrid, detrend=False)
    beatmed = np.median(beatmat, -1)
    beatmat = beatmat - beatmed[:,:,None]

    # subtract of median
    return beatdf, beatmat


def load_ekg_data(encdf=None, constrain_range=False,
                  drop_poor_or_quality=False, keep_normal=False):
    """ Load matrix of EKG data --- subset to encounter ids if encounter
    dataframe is supplied """
    # load in and preprocess EKG data
    Xmeta = feather.read_dataframe(os.path.join(PROCESSED_EKG_DATA_DIR,
        "long_leads_V1_II_V5_metadf.feather"))
    Xmat  = np.load(os.path.join(PROCESSED_EKG_DATA_DIR,
        "long_leads_V1_II_V5.npy"))
    tgrid = np.arange(Xmat.shape[-1]) / 100.  # data are sampled at 100 Hz
    if constrain_range:
        Xmat, _, _ = preproc.preproc_ekg_mat(Xmat, tgrid)

    # drop NAN ptids and poor or quality EKGs
    idx_bad = (pd.isnull(Xmeta['ptid']))
    if drop_poor_or_quality:
        idx_bad = idx_bad | (Xmeta['has_poor_or_quality']==1.)
    if keep_normal:
        has_normal = (Xmeta['has_normal']==1.) | \
                     (Xmeta['has_normal_sinus']==1.) | \
                     (Xmeta['has_normal_ecg'] == 1.)
        idx_bad = idx_bad | (~has_normal)

    if encdf is None:
        Xmeta = Xmeta[~idx_bad]
        Xmat  = Xmat[Xmeta['npy_index']]
        Xmeta['npy_index'] = np.arange(Xmat.shape[0], dtype=np.int)
        Xmeta['ptid'] = Xmeta['ptid'].astype(int).astype(str)
        Xmeta = split_by_id(Xmeta, id_field="ptid", frac_train=.75, frac_val=.15)
    else:
        Xmat = Xmat[ encdf['npy_index'] ]
        Xmeta = encdf

    return Xmat, Xmeta, tgrid


###############################################
# Predicive features --- vae, simple, etc     #
###############################################

def load_vae_features(encdf):
    vae_feats = pd.read_pickle(os.path.join(
      PROCESSED_EHR_DATA_DIR, "enc_vae_features.pkl"))
    vae_feats = vae_feats.loc[encdf.index]
    vae_feats =  fillna_with_mean(vae_feats, encdf.split)
    return vae_feats


def load_simple_features(encdf):
    """ load in demographic, some LVS, some previous diagnoses (diabetes,
    smoking, hypertension, heart failure. Match exactly by
    encdf['ed_enc_idx'] """
    # simple features, should be indexed by ed_enc_idx
    simple_feats = pd.read_pickle(os.path.join(PROCESSED_EHR_DATA_DIR,
        "enc_future_simple_features.pkl"))

    #simple_feats.set_index("ed_enc_idx", inplace=True)
    # match encdf subset
    simple_feats = simple_feats.loc[encdf.index]

    # remove "measured" features
    cpresent = simple_feats.columns.str.contains("-present")
    cs = simple_feats.columns[~cpresent]
    simple_feats = simple_feats[cs]

    # standardize age
    simple_feats['age'] = \
      standardize_features_by_train(simple_feats['age'].values, encdf.split)

    return simple_feats


def load_remark_features(encdf):
    # afib cardiology remark features
    print("loading remark features")
    remark_feats = pd.read_pickle(os.path.join(PROCESSED_EHR_DATA_DIR,
        "enc_future_remark_features.pkl"))
    remark_feats.set_index('ed_enc_idx', inplace=True)

    # isolate feature columns
    cols = ['PR_interval', 'QRS_duration', 'QT_interval', 'QTc_interval',
            'P_axes', 'R_axes', 'T_axes',
            'has_bbb', 'has_st', 'has_pacemaker', 'has_lvh', 'has_normal',
            'has_normal_ecg', 'has_normal_sinus', 'has_depress',
            'has_st_eleva', 'has_twave', 'has_aberran_bbb',
            'has_jpoint_repol', 'has_jpoint_eleva', 'has_twave_inver',
            'has_prolonged_qt', 'has_lead_reversal']
            #, 'has_poor_or_quality'
    remark_feats = remark_feats.loc[encdf.index][cols]

    # fillna's with average values
    remark_feats = fillna_with_mean(remark_feats, encdf.split)

    # standardize features
    cfeats = ['PR_interval', 'QRS_duration', 'QT_interval',
              'QTc_interval', 'P_axes', 'R_axes', 'T_axes']
    for c in cfeats:
        remark_feats[c] = standardize_features_by_train(
            remark_feats[c].values, encdf.split)

    return remark_feats


def load_ehr_features(encdf):
    featdir = os.path.join(PROCESSED_EKG_DATA_DIR, "standard-features/")
    lookback="1y"

    # remove zero columns from each
    def remove_zero_columns(df):
        nonempty = df.columns[df.sum()!=0]
        return df[nonempty]

    # load in dias and meds from past year
    print("   ... dia features")
    diadf = feather.read_dataframe(
        os.path.join(featdir, "ed-dia-counts-%s.feather"%lookback))
    diadf.set_index("ed_enc_idx", inplace=True)
    diadf = remove_zero_columns(diadf)
    diadf = diadf.loc[encdf.index]

    # load in med
    print("   ... med features")
    meddf = feather.read_dataframe(
        os.path.join(featdir, "ed-med-counts-%s.feather"%lookback))
    meddf.set_index("ed_enc_idx", inplace=True)
    meddf = remove_zero_columns(meddf)
    meddf = meddf.loc[encdf.index]

    # long term LVS
    lvdf = feather.read_dataframe(
        os.path.join(featdir, "ed-lvs-binary-features-%s.feather"%lookback))
    lvdf.set_index("ed_enc_idx", inplace=True)
    lvdf = lvdf.loc[encdf.index]

    # long term labs
    print("   ... lab features")
    labdf = feather.read_dataframe(
        os.path.join(featdir, "ed-lab-binary-features-%s.feather"%lookback))
    labdf.set_index("ed_enc_idx", inplace=True)
    labdf = remove_zero_columns(labdf)
    labdf = labdf.loc[encdf.index]

    # concat and return
    print(" concatenating and returning ")
    ehr_feats = pd.concat([lvdf, diadf, meddf, labdf], axis=1)
    ehr_feats.fillna(0., inplace=True)
    return ehr_feats


##############################
# Data Splitting Helpers     #
##############################

def split_by_id(beatdf, id_field='ptid', frac_train=.6, frac_val=.15):
    """ Deterministically splits the beatdf by _patient_ """
    empis = np.sort(beatdf[id_field].unique())
    print("Splitting %d unique patients"%len(empis))

    # deterministic split
    rs = np.random.RandomState(0)
    perm_idx = rs.permutation(len(empis))
    num_train = int(frac_train*len(empis))
    num_val   = int(frac_val*len(empis))
    train_idx = perm_idx[:num_train]
    val_idx   = perm_idx[num_train:(num_train+num_val)]
    test_idx  = perm_idx[(num_train+num_val):]
    empis_train = empis[train_idx]
    empis_val   = empis[val_idx]
    empis_test  = empis[test_idx]
    print(" ... patient splits: %d train, %d val, %d test "%(
      len(empis_train), len(empis_val), len(empis_test)))

    # make dictionaries 
    train_dict = {e: "train" for e in empis_train}
    val_dict   = {e: "val"   for e in empis_val}
    test_dict  = {e: "test"  for e in empis_test}
    split_dict = {**train_dict, **val_dict, **test_dict}

    # add train/val test split to each
    split = []
    for e in pyprind.prog_bar(beatdf[id_field]):
        split.append(split_dict[e])

    beatdf['split'] = split
    return beatdf


def split_data(X, Y, split, metadf=None):
    train_idx      = split == "train"
    val_idx        = split == "val"
    test_idx       = split == "test"
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xval,   Yval   = X[val_idx],   Y[val_idx]
    Xtest,  Ytest  = X[test_idx],  Y[test_idx]
    if metadf is None:
        return (Xtrain, Xval, Xtest), (Ytrain, Yval, Ytest)
    mtrain, mval, mtest = metadf[train_idx], metadf[val_idx], metadf[test_idx]
    return (Xtrain, Xval, Xtest), (Ytrain, Yval, Ytest), (mtrain, mval, mtest)


def combine_date_and_time(date, time):
    return pd.to_datetime(date.astype(str) + " " + time)


def standardize_features_by_train(X, split):
    Xmu, Xstd = np.nanmean(X[split=="train"], 0), np.nanstd(X[split=="train"], 0)
    X = (X-Xmu) / Xstd
    return X


def fillna_with_mean(df, split):
    for c in df.columns:
        cmu = df[ split=="train" ][c].mean()
        df[c].fillna(cmu, inplace=True)
    return df

###################################
# Data table for this cohort      #
###################################

def make_cohort_table():
    # construct feature information, including demographics
    Xdf, Ydf, encdf = make_dataset(
        outcome='mace', features='simple', do_split=False)

    # construct all outcomes
    Ylist = []
    outcome_types = ['future_afib', 'stroke_6m', 'mace', 'troponin']
    for outcome in outcome_types:
        _, Ydf, _ = make_dataset(
            outcome=outcome, features='simple', do_split=False)
        Ylist.append(Ydf)
    Ydf = pd.concat(Ylist, axis=1)

    # split Xdev/ Xtest
    Xdata, Ydata, mdata = split_data(Xdf.values, Ydf.values, encdf.split, encdf)
    Ydev    = Ydf  [ (encdf.split=='train') | (encdf.split=='val')]
    Xdev    = Xdf  [ (encdf.split=='train') | (encdf.split=='val') ]
    encdev  = encdf[ (encdf.split=='train') | (encdf.split=='val') ]
    Ytest   = Ydf[ encdf.split=='test' ]
    Xtest   = Xdf[ encdf.split=='test' ]
    enctest = encdf[ encdf.split == 'test']

    # no history idx
    has_past_afib = encdf.loc[Xtest.index]['has_afib_past']
    no_idx = (Xtest['mi']==0.) & (Xtest['diabetes']==0.) & \
             (Xtest['stroke']==0.) & (Xtest['hypertense']==0.) & \
             (has_past_afib == 0.) #& \
             #(encdf.loc[Xtest.index]['age'] < 50.)
    Xno_hist  = Xtest[no_idx]
    Yno_hist  = Ytest[no_idx]
    enc_nohist = enctest[no_idx]

    # summary statistics (simple features/demographcis)
    from collections import OrderedDict
    tdict = OrderedDict()
    tdict['\# EKGs'] = [len(Xdev), len(Xtest), len(Xno_hist)]
    tdict['Patient demographics'] = ["", "", ""]
    tdict['\hspace{3mm} \# unique patients'] = [np.unique(df['ptid']).shape[0]
        for df in [encdev, enctest, enc_nohist]]
    tdict['\hspace{3mm} mean age (sd)'] = ["%2.1f (%2.1f)"% (d['age'].mean(), d['age'].std())
        for d in [encdev, enctest, enc_nohist]]
    tdict['\hspace{3mm} \# female (\%)'] = \
      ["%2.0f (%2.1f \%%)"%(d['sex_female'].sum(), 100.*d['sex_female'].mean())
        for d in [encdev, enctest, enc_nohist]]

    tdict['Patient with history (\%)'] = ["", "", ""]
    for cat, name in zip(['mi', 'diabetes', 'hypertense', 'stroke', 'smoke'],
                         ['mi', 'diabetes', 'hypertension', 'stroke', 'smoking']):
        tdict['\hspace{3mm} %s'%name] = \
          ["%2.0f (%2.1f \%%)"% (d[cat].sum(), 100.*d[cat].mean())
            for d in [Xdev, Xtest, Xno_hist]]

    #outcomes
    tdict['Outcomes: total positive (\%)'] = ["", "", ""]
    for cat, name in zip(['future_afib', 'stroke_6m', 'mace'],
                         ['future-afib', 'stroke-6m', 'mace-6m']):
        tdict['\hspace{3mm} %s'%name] = \
          ["%2.0f (%2.1f \%%)"% (d[cat].sum(), 100.*d[cat].mean())
              for d in [Ydev, Ytest, Yno_hist]]

    # trop outcome
    tdict["troponin: (\# labs observed, \% positive)"] = ["", "", ""]
    for cat, name in zip(['trop_7d', 'trop_30d', 'trop_180d'],
                         ['trop-7d', 'trop-30d', 'trop-180d']):
        tdict['\hspace{3mm} %s'%name] = \
          ["%2.0f (%2.0f, %2.1f \%%)"%(
            d[cat].sum(), (~pd.isnull(d[cat])).sum(), 100.*d[cat].mean())
            for d in [Ydev, Ytest, Yno_hist]]

    # table, tex i
    tdf = pd.DataFrame(tdict,
        index=["Development", "Test", "Test (no history)"]).T
    tdf.index.name = "Dataset Characteristics"
    return tdf
