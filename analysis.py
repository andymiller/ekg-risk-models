"""
Main script for the paper

  A Comparison of Patient History- and EKG-based Cardiac Risk Scores
  Andrew C. Miller, Sendhil Mullainathan, Ziad Obermeyer
  Proceedings of the AMIA Summit on Clinical Research Informatics (CRI), 2018

Runs various models, saves prediction outcomes.
"""
import feather, os, sys, pickle
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

# import my code
from ekgmodels import misc, base, mlp, resnet
import experiment_data as ed
import simple_baseline as sb


#############################################################################
# Runs all models for all outcomes, saves model output                      #
#  to directory called                                                      #
#    % prediction-output/<outcome>
#
#  where <outcome> is one of 
#      - "future_afib"
#      - "stroke_6m"
#      - "troponin"
#      - "mace"
#############################################################################

run_ehr_baselines = True
run_beatnet = True
run_resnet = True
features = ["simple",
            "remark",
            #"vae",
            #"remark-vae",
            #"simple-vae",
            "simple-remark"]
            #"simple-remark-vae"]


def run_models():
    """ run all models for comparison --- takes a while """
    # feature sets and combinations to compare
    # Non Deep Methods
    if run_ehr_baselines:
        run_outcome(outcome = "future_afib", features=features)
        run_outcome(outcome = "stroke_6m", features=features)
        run_outcome(outcome = "troponin", features=features)
        run_outcome(outcome = "mace", features=features)

    # MLP outcomes
    if run_beatnet:
        run_mlp_outcome(outcome='troponin')
        run_mlp_outcome(outcome='future_afib')
        run_mlp_outcome(outcome='mace')
        run_mlp_outcome(outcome='stroke_6m')

    # full trace Resnet
    if run_resnet:
        run_resnet_outcome(outcome='future_afib')
        run_resnet_outcome(outcome='mace')
        run_resnet_outcome(outcome='stroke_6m')
        run_resnet_outcome(outcome='troponin')


def make_figures():
    """ make all figures from saved files (in "./prediction-output") """

    # first figure --- full ekg example
    plot_full_ekg_example()

    # stitch together results table
    outcomes = ['future_afib', "stroke_6m", "mace"] + \
               ['trop_3d', 'trop_7d' ,'trop_30d', 'trop_180d']
    aucdf_no_hist = results_table(outcomes=outcomes, features=features,
                          do_logreg=True, do_net=True, subset="no_history")
    aucdf = results_table(outcomes=outcomes, features=features,
                          do_logreg=True, do_net=True, subset=None)
    # compute "improvement above simple, improvement above remark, etc"

    # look at high MACE risk scores from beatnet
    plot_age_based_risk(outcome="mace")
    plot_age_based_risk(outcome="stroke_6m")
    plot_age_based_risk(outcome="trop_30d")
    plot_age_based_risk(outcome="trop_180d")
    plot_age_based_risk(outcome="future_afib")

    for cat in ['gender', 'race']:
        plot_aucs_by_category(outcome="mace", category=cat)
        plot_aucs_by_category(outcome="stroke_6m", category=cat)
        plot_aucs_by_category(outcome="trop_180d", category=cat)
        plot_aucs_by_category(outcome="trop_30d", category=cat)
        plot_aucs_by_category(outcome="future_afib", category=cat)

    plot_predictive_aucs(outcome="mace")
    plot_predictive_aucs(outcome="trop_30d")
    plot_predictive_aucs(outcome="trop_180d")
    plot_predictive_aucs(outcome="future_afib")
    plot_predictive_aucs(outcome="stroke_6m")

    # construct and save cohort table
    tabledf = ed.make_cohort_table()
    print(tabledf)
    tabledf.to_latex(os.path.join("prediction-output", "data-table.tex"), escape=False)


def run_outcome(outcome, features):
    print("\n\n========================================================")
    print("  predicting outcome %s for features %s "%(outcome, str(features)))

    # predictor results
    outcome_dir = os.path.join("prediction-output/%s"%outcome)
    if not os.path.exists(outcome_dir):
        os.makedirs(outcome_dir)

    # logistic regression --- do all features in parallel
    from joblib import Parallel, delayed
    res_list = Parallel(n_jobs=len(features), verbose=5)(
        delayed(sb.run_logistic)(outcome=outcome, features=f)
            for f in features)

    for reslr, feats in zip(res_list, features):
        print("  saving logreg with features %s "%feats)
        with open(os.path.join(outcome_dir, "lreg-%s.pkl"%feats), 'wb') as f:
            pickle.dump(reslr, f)


def run_mlp_outcome(outcome):
    # predictor results
    outcome_dir = os.path.join("prediction-output/%s"%outcome)
    if not os.path.exists(outcome_dir):
        os.makedirs(outcome_dir)

    # beat models
    beatmod = sb.run_beat_mlp(outcome=outcome, use_features=False)
    beatmod.save(os.path.join(outcome_dir, "beatnet-raw-ekg.pkl"))

    # beat with simple
    feats = "simple"
    mod = sb.run_beat_mlp(outcome=outcome, use_features=True, features=feats)
    mod.save(os.path.join(outcome_dir, "beatnet-%s.pkl"%feats))


def run_resnet_outcome(outcome):
    # predictor results
    outcome_dir = os.path.join("prediction-output/%s"%outcome)
    if not os.path.exists(outcome_dir):
        os.makedirs(outcome_dir)

    # run EKG Resnet and EKG Beatnet
    mod = sb.run_ekg_mlp(outcome=outcome, use_features=False)
    mod.save(os.path.join(outcome_dir, "resnet-raw-ekg.pkl"))

    feats = "simple"
    smod = sb.run_ekg_mlp(outcome=outcome, use_features=False)
    smod.save(os.path.join(outcome_dir, "resnet-%s.pkl"%feats))


def best_logistic_model(resdict):
    vdf = resdict['valdf']
    vauc = vdf[ vdf['metric'] == 'auc' ]
    best_idx = np.argmax(vauc['value'].values)
    best_mod = vauc['model'].iloc[best_idx]
    return best_mod


def pairwise_compare_aucs(outcome="mace", features=["simple", "remark"]):
    if "trop_" in outcome:
        outcome_dir = os.path.join("prediction-output/troponin")
        Xdf, Ydf, encdf = ed.make_dataset(
            outcome='troponin', features='simple', do_split=False)
    else:
        outcome_dir = os.path.join("prediction-output/%s"%outcome)
        Xdf, Ydf, encdf = ed.make_dataset(
            outcome=outcome, features='simple', do_split=False)
        _, _ , mdata = misc.split_data(Xdf.values, Ydf.values, encdf.split, encdf)

    split = "test"
    subset = "no_history"
    zs_mod, ys_mod = {}, {}
    for feats in features:
        print(" lreg w/ feats: ", feats)
        with open(os.path.join(outcome_dir, "lreg-%s.pkl"%feats), 'rb') as f:
            res = pickle.load(f)[outcome]

        zs = res['z%s'%split]
        Ys = Ydf.loc[zs.index][outcome]
        Xs = Xdf.loc[zs.index]
        encs = encdf.loc[zs.index]
        if subset == "no_history":
            has_past_afib = encdf.loc[zs.index]['has_afib_past']
            no_idx = (Xs['mi']==0.) & (Xs['diabetes']==0.) & \
                     (Xs['stroke']==0.) & (Xs['hypertense']==0.) & \
                     (has_past_afib == 0.) & \
                     (encs['age'] < 50.)
            if outcome == "mace":
                untested_idx = ~pd.isnull(encs['has_mace'])
                no_idx = no_idx & untested_idx
            zs = zs[no_idx]
            Ys = Ys[no_idx]
        zs_mod[feats] = zs
        ys_mod[feats] = Ys

    modfiles = ['beatnet-raw-ekg.pkl', 'beatnet-simple.pkl',
                'resnet-raw-ekg.pkl', 'resnet-simple.pkl']
    modfiles = ['beatnet-raw-ekg.pkl', 'resnet-raw-ekg.pkl', 'beatnet-simple.pkl']
    for modfile in modfiles:
        # load ekg mlp outcome
        print("  ... loading mod file %s"%modfile)
        mod = base.load_model(os.path.join(outcome_dir, modfile))
        print("    ... has %d params"%mod.num_params())
        mdf = mod.fit_res['%sdf-%s'%(split, outcome)]
        #mauc = mdf[ mdf['metric']=='auc' ]['string'].iloc[0]

        zs = mod.fit_res['z%s-enc-%s'%(split, outcome)]
        if not hasattr(zs, 'index'):
            split_idx = ['train', 'val', 'test'].index(split)
            zs = pd.Series(zs, index=mdata[split_idx].index)
        Ys = Ydf.loc[zs.index][outcome]
        Xs = Xdf.loc[zs.index]
        encs = encdf.loc[zs.index]
        if subset == "no_history":
            has_past_afib = encdf.loc[zs.index]['has_afib_past']
            no_idx = (Xs['mi']==0.) & (Xs['diabetes']==0.) & \
                     (Xs['stroke']==0.) & (Xs['hypertense']==0.) & \
                     (has_past_afib == 0.) & \
                     (encs['age'] < 50.)
            if outcome == "mace":
                untested_idx = ~pd.isnull(encs['has_mace'])
                no_idx = no_idx & untested_idx
            zs = zs[no_idx]
            Ys = Ys[no_idx]
        zs_mod[modfile] = zs
        ys_mod[modfile] = Ys

    # compare pairs
    zsekg    = zs_mod['beatnet-raw-ekg.pkl']
    zsresnet = zs_mod['resnet-raw-ekg.pkl'].loc[zsekg.index]
    zsbase   = zs_mod['simple'].loc[zsekg.index]
    #zsbase = zs_mod[0].loc[zsekg.index]
    ##zsrem  = zs_mod[1].loc[zsekg.index]
    ysbase   = Ys.loc[zsekg.index]

    sa, sb, diff = misc.bootstrap_auc_comparison(
        ysbase.values, zsbase.values, zsekg.values, num_samples=1000)
    print(" simple => beatnet ", np.percentile(diff, [2.5, 97.5]))

    sa, sb, diff = misc.bootstrap_auc_comparison(
        ysbase.values, zsrem.values, zsekg.values, num_samples=1000)
    print(" rem => beatnet ", np.percentile(diff, [2.5, 97.5]))

    sa, sb, diff = misc.bootstrap_auc_comparison(
        ysbase.values, zsresnet.values, zsekg.values, num_samples=1000)
    print(" resnet => beatnet ", np.percentile(diff, [2.5, 97.5]))

    sa, sb, diff = misc.bootstrap_auc_comparison(
        ysbase.values, zsrem.values, zsresnet.values, num_samples=1000)
    print(" rem => resnet ", np.percentile(diff, [2.5, 97.5]))


def results_table(outcomes=["future_afib"],
                  features=["simple", "remark"],
                  split="test",
                  subset = None,
                  do_logreg=True,
                  do_net=False):

    # no history subset
    auc_cols = OrderedDict()
    for outcome in outcomes:
        print("\n===== outcome %s ========"%outcome)
        if "trop_" in outcome:
            outcome_dir = os.path.join("prediction-output/troponin")
            Xdf, Ydf, encdf = ed.make_dataset(
                outcome='troponin', features='simple', do_split=False)
        else:
            outcome_dir = os.path.join("prediction-output/%s"%outcome)
            Xdf, Ydf, encdf = ed.make_dataset(
                outcome=outcome, features='simple', do_split=False)
            _, _ , mdata = misc.split_data(Xdf.values, Ydf.values, encdf.split, encdf)
        rows = []
        for feats in features:
            # lreg results
            if do_logreg:
                print(" lreg w/ feats: ", feats)
                with open(os.path.join(outcome_dir, "lreg-%s.pkl"%feats), 'rb') as f:
                    res = pickle.load(f)[outcome]

                #best_mod = best_logistic_model(res)
                #tdf = res['%sdf'%split]
                #auc = tdf[ (tdf['model']==best_mod) & (tdf['metric']=='auc') ]['string'].iloc[0]
                zs = res['z%s'%split]
                Ys = Ydf.loc[zs.index][outcome]
                Xs = Xdf.loc[zs.index]
                encs = encdf.loc[zs.index]
                if subset == "no_history":
                    has_past_afib = encdf.loc[zs.index]['has_afib_past']
                    no_idx = (Xs['mi']==0.) & (Xs['diabetes']==0.) & \
                             (Xs['stroke']==0.) & (Xs['hypertense']==0.) & \
                             (has_past_afib == 0.) & \
                             (encs['age'] < 50.)
                    if outcome == "mace":
                        untested_idx = ~pd.isnull(encs['has_mace'])
                        no_idx = no_idx & untested_idx
                    zs = zs[no_idx]
                    Ys = Ys[no_idx]
                baucs = misc.bootstrap_auc(Ys.values, zs.values, num_samples=1000)
                auc = "%2.3f [%2.3f, %2.3f]"%(
                        baucs.mean(), np.percentile(baucs, 2.5), np.percentile(baucs, 97.5))
                rows.append(auc)
                #print('features: ', feats)
                #print(res['coefdf'][best_mod].sort_values())

            #xg boost results
            else:
                with open(os.path.join(outcome_dir, "xgb-%s.pkl"%feats), 'rb') as f:
                    res = pickle.load(f)
                tdf = res['%sdf'%split]
                auc = tdf[tdf['metric']=='auc']['string'].iloc[0]
                rows.append(auc)

        if do_net:
            modfiles = ['beatnet-raw-ekg.pkl', 'beatnet-simple.pkl',
                        'resnet-raw-ekg.pkl', 'resnet-simple.pkl']
            for modfile in modfiles:
                # load ekg mlp outcome
                print("  ... loading mod file %s"%modfile)
                mod = base.load_model(os.path.join(outcome_dir, modfile))
                mdf = mod.fit_res['%sdf-%s'%(split, outcome)]
                mauc = mdf[ mdf['metric']=='auc' ]['string'].iloc[0]

                zs = mod.fit_res['z%s-enc-%s'%(split, outcome)]
                if not hasattr(zs, 'index'):
                    split_idx = ['train', 'val', 'test'].index(split)
                    zs = pd.Series(zs, index=mdata[split_idx].index)
                Ys = Ydf.loc[zs.index][outcome]
                Xs = Xdf.loc[zs.index]
                encs = encdf.loc[zs.index]
                if subset == "no_history":
                    has_past_afib = encdf.loc[zs.index]['has_afib_past']
                    no_idx = (Xs['mi']==0.) & (Xs['diabetes']==0.) & \
                             (Xs['stroke']==0.) & (Xs['hypertense']==0.) & \
                             (has_past_afib == 0.) & \
                             (encs['age'] < 50.)
                    if outcome == "mace":
                        untested_idx = ~pd.isnull(encs['has_mace'])
                        no_idx = no_idx & untested_idx
                    zs = zs[no_idx]
                    Ys = Ys[no_idx]
                print(Ys, zs)
                baucs = misc.bootstrap_auc(Ys.values, zs.values, num_samples=1000)
                mauc = "%2.3f [%2.3f, %2.3f]"%(
                    baucs.mean(), np.percentile(baucs, 2.5), np.percentile(baucs, 97.5))
                rows.append(mauc)

        auc_cols[outcome] = rows

    import copy
    fidx = copy.deepcopy(features)
    if do_net:
        fidx += ['beatnet', 'beatnet+simple',
                 'resnet', 'resnet+simple']
    aucdf = pd.DataFrame(auc_cols, index=fidx)
    return aucdf


from ekgmodels import viz
import matplotlib.pyplot as plt; plt.ion()


#########################
# EKG Plotting Code     #
#########################


def plot_full_ekg_example():
    # load outcome data and encounter dataframe
    Xdf, Ydf, encdf = ed.make_dataset(outcome='mace', features='remarks', do_split=False)
    Xmat, _, tgrid = ed.load_ekg_data(encdf=encdf, constrain_range=False)

    fig, ax = plt.figure(figsize=(12,3)), plt.gca()
    ax = viz.plot_stacked_ecg(ax, Xmat[300], linewidth=1.5) # **kwargs)
    ax.set_xlabel("time (seconds)", fontsize=16)
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    fig.savefig('prediction-output/example_ekg.png', bbox_inches='tight', dpi=150)
    plt.close("all")

    # plot segmented beats
    from ekgmodels import preproc
    beatdf, beatmat = preproc.create_beat_dataset_fixed(
        encdf, Xmat, tgrid, detrend=False)

    def savebeat(bis, name="example_beats.png"):
        fig, axarr = plt.subplots(1, 3, figsize=(6, 3.5))
        for bi, ax in zip(bis, axarr.flatten()):
            blen = beatdf['beat_len'].iloc[bi]
            ax = viz.plot_stacked_beat(ax, beatmat[bi], beatlen=blen)

        axarr[0].set_xlabel("")
        axarr[2].set_xlabel("")
        axarr[1].set_ylabel("")
        axarr[2].set_ylabel("")
        fig.savefig("prediction-output/%s"%name, bbox_inches='tight', dpi=150)
        plt.close("all")

    bis = [0, 1, 2]
    savebeat(bis, name="example_beats.png")

    bis = [300, 301, 302]
    savebeat(bis, name="example_beats-2.png")



def load_prediction_scores(outcome, feats, split="test"):
    # load data for reference
    if "trop_" in outcome:
        outcome_dir = os.path.join("prediction-output/troponin")
        Xdf, Ydf, encdf = ed.make_dataset(
            outcome='troponin', features='simple', do_split=False)
    else:
        outcome_dir = os.path.join("prediction-output/%s"%outcome)
        Xdf, Ydf, encdf = ed.make_dataset(
            outcome=outcome, features='simple', do_split=False)

    _, _ , mdata = misc.split_data(Xdf.values, Ydf.values, encdf.split, encdf)

    # load predictions
    if '.pkl' in feats:
        print("  ... loading mod file %s"%feats)
        mod = base.load_model(os.path.join(outcome_dir, feats))
        zs = mod.fit_res['z%s-enc-%s'%(split, outcome)]
        if not hasattr(zs, 'index'):
            split_idx = ['train', 'val', 'test'].index(split)
            zs = pd.Series(zs, index=mdata[split_idx].index)
    else:
        with open(os.path.join(outcome_dir, "lreg-%s.pkl"%feats), 'rb') as f:
            res = pickle.load(f)[outcome]
        zs = res['z%s'%split]

    # subselect to test split
    Ys = Ydf.loc[zs.index][outcome]
    Xs = Xdf.loc[zs.index]
    encs = encdf.loc[zs.index]
    return zs, Ys, Xs, encs


def get_age_bin_aucs(outcome="mace", feats="simple"):
    # make sure we subsample to untested for mace
    zs, Ys, Xs, encs = load_prediction_scores(
        outcome=outcome, feats=feats, split='test')
    if outcome == "mace":
        untested_idx = ~pd.isnull(encs['has_mace'])
        zs = zs[untested_idx]
        Ys = Ys[untested_idx]

    abins = pd.qcut(encs['age'], 5)
    aucs, stds = [], []
    for ab in abins.cat.categories:
        zsub = zs[ abins == ab ]
        ysub = Ys[ abins == ab ]
        baucs = misc.bootstrap_auc(ysub.values, zsub.values, num_samples=1000)
        mauc = "%2.3f [%2.3f, %2.3f]"%(
            baucs.mean(), np.percentile(baucs, 2.5), np.percentile(baucs, 97.5))
        print(ab, mauc)
        aucs.append(baucs.mean())
        stds.append(baucs.std())
    return pd.DataFrame({'auc':aucs, 'se': stds}, index=abins.cat.categories)


def get_aucs(outcome="mace", feats="simple", subset=""):
    zs, Ys, Xs, encs = load_prediction_scores(
        outcome=outcome, feats=feats, split='test')

    if outcome == "mace":
        untested_idx = ~pd.isnull(encs['has_mace'])
        zs = zs[untested_idx]
        Ys = Ys[untested_idx]
        encs = encs[untested_idx]

    # get normal aucs and no_history aucs all at once
    baucs = misc.bootstrap_auc(Ys.values, zs.values, num_samples=1000)

    # no history AUCs
    has_past_afib = encs['has_afib_past']
    no_idx = (Xs['mi']==0.) & (Xs['diabetes']==0.) & \
             (Xs['stroke']==0.) & (Xs['hypertense']==0.) & \
             (has_past_afib == 0.)
    no_aucs = misc.bootstrap_auc(Ys[no_idx].values, zs[no_idx].values, num_samples=1000)

    return (baucs.mean(), baucs.std()), (no_aucs.mean(), no_aucs.std())


def get_aucs_by_category(outcome="mace", feats="simple", category="race"):
    zs, Ys, Xs, encs = load_prediction_scores(
        outcome=outcome, feats=feats, split='test')
    if outcome == "mace":
        untested_idx = ~pd.isnull(encs['has_mace'])
        zs = zs[untested_idx]
        Ys = Ys[untested_idx]
        encs = encs[untested_idx]

    # get normal aucs and no_history aucs all at once
    if category=="race":
        w_idx = encs['race_white']==1.
        b_idx = encs['race_black']==1.
    elif category=="gender":
        w_idx = encs['sex_female']==1.
        b_idx = encs['sex_female']==0.
    else:
        raise NotImplementedError
    waucs = misc.bootstrap_auc(Ys[w_idx].values, zs[w_idx].values, num_samples=1000)
    baucs = misc.bootstrap_auc(Ys[b_idx].values, zs[b_idx].values, num_samples=1000)
    return (waucs.mean(), waucs.std()), (baucs.mean(), baucs.std())


def plot_age_based_risk(outcome="mace", feats="simple", ax=None):
    auc_simple = get_age_bin_aucs(outcome=outcome, feats="simple")
    auc_beatnet = get_age_bin_aucs(outcome=outcome, feats='beatnet-raw-ekg.pkl')
    auc_beatsimple = get_age_bin_aucs(outcome=outcome, feats='beatnet-simple.pkl')

    fig, ax = plt.figure(figsize=(5,3)), plt.gca()
    bar_width=.25
    index = np.arange(len(auc_simple))
    rs = ax.bar(index, auc_simple['auc'].values, bar_width,
        yerr = auc_simple['se'].values, label="history", color=viz.colors[0])
    rs = ax.bar(index+bar_width, auc_beatnet['auc'].values, bar_width,
        yerr = auc_beatnet['se'].values, label="beatnet", color=viz.colors[1])
    rs = ax.bar(index+2*bar_width, auc_beatsimple['auc'].values, bar_width,
        yerr=auc_beatsimple['se'].values, label="beatnet+history", color=viz.colors[2])
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(auc_simple.index.values, fontsize=15, rotation=20)
    ax.set_ylim(.5, 1.)
    ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig("prediction-output/%s-age.png"%outcome, bbox_inches='tight', dpi=150)
    plt.close("all")


def plot_predictive_aucs(outcome="mace"):
    ffeats = ['simple', 'remark', 'resnet-raw-ekg.pkl', 'beatnet-raw-ekg.pkl', 'beatnet-simple.pkl']
    names  = ['history', 'remark', 'resnet', 'beatnet', 'beatnet+history']
    aucs, aucs_std = [], []
    aucs_nohist, aucs_nohist_std = [], []
    for feats in ffeats:
        (a, astd), (n, nstd) = get_aucs(outcome=outcome, feats=feats)
        aucs.append(a)
        aucs_std.append(astd)
        aucs_nohist.append(n)
        aucs_nohist_std.append(nstd)

    print(names, aucs, aucs_std)
    #print(names, aucs_nohist

    fig, ax = plt.figure(figsize=(4, 3)), plt.gca()
    bar_width=.35
    index = np.arange(len(aucs))
    rs = ax.bar(index, aucs, bar_width, yerr=aucs_std, color=viz.colors[1],
                alpha=.85, label="test")
    #for r, c in zip(rs, viz.colors):
    #    r.set_color(c)
    rs = ax.bar(index+bar_width, aucs_nohist, bar_width, yerr=aucs_nohist_std, 
                color=viz.colors[2], label="test-no-hist", alpha=.85, hatch='///')
    #for r, c in zip(rs, viz.colors):
    #    r.set_color(c)

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(names, fontsize=15, rotation=20)
    ax.set_ylim(.5, 1.)
    ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig("prediction-output/%s-auc-comparison.png"%outcome, bbox_inches='tight', dpi=150)
    plt.close("all")


def plot_aucs_by_category(outcome="mace", category="gender"):
    ffeats = ['simple', 'remark', 'resnet-raw-ekg.pkl', 'beatnet-raw-ekg.pkl', 'beatnet-simple.pkl']
    names  = ['history', 'remark', 'resnet', 'beatnet', 'beatnet+history']
    waucs, wstd = [], []
    baucs, bstd = [], []
    for feats in ffeats:
        (wa, ws), (ba, bs) = get_aucs_by_category(
            outcome=outcome, feats=feats, category=category)
        waucs.append(wa)
        wstd.append(ws)
        baucs.append(ba)
        bstd.append(bs)

    if category == "race":
        label_a, label_b = "black", "white"
    elif category == "gender":
        label_a, label_b = "female", "male"

    fig, ax = plt.figure(figsize=(4, 3)), plt.gca()
    bar_width=.35
    index = np.arange(len(waucs))
    rs = ax.bar(index, baucs, bar_width, yerr=bstd, color=viz.colors[1],
                alpha=.85, label=label_a)
    rs = ax.bar(index+bar_width, waucs, bar_width, yerr=bstd,
            color=viz.colors[2], label=label_b, alpha=.85, hatch='///')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(names, fontsize=15, rotation=20)
    ax.set_ylim(.5, 1.)
    ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig("prediction-output/%s-auc-by-%s.png"%(outcome, category),
                bbox_inches='tight', dpi=150)
    plt.close("all")



if __name__=="__main__":
    main()
