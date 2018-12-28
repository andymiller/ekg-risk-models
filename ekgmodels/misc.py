import numpy as np
import pyprind
from sklearn.metrics import roc_auc_score, precision_score, \
                            recall_score, f1_score, r2_score, \
                            average_precision_score
import pandas as pd

#########################################################
# Various metrics and bootstrap estimation wrappers     #
#########################################################
from sklearn.metrics import roc_auc_score, r2_score, f1_score
def roc_auc_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    return roc_auc_score(y[~nan_idx], p[~nan_idx])

def f1_score_nan(y, p):
    nan_idx = np.isnan(y)
    return f1_score(y[~nan_idx], p[~nan_idx])

def r2_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    return r2_score(y[~nan_idx], p[~nan_idx])



def bootstrap_auc(ytrue, ypred, fun=roc_auc_score, num_samples=100):
    # make nan safe
    nan_idx = np.isnan(ytrue)
    ytrue = ytrue[~nan_idx]
    ypred = ypred[~nan_idx]

    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        auc = fun(ytrue[idx], ypred[idx])
        #auc = np.max([auc, 1-auc])
        samps.append(auc)

    return np.array(samps)


def bootstrap_average_precision_score(ytrue, ypred, num_samples=100):
    return bootstrap_auc(ytrue, ypred,
        fun=average_precision_score, num_samples=num_samples)


def bootstrap_auc_comparison(ytrue, ypreda, ypredb, num_samples=100):
    samps_a, samps_b, diff = [], [], []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        auc_a = roc_auc_score(ytrue[idx], ypreda[idx])
        auc_b = roc_auc_score(ytrue[idx], ypredb[idx])
        samps_a.append(auc_a)
        samps_b.append(auc_b)
        diff.append(auc_a-auc_b)
    return samps_a, samps_b, diff


def bootstrap_prec_recall_f1(ytrue, ypred, num_samples=100):
    psamps, rsamps, fsamps = [], [], []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        psamps.append( precision_score(ytrue[idx], ypred[idx]) )
        rsamps.append( recall_score(ytrue[idx], ypred[idx]) )
        fsamps.append( f1_score(ytrue[idx], ypred[idx]) )

    return np.array(psamps), np.array(rsamps), np.array(fsamps)


def bootstrap_corr(x, y, num_samples=100):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        samps.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return np.array(samps)


def bootstrap_summary(y, fun=np.mean, num_samples=1000):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        samps.append(fun(y[idx]))
    return np.array(samps)


def bootstrap_r2(ytrue, ypred, num_samples=1000):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        samps.append(r2_score(ytrue[idx], ypred[idx]))
    return np.array(samps)


def classification_stats(ytrue, ypred, num_samples=1000):
    """ construct a dataframe that summarizes classification """
    # compute aucs, prec, rec, f1
    auc  = roc_auc_score(ytrue, ypred)

    yp   = ypred > .5
    prec = precision_score(ytrue, yp)
    rec  = recall_score(ytrue, yp)
    f1   = f1_score(ytrue, yp)

    auc_samps = bootstrap_auc(ytrue, ypred, num_samples=num_samples)
    prec_samps, rec_samps, f1_samps = \
        bootstrap_prec_recall_f1(ytrue, yp, num_samples=num_samples)

    alo, ahi = np.percentile(auc_samps, [2.5, 97.5])
    plo, phi = np.percentile(prec_samps, [2.5, 97.5])
    rlo, rhi = np.percentile(rec_samps, [2.5, 97.5])
    flo, fhi = np.percentile(f1_samps, [2.5, 97.5])

    metrics = ["auc", "precision", "recall", "f1"]
    vals    = [auc, prec, rec, f1]
    ci_los  = [alo, plo, rlo, flo]
    ci_his  = [ahi, phi, rhi, fhi]
    strs    = ["%2.4f [%2.4f, %2.4f]"%(i, ilo, ihi) 
               for i, ilo, ihi in zip(vals, ci_los, ci_his)]
    return pd.DataFrame({"metric": metrics,
                         "value" : vals,
                         "ci_lo" : ci_los,
                         "ci_hi" : ci_his,
                         "string": strs})


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


def samples_in_box(xlo, xhi, ylo, yhi, samps):
    """ returns indices of samples that fall in (xlo,xhi); (ylo,yhi) box """
    idx = np.where( (samps[:,0] > xlo) & (samps[:,0] < xhi) & \
                    (samps[:,1] > ylo) & (samps[:,1] < yhi) )[0]
    return idx


def run_logistic(Xdata, Ydata, Xnames, penalty='l1', cs=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import l1_min_c
    Xtrain, Xval, Xtest = Xdata
    Ytrain, Yval, Ytest = Ydata

    # determine range of regularization weights
    if cs is None:
        cs = l1_min_c(Xtrain, Ytrain, loss='log') * np.logspace(0, 7, 20)
    print("Testing complexity penalties: ", cs)

    # instantiate model
    solver = "saga" if penalty=="l1" else "sag"
    solver = "liblinear"
    print("  using solver: ", solver)
    clf = LogisticRegression(C=1.0, penalty=penalty, tol=1e-6, solver=solver)
    coefs_ = {}
    vstats, tstats = [], []
    best_vauc, best_c = 0., None
    for c in cs:
        # model name
        name = "logreg-%s-%2.5f"%(penalty, c)
        print("  fitting %s "%name)
        clf.set_params(C=c)
        clf.fit(Xtrain, Ytrain)

        # store coefficients
        coefs_[name] = np.concatenate([clf.intercept_.copy(),
                                       clf.coef_.ravel().copy()])
        num_nonzero = np.sum(np.abs(coefs_[name]) > .001)
        print("  ... num nonzero: %d"%num_nonzero)

        # compute validation loss/statistics
        Yval_pred = np.exp(clf.predict_log_proba(Xval)[:,1])
        cdf = classification_stats(Yval, Yval_pred)
        cdf['model'] = [name] * len(cdf)
        vstats.append(cdf)
        print("  ... val auc ", cdf[cdf['metric']=='auc']['string'].iloc[0])
        vauc = cdf[cdf['metric']=='auc']['value'].iloc[0]
        if vauc > best_vauc:
            best_vauc, best_c = vauc, c

        # compute validation loss/statistics
        Ytest_pred = np.exp(clf.predict_log_proba(Xtest)[:,1])
        tdf = classification_stats(Ytest, Ytest_pred)
        tdf['model'] = [name] * len(tdf)
        tdf['c']     = [c] * len(tdf)
        tstats.append(tdf)
        print("  ... test auc ", tdf[tdf['metric']=='auc']['string'].iloc[0])

    print(" retraining best model ")
    clf.set_params(C=c)
    clf.fit(Xtrain, Ytrain)
    best_coef = np.concatenate([clf.intercept_.copy(), clf.coef_.ravel().copy()])

    # construct resdict and return
    coefdf = pd.DataFrame(coefs_, index = ['intercept'] + Xnames)
    valdf  = pd.concat(vstats, axis=0)
    testdf = pd.concat(tstats, axis=0)
    resdict = {"valdf": valdf, "testdf": testdf, "coefdf": coefdf,
               "best_mod": clf,
               "best_coef": best_coef,
               "lnp_train": clf.predict_log_proba(Xtrain)[:,1],
               "lnp_val"  : clf.predict_log_proba(Xval)[:,1],
               "lnp_test" : clf.predict_log_proba(Xtest)[:,1]}
    return resdict


