""" MLP and fitting functions """
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, r2_score, f1_score
from ekgmodels import base

# nn transformations
def logit(p):    return np.log(p) - np.log(1.-p)
def sigmoid(a):  return 1. / (1. + np.exp(-a))
def relu(x):     return np.maximum(0, x)
def softplus(a): return np.logaddexp(a, 0)


######################
# Beat MLP Models    #
######################

class BeatMlpMixed(base.Model):
    def __init__(self, **kwargs):
        super(BeatMlpMixed, self).__init__(**kwargs)
        self.bin_dims = kwargs.get("binary_dims")
        self.cont_dims = kwargs.get("continuous_dims")
        self.net = MLP(**kwargs)
        self.cont_loss = base.NanMSELoss(size_average=True)
        self.bin_loss  = base.NanBCEWithLogitsLoss(size_average=True)
        self.is_continuous = True

    def lossfun(self, data, target):
        logit = self.forward(data)
        bin_loss  = self.bin_loss(logit[:, self.bin_dims], target[:, self.bin_dims])
        cont_loss = self.cont_loss(logit[:, self.cont_dims], target[:, self.cont_dims])
        print("bin/cont: ", bin_loss.mean()[0,0], cont_loss.mean()[0,0])
        return torch.mean(bin_loss + cont_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest, 
                                    **kwargs)
        return self.fit_res


class BeatMlpClassifier(base.Model):
    def __init__(self, **kwargs): 
        super(BeatMlpClassifier, self).__init__(**kwargs)
        self.net = MLP(**kwargs)
        #self.loss = nn.BCEWithLogitsLoss(size_average=True)
        self.loss = base.NanBCEWithLogitsLoss(size_average=True)
        self.is_continuous = False

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest, 
                                    **kwargs)
        return self.fit_res


class BeatDeepWideMlpClassifierFixed(BeatMlpClassifier):
    def __init__(self, **kwargs):
        super(BeatDeepWideMlpClassifierFixed, self).__init__(**kwargs)
        self.dim_wide = kwargs.get("dim_wide")

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        last_out = data[:, -self.dim_wide:]
        ekg_data = data[:, :-self.dim_wide].contiguous()
        return self.net(ekg_data) + last_out


class BeatDeepWideMlpClassifier(BeatMlpClassifier):
    """ wide and deep EKG beat mlp.  Expects the last `dim_wide` dimensions
        to be linearly added into the final prediction. 
    """
    def __init__(self, **kwargs):
        # first, make the EKGResnet have 100 outputs
        self.h_dim     = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide  = kwargs.get("dim_wide")
        kwargs['n_outputs'] = self.h_dim
        super(BeatDeepWideMlpClassifier, self).__init__(**kwargs)
        self.wide_out = nn.Linear(self.h_dim + self.dim_wide, self.n_outputs, bias=True)

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        # ekg transform
        last_out = data[:, -self.dim_wide:]
        ekg_data = data[:, :-self.dim_wide].contiguous()
        # wide + EKG representation
        zout = torch.cat([self.net(ekg_data), last_out], 1)
        return self.wide_out(zout)


class BeatMlpRegression(base.Model):
    def __init__(self, **kwargs): 
        super(BeatMlpRegression, self).__init__(**kwargs)
        self.net = MLP(**kwargs)
        #self.loss = nn.MSELoss(size_average=True)
        self.loss = base.NanMSELoss(size_average=True)
        self.is_continuous = True

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest, 
                                    **kwargs)
        return self.fit_res


class BeatDeepWideMlpRegression(BeatMlpRegression):
    """ wide and deep EKG beat mlp.  Expects the last `dim_wide` dimensions
        to be linearly added into the final prediction. 
    """
    def __init__(self, **kwargs):
        self.h_dim     = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide  = kwargs.get("dim_wide")
        kwargs['n_outputs'] = self.h_dim
        super(BeatDeepWideMlpRegression, self).__init__(**kwargs)
        self.wide_out = nn.Linear(self.h_dim + self.dim_wide, self.n_outputs, bias=True)

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        # ekg transform
        last_out = data[:, -self.dim_wide:]
        ekg_data = data[:, :-self.dim_wide].contiguous()
        # wide + EKG representation
        zout = torch.cat([self.net(ekg_data), last_out], 1)
        return self.wide_out(zout)


class BeatGroupedRegression(BeatMlpRegression):
    """
      Regress some target on a high-dimensional EKG Beat and side information
      vector.  Side information vector modulates the weights of the last
      layer for the EKG --- so if P-dimensional side info
      S = [race_white, race_black, ..., ...] then the final predictor looks like

          yhat = (S_{1xP} * W_{PxH}) * mlp(EKG)_{H} + bias

      args:
        - h_dim           : dimension of last hidden layer in Beat MLP
        - total_n_outputs : dimensionality of target variable
        - dim_wide        : dimension of side information vector we condition on
    """
    def __init__(self, **kwargs):
        self.h_dim = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide = kwargs.get("dim_wide")
        self.include_last_bias = kwargs.get("include_last_bias", False)
        kwargs['n_outputs'] = self.h_dim
        super(BeatGroupedRegression, self).__init__(**kwargs)
        self.side_info_mat = nn.Linear(self.dim_wide, self.h_dim+1, bias=False)

        #if self.include_last_bias:
        #self.last_bias = nn.Parameter(torch.FloatTensor([0.]), requires_grad=True)
        #self.last_bias = nn.Parameter(taorch.randn(1, self.dim_wideFloatTensor([0.]), requires_grad=True)

    def forward(self, data):
        side_info = data[:, -self.dim_wide:]
        ekg_data  = data[:, :-self.dim_wide].contiguous()
        last_w = self.side_info_mat(side_info)
        last_h = self.net(ekg_data)
        zout  = torch.sum(last_w[:,:-1]*last_h, dim=1, keepdim=True) + last_w[:,-1][:, None]
        return zout


class MLP(base.Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.data_dim   = kwargs.get("data_dim")
        self.output_dim = kwargs.get("n_outputs")
        self.hdims      = kwargs.get("hdims", [50,50,50])
        self.dropout_p  = kwargs.get("dropout_p", .5)

        # compute log Pr(Y | h_last)
        # construct generative network for beats
        sizes = [self.data_dim] + self.hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            #modules.append(nn.BatchNorm1d(dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout_p))

        modules.append(nn.Linear(sizes[-1], self.output_dim))
        self.forward_net = nn.Sequential(*modules)
        self.init_params()
        self.is_binary=True
        self.is_continuous=False

    def forward(self, x):
        return self.forward_net(x.view(x.shape[0], -1))


#class MLPRegression(MLP):
#    def __init__(self, *args, **kwargs):
#    #data_dim, output_dim, hdims=[50, 50, 50]):
#        super(MLPRegression, self).__init__(*args, **kwargs)
#        self.pred_loss = nn.MSELoss(size_average=True, reduce=False)
#        self.is_continuous = True
#
#    def init_params(self):
#        for p in self.parameters():
#            p.data.uniform_(-.25, .25)



##############################
# standard multi-layer MLP   #
##############################

#def fit_mlp(Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, **kwargs):
#    # args
#    #kwargs = {}
#    do_cuda       = kwargs.get("do_cuda", torch.cuda.is_available())
#    batch_size    = kwargs.get("batch_size", 256)
#    epochs        = kwargs.get("epochs", 50)
#    output_dir    = kwargs.get("output_dir", "./")
#    model_type    = kwargs.get("model_type", "MLP")
#    weight_decay  = kwargs.get("weight_decay", 1e-5)
#    learning_rate = kwargs.get("learning_rate", 1e-3)
#    hdims         = kwargs.get("hdims", [50, 50, 50])
#    side_dim      = kwargs.get("side_dim", None)
#    wide_weights  = kwargs.get("wide_weights", None)
#    #Wtrain, Wval, Wtest = kwargs.get("Wdata", (None, None, None))
#    print("-------------------")
#    print("fitting mlp: ", kwargs)
#
#    # set up data
#    kwargs = {'num_workers': 1, 'pin_memory': True} if do_cuda else {}
#    train_data = torch.utils.data.TensorDataset(
#        torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain))
#    train_loader = torch.utils.data.DataLoader(train_data,
#        batch_size=batch_size, shuffle=True)
#    val_data = torch.utils.data.TensorDataset(
#        torch.FloatTensor(Xval), torch.FloatTensor(Yval))
#    val_loader = torch.utils.data.DataLoader(val_data,
#        batch_size=batch_size, shuffle=False)
#
#    # instantiate model
#    #side_dim = None if Wtrain is None else Wtrain.shape[1]
#    model = make_model(model_type=model_type,
#                       data_dim=Xtrain.shape[1],
#                       side_dim=side_dim,
#                       output_dim=train_data.target_tensor.size(1),
#                       hdims=hdims)
#    if do_cuda:
#        model.cuda()
#        model.is_cuda = True
#
#    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
#    optimizer = optim.Adam(plist, lr=learning_rate, weight_decay=weight_decay)
#    #optimizer = optim.SGD(plist, lr=learning_rate, weight_decay=weight_decay)
#
#    best_val_loss = np.inf
#    best_val_state = None
#    prev_val_loss = np.inf
#    train_loss = []
#    val_loss   = []
#    print("{:10}  {:10}  {:10}  {:10}  {:10}". \
#        format("Epoch", "train-loss", "val-loss",
#                        "train-auc", "val-auc",
#                        "train-f1", "val-f1"))
#    for epoch in range(1, epochs + 1):
#        tloss, ttrue, tpred, tauc, tf1 = run_epoch(epoch, model, train_loader, optimizer,
#            do_cuda, only_compute_loss=False, log_interval=False)
#        vloss, vtrue, vpred, vauc, vf1 = run_epoch(epoch, model, val_loader, optimizer,
#            do_cuda, only_compute_loss=True, log_interval=False)
#        print("{:10}  {:10}  {:10}  {:10}  {:10}". \
#            format(epoch, "%2.6f"%tloss, "%2.6f"%vloss, 
#                          tauc, vauc, tf1, vf1))
#
#        train_loss.append(tloss)
#        val_loss.append(vloss)
#        if vloss < best_val_loss:
#            best_val_loss = vloss
#            best_val_state = model.state_dict()
#
#        # update learning rate if we're not doing better
#        if epoch % 25 == 0: # 100 and vloss >= prev_val_loss:
#            print("... reducing learning rate!")
#            for param_group in optimizer.param_groups:
#                param_group['lr'] *= .5
#
#    # load in best state by validation loss
#    model.load_state_dict(best_val_state)
#    model.eval()
#
#    # transform train/val/test into logit zs
#    model.cpu()
#    ztrain = model(Variable(train_data.data_tensor))
#    zval   = model(Variable(val_data.data_tensor))
#    ztest  = model(Variable(torch.FloatTensor(Xtest)))
#    resdict = {'train_elbo'  : train_loss,
#               'val_elbo'    : val_loss,
#               'model_type'  : model_type,
#               'model_state' : model.state_dict(),
#               'ztrain'      : ztrain,
#               'zval'        : zval,
#               'ztest'       : ztest,
#               'hdims'       : hdims,
#               'wide_weights': wide_weights,
#               'side_dim'    : side_dim,
#               'data_dim'    : Xtrain.shape[1],
#               'output_dim'  : Ytrain.shape[1]}
#    return resdict
#
#
#def make_model(**kwargs):
#    """ instantiate MLP model from results dict """
#    model_type   = kwargs.get("model_type")
#    data_dim     = kwargs.get("data_dim")
#    side_dim     = kwargs.get("side_dim")
#    output_dim   = kwargs.get("output_dim")
#    hdims        = kwargs.get("hdims")
#    wide_weights = kwargs.get("wide_weights", None)
#    model_state  = kwargs.get("model_state", None)
#
#    if model_type == "MLP":
#        model = MLP(data_dim=data_dim, output_dim=output_dim, hdims=hdims)
#    elif model_type == "MLPRegression":
#        model = MLPRegression(data_dim=data_dim, output_dim=output_dim, hdims=hdims)
#    elif model_type == "DeepWideMLP":
#        model = DeepWideMLP(data_dim=data_dim, side_dim=side_dim,
#            output_dim=output_dim, wide_weights=wide_weights, hdims=hdims)
#    else:
#        raise NotImplementedError("Model not recognized %s"%model_type)
#
#    # load state if we passed it in
#    if model_state is not None:
#        model.load_state_dict(model_state)
#    return model
#
#
#def run_epoch(epoch, model, data_loader, optimizer, do_cuda,
#              only_compute_loss = False,
#              log_interval = 20,
#              num_samples  = 1):
#    binary_loss = nn.BCEWithLogitsLoss()
#    if only_compute_loss:
#        model.eval()
#    else:
#        model.train()
#
#    # iterate over batches
#    total_loss = 0
#    trues, preds = [], []
#    for batch_idx, (data, target) in enumerate(data_loader):
#        data, target = Variable(data), Variable(target)
#        if do_cuda:
#            data, target = data.cuda(), target.cuda()
#
#        # set up optimizer
#        if not only_compute_loss:
#            optimizer.zero_grad()
#
#        # push data through model (make sure the recon batch batches data)
#        loss, logitpreds = model.lossfun(data, target)
#
#        # backprop
#        if not only_compute_loss:
#            loss.backward()
#            optimizer.step()
#
#        # track pred probs
#        tprobs = F.sigmoid(logitpreds).data.cpu().numpy()
#        preds.append(tprobs)
#        trues.append(target.data.cpu().numpy())
#        total_loss += loss.data[0]
#        if log_interval==True and (batch_idx % log_interval == 0):
#            print('{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}'.format(
#                pre = "Val" if only_compute_loss else "Train",
#                ep  = epoch,
#                cb  = batch_idx*data_loader.batch_size,
#                tb  = len(data_loader.dataset),
#                frac = 100. * batch_idx / len(data_loader),
#                loss = loss.data[0] / (batch_idx+1)))
#
#    total_loss /= len(data_loader)
#    trues, preds = np.row_stack(trues), np.row_stack(preds)
#
#    # compute auc
#    if model.is_continuous:
#        target_aucs = np.array([ #np.sqrt(np.mean((t-p)**2))
#                                 r2_score(t, p)
#                                 for t,p in zip(trues.T, preds.T)])
#        target_f1s  = "n/a"
#    else:
#        target_aucs = np.array([ roc_auc_score(t, p)
#                                 for t,p in zip(trues.T, preds.T)])
#        target_f1s  = np.array([ f1_score(t, np.array(p>.5, dtype=np.float))
#                                 for t, p in zip(trues.T, preds.T)
#        if len(target_f1s) > 1:
#          target_f1 = "[%2.3f - %2.3f]"%(np.min(target_f1s), np.max(target_f1s))
#        else:
#          target_f1 = "%2.5f"%target_f1s[0]
#
#    if len(target_aucs) > 1:
#      target_auc = "[%2.3f - %2.3f]"%(np.min(target_aucs), np.max(target_aucs))
#    else:
#      target_auc = "%2.5f"%target_aucs[0]
#    return total_loss, trues, preds, target_auc, target_f1


class DeepWideMLP(nn.Module):
    """ A Wide+Deep MLP --- in the style of 
    https://arxiv.org/pdf/1606.07792.pdf
    Args:
      data_dim   : total input dimension (high dimensional signal plus
                   side information dimension)
      side_dim   : aux data size to be concat w/ at last layer
      output_dim : standard output size
    """
    def __init__(self, data_dim, side_dim, output_dim,
                 wide_weights=None,
                 hdims=[50, 50, 50]):
        super(DeepWideMLP, self).__init__()
        self.data_dim   = data_dim
        self.side_dim   = side_dim
        self.output_dim = output_dim
        self.deep_data_dim = data_dim - side_dim
        self.is_continuous = False
        self.wide_weights = wide_weights

        # compute log Pr(Y | h_last)
        self.pred_loss = nn.BCEWithLogitsLoss(size_average=True)

        # construct forward for deep part
        sizes = [self.deep_data_dim] + hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            #modules.append(nn.BatchNorm1d(dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout())

        modules.append(nn.Linear(sizes[-1], output_dim))
        self.forward_net = nn.Sequential(*modules)
        self.forward_drop = nn.Dropout(p=.25)

        # last layer incorporates shallow part
        #self.combined = nn.Sequential(
        #    nn.Linear(output_dim + self.side_dim),
        #    nn.ReLU(),
        #    nn.Dropout(),
        self.wide_layer = nn.Linear(self.side_dim, output_dim, bias=False)
        if self.wide_weights is not None:
            self.wide_layer.weight.data[:] = torch.FloatTensor(self.wide_weights)
            self.wide_layer.weight.requires_grad = False

        #self.final = nn.Linear(output_dim + self.side_dim, output_dim)
        self.init_params()

    def forward(self, x): #, xaux):
        # deep part: 
        hx = self.forward_net(x[:,:self.deep_data_dim])
        hx = self.forward_drop(hx)
        # map side information
        hs = self.wide_layer(x[:, self.deep_data_dim:])

        # return their sum as the logit(prob)
        return hx + hs
        #hx = torch.cat([hx, x[:,self.deep_data_dim:]], 1)
        #return self.final(hx)

    def init_params(self):
        for p in self.parameters():
            if p.requires_grad:
                p.data.uniform_(-.05, .05)

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.pred_loss(logit, target)
        return torch.mean(pred_loss), logit



class IBMLP(nn.Module):
    def __init__(self, data_dim, output_dim, hdims=[25, 25, 25], beta=.175):
        super(IBMLP, self).__init__()
        self.data_dim   = data_dim
        self.output_dim = output_dim
        self.beta = beta

        # compute log Pr(Y | h_last)
        self.pred_loss = nn.BCEWithLogitsLoss()

        # construct generative network for beats
        sizes = [data_dim] + hdims
        modules = []
        for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            if i < len(sizes[:1])-1:
                modules.append(nn.Dropout())

        # module outputs
        self.forward_net = nn.Sequential(*modules)
        self.mu_net      = nn.Linear(sizes[-1], output_dim)
        self.logvar_net  = nn.Linear(sizes[-1], output_dim)

        self.init_params()

    def forward(self, x):
        """ randomly construct last layer """
        hlast = self.forward_net(x)
        mu, logvar = self.mu_net(hlast), self.logvar_net(hlast)
        z = reparameterize(mu, logvar, training=self.training)
        return z, mu, logvar

    def init_params(self):
        for p in self.parameters():
            p.data.uniform_(-.025, .025)

    def lossfun(self, data, target):
        z, mu, logvar = self.forward(data)
        pred_ll = self.pred_loss(z, target)
        kl_to_prior = self.beta * kldiv_to_std_normal(mu, logvar)
        return torch.mean(pred_ll + kl_to_prior), z


class StitchedMLP(nn.Module):
    def __init__(self, mlp_a, mlp_b):
        super(StitchedMLP, self).__init__()
        self.mlp_a = mlp_a
        self.mlp_b = mlp_b

    def forward(self, x):
        """ output the concatenation of mlp_a and mlp_b """
        return torch.cat([self.mlp_a(x), self.mlp_b(x)], 1)

