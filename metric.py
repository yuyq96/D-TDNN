import numpy as np


def compute_fnr_fpr(scores, labels):
    """ computes false negative rate (FNR) and false positive rate (FPR)
    given trial scores and their labels.
    """

    indices = np.argsort(scores)
    labels = labels[indices]

    target = (labels == 1).astype('f8')
    nontar = (labels == 0).astype('f8')

    fnr = np.cumsum(target) / np.sum(target)
    fpr = 1 - np.cumsum(nontar) / np.sum(nontar)
    return fnr, fpr


def compute_eer(fnr, fpr, requires_threshold=False, scores=None):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
        *kaldi style*
    """

    diff_miss_fa = fnr - fpr
    x = np.flatnonzero(diff_miss_fa >= 0)[0]
    eer = fnr[x - 1]
    if requires_threshold:
        assert scores is not None
        scores = np.sort(scores)
        th = scores[x]
        return eer, th
    return eer


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return c_det / c_def
