

def get_f1(p, r):
    if p == 0 and r == 0:
        return 'P=R=0'
    if type(p) != float or type(r) != float:
        return 'P,R not int'
    f1 = 2 * p * r / (p + r)
    return f1

"""
    tp: H@1=1 && has pred
    fp: H@1=0 && has pred
"""
def compute_p(tp, fp):
    if tp + fp == 0:
        return 'TP+FP=0'
    return tp / (tp + fp)

"""
    tp: H@1=1 && has pred
    fn: H@1=0 && no pred => num_negative
"""
def compute_r(tp, fn):
    if tp + fn == 0:
        return 'TP+FN=0'
    return tp / (tp + fn)

def compute_acc(tp, tn, fp, fn):
    return (tp+tn) / (tp+tn+fp+fn)