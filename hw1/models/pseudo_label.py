"""
Functions for pseudo label
"""

def default_pseudo_label_func(unlabel, t):
    T1 = 200
    T2 = 800
    alpha_f = 3.0
    if not unlabel:
        return 1
    else:
        if t < T1:
            return 0
        elif t < T2:
            return float(t-T1)/(T2-T1)*alpha_f
        else:
            return alpha_f