"""

"""

from __future__ import division
import numpy as np


def get_f(t1, t2, t3):
    
    def f((p_1, p_2)):
        y = np.array([np.exp(-p_1*t1)+np.exp(-p_2*t1),
                      np.exp(-p_1*t2)+np.exp(-p_2*t2),
                      np.exp(-p_1*t3)+np.exp(-p_2*t3)])
        return y
    
    return f