import scipy.stats

def gaussian(t, h=1):
    return scipy.stats.norm(0, 1).pdf(t/h)/h

KERNEL_OPTIONS = {'gaussian': gaussian}