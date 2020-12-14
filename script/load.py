from scipy.stats import sem
from statistics import mean

mcc=[0.166,0.182,0.181,0.164,0.151]
se=sem(mcc)
print(mean(mcc),se)
