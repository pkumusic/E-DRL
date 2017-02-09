import numpy as np

def g(l, u, s, r):
  return (np.random.randint(u - l, size=s) + l) / r

def p(f, a):
  s = len(a)
  fs = f * s
  return fs.format(*a)[:-2]

def d():
  l1 = "tes_list     : '{}'".format(p("{:5.0f}, ", g(20, 40, 8, 1)))
  l2 = "psc_beta_list: '{}'".format(p("{:5.3f}, ", g(18, 22, 8, 1000)))
  l3 = "psc_pow_list : '{}'".format(p("{:5.1f}, ", g(18, 22, 8, 10)))
  return l1 + "\n" + l2 + "\n" + l3 +"\n"

for i in range(10, 90, 10):
  fname = "gcp{:2d}.yaml".format(i)
  with open(fname, "w") as f:
    f.write(d())



