import numpy as np
import argparse
import time
import re
from operator import itemgetter

parser = argparse.ArgumentParser(description="show average of data in A3C log file and update it periodically")
parser.add_argument('filename')
parser.add_argument('-a', '--average-number-of-samples', dest="ans", type=int, default=100,
                    help="average number of samples")
parser.add_argument('-n', '--interval', type=int, default=30,
                    help="interval of refresh (0 means no refresh)")
parser.add_argument('-s', '--samples', type=int, default=20,
                    help="number of samples")
parser.add_argument('-e', '--endmark', default="END",
                    help="End Mark of in reward line")

def read_data(f):
  data = []
  line = f.readline()
  while line != "":
    match = prog.match(line)
    if match:
      t = float(match.group(1))
      s = float(match.group(2))
      r = float(match.group(3))
      data.append([t, s, r])
    line = f.readline()
  return data

def show_average(data):
  ans = args.ans
  samples = args.samples
  if len(data) < 5:
    return
  elif len(data) < args.samples:
    ans = 1
    samples = len(data)
  elif len(data) < args.samples * args.ans:
    ans = len(data) // args.samples
    print("len(data)=", len(data), "ans=", ans, ", samples=", samples)
  elif len(data) < args.ans * args.ans:
    ans = len(data) // args.ans
    print("len(data)=", len(data), "ans=", ans, ", samples=", samples)


# sort data along args.x_column and make it np.array again
  data = sorted(data, key=itemgetter(1))
  data = np.array(data)

  t = data[:, 0]
  s = data[:, 1]
  r = data[:, 2]
  t = t / 3600.0
  s = s / 1e6

  weight = np.ones(ans, dtype=np.float)/ans
  ra = np.convolve(r, weight, 'valid')
  rim = ans - 1

  t = t[rim:]
  s = s[rim:]
  l = t.shape[0]
  i = np.linspace(0, l-1, samples, dtype=np.int)
  print("=====================================")
  for t1, s1, ra1 in zip(t[i], s[i], ra[i]):
    print("t={:.2f} hours, s={:.2f} Msteps, ra={:.2f}".format(t1, s1, ra1))

args = parser.parse_args()

f = open(args.filename, "r")
prog = re.compile('t=\s*(\d+),s=\s*(\d+).*r=\s*(\d+)@' + args.endmark)

data = []
while True:
  new_data = read_data(f)
  print(len(new_data), "data added.")
  if (len(new_data) > 0):
      data.extend(new_data)
      show_average(data)
  time.sleep(args.interval)

