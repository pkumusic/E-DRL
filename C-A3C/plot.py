import numpy as np
import argparse
import time
import re
import sys
from operator import itemgetter

parser = argparse.ArgumentParser(description="plot data in A3C log file and update it periodically")
parser.add_argument('filename')
parser.add_argument('-x', '--x-column', type=int, default=1,
                    help="column index of x-axis (0 origin)")
parser.add_argument('-y', '--y-column', type=int, default=2,
                    help="column index of y-axis (0 origin)")
parser.add_argument('-a', '--average-number-of-samples', dest="ans", type=int, default=100,
                    help="average number of samples")
parser.add_argument('-s', '--scale', type=float, default=1e6,
                    help="scale factor: data in x-column is divided by SCALE")
parser.add_argument('-xl', '--xlabel', default="M steps",
                    help="label of x-axis")
parser.add_argument('-yl', '--ylabel', default="Score",
                    help="label of y-axis")
parser.add_argument('-t', '--title', default=None,
                    help="title of figure")
parser.add_argument('-n', '--interval', type=int, default=10,
                    help="interval of refresh (0 means no refresh)")
parser.add_argument('-e', '--endmark', default="END",
                    help="End Mark of in reward line")
parser.add_argument('--save', action='store_true',
                    help="save graph to file 'filename.png' and don't display it")
parser.add_argument('-i', '--info', choices=["r", "lives", "s", "tes", "v", "pr"], default="r",
                    help="information in y-axis : r (reward), lives (OHL), s (OHL) tes (OHL), v, pr (psc-reward)")

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

def draw_graph(ax, data):
  ans = args.ans
  if len(data) < 5:
    return
  elif len(data) < args.ans:
    ans = len(data) - 1

# sort data along args.x_column and make it np.array again
  data = sorted(data, key=itemgetter(args.x_column))
  data = np.array(data)

  x = data[:, args.x_column]
  y = data[:, args.y_column]
  x_max = np.max(x)
  x_min = np.min(x)
  y_max = np.max(y)
  y_min = np.min(y)
  # print("ymax=", y_max, "ymin=", y_min)
  y_width = y_max - y_min
  if y_width == 0:
    if y_max == 0:
      y_width = 1.0
    else:
      y_min = 0
      y_width = y_max
  ax.set_xlim(xmax = x_max / args.scale)
  ax.set_xlim(xmin = 0)
  ax.set_ylim(ymax = y_max + y_width * 0.05)
  ax.set_ylim(ymin = y_min - y_width * 0.05)

  x = x / args.scale
  ax.plot(x, y, ',')

  weight = np.ones(ans, dtype=np.float)/ans
  y_average = np.convolve(y, weight, 'valid')
  rim = ans - 1
  rim_l = rim // 2
  rim_r = rim - rim_l
  ax.plot(x[rim_l:-rim_r], y_average)

  ax.set_xlabel(args.xlabel)
  ax.set_ylabel(args.ylabel)

  ax.grid(linewidth=1, linestyle="-", alpha=0.1)

def draw_ohl_graph(ax, data):

# sort data along args.x_column and make it np.array again
  all_data = sorted(data, key=itemgetter(args.x_column))
  scores = list({e[0] for e in all_data})
  scores.sort()
  print("scores=", scores)

  np_all_data = np.array(all_data)
  all_x = np_all_data[:, args.x_column]
  all_y = np_all_data[:, args.y_column]
  x_max = np.max(all_x)
  x_min = np.min(all_x)
  y_max = np.max(all_y)
  y_min = np.min(all_y)
  # print("ymax=", y_max, "ymin=", y_min)
  y_width = y_max - y_min
  if y_width == 0:
    if y_max == 0:
      y_width = 1.0
    else:
      y_min = 0
      y_width = y_max
  ax.set_xlim(xmax = x_max / args.scale)
  ax.set_xlim(xmin = 0)
  ax.set_ylim(ymax = y_max + y_width * 0.05)
  ax.set_ylim(ymin = y_min - y_width * 0.05)

  for score in scores:
    # print("score=", score)
    data = list(filter(lambda e: e[0] == score, all_data))

    data = np.array(data)

    x = data[:, args.x_column]
    y = data[:, args.y_column]
    x = x / args.scale

    ans = args.ans
    if len(data) < 5:
      ax.plot(x, y, '.', label=str(score))
      continue
    elif len(data) * 0.1 < args.ans:
      ans = int(len(data) * 0.1)
      if ans < 4:
        ans = 4
    # print("ans=", ans)

    weight = np.ones(ans, dtype=np.float)/ans
    y_average = np.convolve(y, weight, 'valid')
    rim = ans - 1
    rim_l = rim // 2
    rim_r = rim - rim_l
    ax.plot(x[rim_l:-rim_r], y_average, label=str(score))

  ax.legend(loc=2)
  ax.set_xlabel(args.xlabel)
  ax.set_ylabel(args.ylabel)

  ax.grid(linewidth=1, linestyle="-", alpha=0.1)


args = parser.parse_args()

ohl=False
if args.info == "r":
  pattern = 't=\s*(\d+),s=\s*(\d+).*r=\s*(\d+)@' + args.endmark
  args.ylabel = "score"
elif args.info == "lives":
  pattern = '.*SCORE=\s*(\d+),s=\s*(\d+).*lives=\s*(\d+)'
  args.ylabel = "lives (OHL)"
  ohl=True
elif args.info == "s":
  pattern = '.*SCORE=\s*(\d+),s=\s*(\d+).*steps=\s*(\d+)'
  args.ylabel = "steps (OHL)"
  ohl=True
elif args.info == "tes":
  pattern = '.*SCORE=\s*(\d+),s=\s*(\d+).*tes=\s*(\d+)'
  args.ylabel = "tes (OHL)"
  ohl=True
elif args.info == "v":
  pattern = 't=\s*(\d+),s=\s*(\d+).*v=(\d+\.\d+)'
  args.ylabel = "v"
elif args.info == "pr":
  pattern = 't=\s*(\d+),s=\s*(\d+).*pr=(\d+\.\d+)'
  args.ylabel = "pr (psc rewward)"
else:
  pass

if args.title is None:
  args.title = args.filename + "." + args.info

# trick for headless environment 
if args.save:
  import matplotlib as mpl
  mpl.use('Agg')
import matplotlib.pyplot as plt

f = open(args.filename, "r")
prog = re.compile(pattern)

data = []
fig = plt.figure(args.title)
ax = fig.add_subplot(111)
while True:
  new_data = read_data(f)
  print(len(new_data), "data added.")
  if (len(new_data) > 0):
      data.extend(new_data)
      ax.clear()
      ax.set_title(args.title)
      if ohl:
        draw_ohl_graph(ax, data)
      else:
        draw_graph(ax, data)
  if args.save:
    savefilename = args.title + ".png"
    plt.savefig(savefilename)
    print("Graph saved to ", savefilename)
    sys.exit(0)
  elif args.interval == 0:
    plt.show()
    sys.exit(0)
  else:
    plt.pause(args.interval)

