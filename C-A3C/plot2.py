import numpy as np
import argparse
import time
import re
import sys
from operator import itemgetter

parser = argparse.ArgumentParser(description="plot data in A3C log file and update it periodically")
parser.add_argument('filename')
parser.add_argument('--sx', type=int, default=None,
                    help="x size (inch) of graph")
parser.add_argument('--sy', type=int, default=None,
                    help="y size (inch) of graph")
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
                    help="End Mark in reward line")
parser.add_argument('--save', action='store_true',
                    help="save graph to file 'filename.png'")
parser.add_argument('--no-display', action='store_true',
                    help="no display")
parser.add_argument('-i', '--info', choices=["r", "lives", "s", "tes", "RO", "v", "pr", "k", "R"], default="r",
                    help="information in y-axis : r (reward), lives (OHL), s (OHL), tes (OHL), RO (OHL), v, pr (psc-reward), k (kill), R (room)")
parser.add_argument('-nc', '--num-class', type=int, default=300,
                    help="number of class in x-axis for -i RO, -i k , -i R")
parser.add_argument('-uc', '--unit-of-class', type=int, default=100000,
                    help="unit of class in x-axis for -i RO, -i k , -i R")
parser.add_argument('-er', '--except-rooms', default="0, 1",
                    help="rooms except EXCEPT-ROOMS")

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

  if args.info != "RO":
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

  ax.legend(loc=0, fontsize="small")
  ax.set_xlabel(args.xlabel)
  ax.set_ylabel(args.ylabel)

  ax.grid(linewidth=1, linestyle="-", alpha=0.1)


def draw_room_graph(ax, data):

# sort data along args.x_column and make it np.array again
  all_data = sorted(data, key=itemgetter(args.x_column))
  rooms = list({int(e[2]) for e in all_data}.difference(except_rooms))
  rooms.sort()
  print("rooms=", rooms)

  np_all_data = np.array(all_data)
  all_x = np_all_data[:, args.x_column]
  x_max = np.max(all_x)

  ax.set_xlim(xmax = x_max / args.scale)
  ax.set_xlim(xmin = 0)

  x_max = (x_max + args.unit_of_class - 1) // args.unit_of_class * args.unit_of_class
  d = x_max // args.num_class

  # ax.set_ylim(ymin = 0) # this makes ymax=1 ... so bad!

  for room in rooms:
    # print("room=", room)
    data = list(filter(lambda e: e[2] == room, all_data))

    data = np.array(data)

    x = data[:, args.x_column]

    # http://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python 
    class_index = x // d
    unique, count = np.unique(class_index, return_counts = True)
    x = unique * d

    x = x / args.scale

    x = np.hstack((0, x))
    count = np.hstack((0, count))
    ax.plot(x, count, label=str(room))

  ax.legend(loc=0, fontsize="small")
  ax.set_xlabel(args.xlabel)
  ax.set_ylabel(args.ylabel)

  ax.grid(linewidth=1, linestyle="-", alpha=0.1)



args = parser.parse_args()

ohl=False
room=False
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
elif args.info == "RO":
  pattern = '.*SCORE=\s*(\d+),s=\s*(\d+).*RM(\d+)'
  args.ylabel = "rooms (OHL)"
  room=True
elif args.info == "v":
  pattern = 't=\s*(\d+),s=\s*(\d+).*v=(\d+\.\d+)'
  args.ylabel = "v"
elif args.info == "pr":
  pattern = 't=\s*(\d+),s=\s*(\d+).*pr=(\d+\.\d+)'
  args.ylabel = "pr (psc rewward)"
elif args.info == "k":
  pattern = 't=\s*(\d+),s=\s*(\d+).*l=\d>\dRM(\d\d)'
  args.ylabel = "kill"
  room=True
elif args.info == "R":
  pattern = 't=\s*(\d+),s=\s*(\d+).*r=\s*\d+RM(\d\d)'
  args.ylabel = "rooms"
  room=True
else:
  pass

except_rooms={int(r) for r in args.except_rooms.split(",")}
#print("except_rooms=", except_rooms)

if args.title is None:
  args.title = args.filename + "." + args.info

# trick for headless environment 
if args.no_display:
  import matplotlib as mpl
  mpl.use('Agg')
import matplotlib.pyplot as plt

f = open(args.filename, "r")
prog = re.compile(pattern)

data = []
if args.sx is not None:
  fig = plt.figure(args.title, figsize=(args.sx, args.sy))
else:
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
      elif room:
        draw_room_graph(ax, data)
      else:
        draw_graph(ax, data)
  if args.save:
    savefilename = args.title + ".png"
    plt.savefig(savefilename)
    print("Graph saved to ", savefilename)
  if args.interval == 0:
    if not args.no_display:
      plt.show()
    sys.exit(0)
  else:
    if args.no_display:
      time.sleep(args.interval)
    else:
      plt.pause(args.interval)

