import numpy as np
import argparse
import time
import re
import sys

parser = argparse.ArgumentParser(description="extract visited rooms in A3C")
parser.add_argument('filename')
args = parser.parse_args()

f = open(args.filename, "r")
prog = re.compile(".*ROOM\((\d+)\)")
rooms = np.zeros(24)

line = f.readline()
while line != "":
  match = prog.match(line)
  if match:
    room_no =int(match.group(1))
    rooms[room_no] += 1
  line = f.readline()

visited_rooms = np.arange(24)[rooms != 0]
print(visited_rooms)
