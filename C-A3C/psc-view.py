import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

parser = argparse.ArgumentParser(description="view psc_info")
parser.add_argument('filename')
parser.add_argument('-t', '--title', default=None,
                    help="title of figure")
parser.add_argument('--save', action='store_true',
                    help="save graph to file 'filename.png' and don't display it")

args = parser.parse_args()
if args.title is None:
  args.title = args.filename

f = open(args.filename, "rb")
psc_info = pickle.load(f)
psc_n = psc_info["psc_n"]
psc_vcount = psc_info["psc_vcount"]

print("psc_n = ", psc_n)

vcount = np.array(psc_vcount)
vp = vcount / psc_n
#v = np.arange(128)
#vexp = np.sum(v * vp, axis=1)
print("vp.max = ", vp.max())

vp_argsort = np.argsort(vp, axis=1)

if True: # picture of saturated pixels
  imagei = np.max(vp, axis=1)
  print("imagei.max = ", imagei.max())
  imagei[imagei < 1.0] = 0.0
  image = np.reshape(imagei, (42, 42))
  image = cv2.resize(image, (42*8, 42*8))
  savefilename = args.title + "-" + "saturated" + ".png"
  plt.imsave(savefilename, image, cmap="gray")
  print("Graph saved to ", savefilename)

if True: # picture of average value of pixels
  imagei = np.sum(np.arange(128) * vp, axis=1)
  image = np.reshape(imagei, (42, 42))
  image = cv2.resize(image, (42*8, 42*8))
  savefilename = args.title + "-" + "average" + ".png"
  plt.imsave(savefilename, image, cmap="gray")
  print("Graph saved to ", savefilename)

print(vp.shape)
for i in range(8):
  imagei = vp_argsort[:, -(i+1)] / 128.0
  image = np.reshape(imagei, (42, 42))
  image = cv2.resize(image, (42*8, 42*8))
  savefilename = args.title + "-" + str(i) + ".png"
  plt.imsave(savefilename, image, cmap="gray")
  print("Graph saved to ", savefilename)
