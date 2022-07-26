import cv2
import numpy as np
import matplotlib.pyplot as plt

def renderBones():
   link = [[0,1], [1,2], [3,4], [4, 5], [6, 7], [6, 8], [2, 6], [6, 3]]
   for l in link:
       index1,index2 = l[0],l[1]
       ax.plot([xs[index1],xs[index2]], [ys[index1],ys[index2]], [zs[index1],zs[index2]], linewidth=1, label=r'$z=y=x$')

pose3d=[[ 112.55089297 , 176.5650438,   190.23322607,  -77.3104592,  -103.93626565,
   -68.92123422,   82.76147856,   47.08444295,  143.49664814],
 [ 204.38049677  ,198.43532867  ,  2.57974546, -115.89117641, -116.79105651,
    72.02047288 , -74.09010695 , -88.65468646 , -60.04268609],
 [1862.20136909, 1641.63543606, 1421.83254326, 1348.20396755, 1133.8611157,
  1137.06147551 ,1428.99508943 ,1641.72649627 ,1034.77239457]]


xs=pose3d[0]
ys=pose3d[1]
zs=pose3d[2]
fig = plt.figure()
ax = fig.add_subplot( projection='3d')

renderBones()
ax.scatter(pose3d[0],pose3d[1],pose3d[2])
ax.set_xlim3d(-400, 400)
ax.set_ylim3d(-400, 400)
ax.set_zlim3d(1100, 1900)

plt.show()