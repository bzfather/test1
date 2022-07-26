import cv2
import numpy as np
import matplotlib.pyplot as plt



projmat_0=[[486.480552077510,	-610.411487594914	,-252.673606272194,	3025430.23988222],
[54.4760401281700,	-60.0531933874240,	-614.157551179614	,1769367.76495847],
[0.871215000000000,	-0.157877000000000,	-0.464822000000000,	3972.63932300000]]
projmat_4=np.array(projmat_0).astype ( np.float32 )
projmat_1=[[637.437917819548,	518.202043078707	,-216.342672579927	,1608592.46787206],
[89.1909437621230,	27.2651495507700,	-626.394996004983,	1540701.65661286],
[0.150233000000000,	0.881376000000000,	-0.447891000000000,	2561.67247600000]]
projmat_1=np.array(projmat_1).astype ( np.float32 )


pose2d_1=[[655, 605, 620, 693, 699, 677, 664,670,659],
 [298, 282,234,238, 288,301,218,165, 300]]

pose2d_1=np.array(pose2d_1).astype ( np.float32 )

pose2d_4=[[784,776,787,825,819,789,810,817,795],
 [196,229,256,283,318,320,265,231,312]]

pose2d_4=np.array(pose2d_4).astype ( np.float32 )

pose3d_homo = cv2.triangulatePoints ( projmat_1, projmat_4, pose2d_1, pose2d_4 )
print(pose3d_homo)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(pose3d_homo[0],pose3d_homo[1],pose3d_homo[2])
plt.show()