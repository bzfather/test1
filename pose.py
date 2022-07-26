import os
import pickle
import numpy as np

def plot3dpose(allpose):
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    '''
    ax.view_init(elev=0)
    lim=2000
    scale=1
    bais=
    '''
    ax.set_xlim3d(-800, 1200)
    ax.set_ylim3d(-800, 1200)
    ax.set_zlim3d(500, 2500)
    
    for i,pose3d in enumerate(allpose):

        
        
        xs=pose3d[0]
        ys=pose3d[1]
        zs=pose3d[2]



        link = [[0,1], [1,2], [3,4], [4, 5], [6, 7], [6, 8], [2, 6], [6, 3]]
        for l in link:
           index1,index2 = l[0],l[1]
           ax.plot([xs[index1],xs[index2]], [ys[index1],ys[index2]], [zs[index1],zs[index2]])

        ax.scatter(xs,ys,zs)
    matrix=[[0.985754 ,-0.098885, 0.136055, -100.313354],
            [0.075626, -0.461950, -0.883676 ,1325.817776],
            [0.150233, 0.881376, -0.447891, 2561.672476]]
    A=np.array(matrix)
    R=A[0:3,0:3]
    T=A[:,3]
    C=-R.T@T
    C=np.array([C[0],C[1],C[2],1.0])
    ax.scatter(C[0],C[1],C[2])
    proj=np.array([[637.437917819548,	518.202043078707	,-216.342672579927	,1608592.46787206],
[89.1909437621230,	27.2651495507700,	-626.394996004983,	1540701.65661286],
[0.150233000000000,	0.881376000000000,	-0.447891000000000,	2561.67247600000]])
    projinv=np.linalg.pinv(proj)
    print(projinv)
    new=np.matmul(projinv,[548, 340,1])

    #new=new[0:-1]/new[-1]
    new=new+0.00001*C
    new=new[0:-1]/new[-1]
    print(new,C)
    ax.plot([C[0],new[0]],[C[1],new[1]],[C[2],new[2]]) 
    
    plt.show()
    
with open ( 'allpose3d.pickle' , 'rb' ) as f:
    x = pickle.load(f)

plot3dpose(x)