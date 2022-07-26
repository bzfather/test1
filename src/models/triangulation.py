import numpy as np

def triangulateLinearEigen(cameras, x):
    '''
    Triangulate using the linear eigen method
    cameras: list of cameras, shape [N,]
    x: image projection coordinates, shape [N,2] or [N,2,1]
    '''
    longX = True
    if len(x.shape) == 2:
        longX = False
        x = x[:,:,None] # Ensure shape [N,2,1]
    x = np.concatenate([x, np.ones(shape=[x.shape[0],1,1])], axis=1) # [N,3,1]

    P = np.asarray([cam for cam in cameras]) # [N,3,4]
    A1 = x[:,0,:]*P[:,2,:] - P[:,0,:] # [N,1]*[N,4] = [N,4]
    A2 = x[:,1,:]*P[:,2,:] - P[:,1,:] # [N,1]*[N,4] = [N,4]
    A = np.concatenate([A1, A2], axis=0) # [2N,4]

    X,_,_,_ = np.linalg.lstsq(A[:,:3], -A[:,3,None]) # [2N,3]\[2N,1] = [3,1]

    return X if longX else X[:,0]

def baseFuncMVMP(cameras, x):
    if len(x.shape) == 2:
        x = x[:,:,None] # Ensure shape [N,2,1]
    x = np.concatenate([x, np.ones(shape=[x.shape[0],1,1])], axis=1) # [N,3,1]

    b = np.asarray([cameras[n].get3DLine(x[n,:2,0]/x[n,2,0], format='TdL') for n in range(len(cameras))]) # [N,2,3]
    o = b[:,0,:] # Translations (optical center)
    b = b[:,1,:]/np.linalg.norm(b[:,1,:], ord=2, axis=1)[:,None] # unit direction vectors

    B = np.eye(3, dtype=float)[None,:,:]-b[:,:,None]*b[:,None,:] # [1,3,3]-[N,3,1]*[N,1,3]=[N,3,3]

    X = np.linalg.solve(np.sum(B, axis=0), np.sum(B@o[:,:,None], axis=0)) # [3,3]\[3,1]=[3,1]
    
    return X, o, B

def triangulateMVMP(cameras, x):
    '''
    Triangulate using the Multiple View MidPoint Method
    cameras: list of cameras, shape [N,]
    x: image projection coordinates, shape [N,2] or [N,2,1]
    '''
    longX = True
    if len(x.shape) == 2:
        longX = False

    X,_,_ = baseFuncMVMP(cameras, x) # [3,1]
    
    return X if longX else X[:,0]

def triangulateIRMP(cameras, x, eps=1e-15, maxIters=500):
    '''
    Triangulate using the iteratively Reweighted MidPoint Method
    cameras: list of cameras, shape [N,]
    x: image projection coordinates, shape [N,2] or [N,2,1]
    eps: minimum step size per iteration
    maxIters: maximum number of iterations
    '''
    longX = True
    if len(x.shape) == 2:
        longX = False
        x = x[:,:,None] # Ensure shape [N,2,1]
    x = np.concatenate([x, np.ones(shape=[x.shape[0],1,1])], axis=1) # [N,3,1]

    X, o, B = baseFuncMVMP(cameras, x)
    
    i = 0
    lastStep = None
    while (not lastStep or lastStep >= eps) and i < maxIters:
        i += 1
        w = 1/np.linalg.norm(X[None,:,:]-o[:,:,None], ord=2, axis=1)[:,0] # [N]
        A = (w[:,None,None]**2)*B # [N,1,1]*[N,3,3]=[N,3,3]
        e = (np.linalg.norm(B@(X[None,:,:]-o[:,:,None]), ord=2, axis=1)[:,0]**2)/(np.linalg.norm(X[None,:,:]-o[:,:,None], ord=2, axis=1)**2)[:,0] # [N]
        r = (w**2)[:,None,None]*(e**2)[:,None,None]*(X[None,:,:]-o[:,:,None]) # [N,3,1]
        
        lastStep = -X + np.linalg.solve(np.sum(A, axis=0), np.sum(A@o[:,:,None]+r, axis=0)) # [3,1]
        X += lastStep
        lastStep = np.linalg.norm(lastStep[:,0], ord=2)
    
    return X if longX else X[:,0]


def baseFuncLM(cameras, x, XInit, eps1=1e-7, eps2=1e-15, maxIters=500, tau=1e-4, v=2):
    '''
    Base function that applies Levenberg-Marquardt for the triangulateLM and triangulateMPLM functions
    '''
    P = np.asarray([cam.P for cam in cameras]) # Projection matrices, [N,3,4]

    X = np.concatenate([XInit, np.ones([1,1])], axis=0) # [4,1]

    f = (x[:,:2,:]/x[:,2,None,:]-(P[:,:2,:]@X[None,:,:])/(P[:,2,None,:]@X[None,:,:])).reshape((2*x.shape[0],1)) # Current loss vector, [2N,1]. [4,1]->[2N,1]
    F = (.5*f.T@f)[0,0] # [1,]
    L = F # [1,]
    J = ((P[:,:2,:]*(P[:,2,None,:]@X[None,:,:]) - P[:,2,None,:]*(P[:,:2,:]@X[None,:,:]))/((P[:,2,None,:]@X[None,:,:])**2)).reshape((2*x.shape[0],4)) # Current Jacobian of f, [2N,4]

    A = J.T@J # [4,4]
    g = J.T@f # [4,1]

    mu = tau*np.max(np.diag(A)) # Damping parameter

    found = False
    i = 0 # Current iteration
    while not found and (np.linalg.norm(g, ord=np.inf) >= eps1) and i < maxIters:
        i += 1
        nextStep = np.linalg.solve(A+mu*np.eye(4), g) # [4,1]

        fNext = (x[:,:2,:]/x[:,2,None,:]-(P[:,:2,:]@(X+nextStep)[None,:,:])/(P[:,2,None,:]@(X+nextStep)[None,:,:])).reshape((2*x.shape[0],1))
        FNext = (.5*fNext.T@fNext)[0,0] # [1,]
        LNext = FNext + (nextStep.T@J.T@f)[0,0] + .5*(nextStep.T@J.T@J@nextStep)[0,0] # [1,]
        JNext = ((P[:,:2,:]*(P[:,2,None,:]@(X+nextStep)[None,:,:]) - P[:,2,None,:]*(P[:,:2,:]@(X+nextStep)[None,:,:]))/((P[:,2,None,:]@(X+nextStep)[None,:,:])**2)).reshape((2*x.shape[0],4))
        q = (F-FNext)/(L-LNext) if L != LNext else 1 # Gain ratio

        if np.linalg.norm(nextStep) <= eps2*(np.linalg.norm(X)+eps2):
            found = True
        elif q > 0:
            X += nextStep
            
            f = fNext
            F = FNext
            L = LNext
            J = JNext

            A = J.T@J
            g = J.T@f

            found = np.linalg.norm(g, ord=np.inf) <= eps1

            mu *= max(1/3, 1-(2*q-1)**3)
        else:
            mu *= v
            v *= 2
    
    return X[:3,:]/X[3,:] if X[3,:] != 0 else None

def triangulateLM(cameras, x, eps1=1e-7, eps2=1e-15, maxIters=500, tau=1e-4, v=2):
    '''
    Triangulate by minimizing sum(d(x, x_reproj)) with the Levenberg-Marquardt Method
    cameras: list of cameras, shape [N,]
    x: image projection coordinates, shape [N,2] or [N,2,1]
    eps1: threshold for g, below which optimization ends (local minimum found)
    eps2: threshold for step size, below which optimization ends (last step too small)
    maxIters: maximum number of iterations
    tau: Damping parameter initial scaling factor
    v: Damping parameter rescaling factor
    '''
    longX = True
    if len(x.shape) == 2:
        longX = False
        x = x[:,:,None] # Ensure shape [N,2,1]
    x = np.concatenate([x, np.ones(shape=[x.shape[0],1,1])], axis=1) # [N,3,1]

    X = triangulateLinearEigen(cameras, x) # Initialize value with linear triangulation, [4,1]
    X = baseFuncLM(cameras, x, X, eps1, eps2, maxIters, tau, v)

    return X if longX else X[:,0]

def triangulateMPLM(cameras, x, eps1=1e-7, eps2=1e-15, maxIters=500, tau=1e-4, v=2):
    '''
    Triangulate by minimizing sum(d(x, x_reproj)) with the Levenberg-Marquardt Method, using different initialization than triangulateMPLM
    cameras: list of cameras, shape [N,]
    x: image projection coordinates, shape [N,2] or [N,2,1]
    eps1: threshold for g, below which optimization ends (local minimum found)
    eps2: threshold for step size, below which optimization ends (last step too small)
    maxIters: maximum number of iterations
    tau: Damping parameter initial scaling factor
    v: Damping parameter rescaling factor
    '''
    longX = True
    if len(x.shape) == 2:
        longX = False
        x = x[:,:,None] # Ensure shape [N,2,1]
    x = np.concatenate([x, np.ones(shape=[x.shape[0],1,1])], axis=1) # [N,3,1]

    X = triangulateMVMP(cameras, x) # Initialize value with linear triangulation, [4,1]
    X = baseFuncLM(cameras, x, X, eps1, eps2, maxIters, tau, v)

    return X if longX else X[:,0]



