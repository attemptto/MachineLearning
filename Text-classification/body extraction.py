import cv2 as cv
import numpy as np
import pandas as pd
def robust_pca(X, lamda, mu, max_iter):
    M, N = X.shape
    X = np.array(X)
    normX = np.linalg.norm(X)

    L = np.zeros(shape=(M, N))
    S = np.zeros(shape=(M, N))
    Y = np.zeros(shape=(M, N))

    for iter in range(0,max_iter):
        L = Do(1/mu, X - S + (1/mu)*Y)
        S = So(lamda/mu, X - L +(1/mu)*Y)
        Z = X - L - S

        Y = Y + mu*Z

    return L,S

def So(tau, X):

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j]<0:
                X[i][j] = 0
    r = np.multiply(np.sign(X), X)
    return r

def Do(tau, X):
    U, S, V = np.linalg.svd(X,full_matrices=False)
    S = np.diag(S)
    r_front = np.dot(U,S)
    r = np.dot(r_front, V)
    return r

bga = cv.VideoCapture('video/RobustPCA_video_demo.avi')

fps = int(bga.get(cv.CAP_PROP_FPS))

size=(int(bga.get(cv.CAP_PROP_FRAME_HEIGHT)),int(bga.get(cv.CAP_PROP_FRAME_WIDTH)))

X = []
while True:
    success, frame = bga.read()
    if not success:
        break
    frame = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    frame = frame.reshape(-1)
    X.append(frame)
X = pd.DataFrame(X).fillna('0')
X = np.array(X)
lamda = 1 / np.sqrt(X.shape[1])
L, S = robust_pca(X, lamda=lamda/3, mu=10*lamda/3, max_iter=20)

output = cv.VideoWriter('video/RobustPCA_videoOutput_demo.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20,(192,36))
L = L.reshape(180, 36, 64)
S = S.reshape(180, 36, 64)
X = X.reshape(180, 36, 64)
for i in range(180):
    frame1 = L[i]
    frame2 = S[i]
    frame3 = X[i]
    frame = np.concatenate((frame1,frame2,frame3),axis=1)
    frame = cv.cvtColor(frame.astype('uint8'),cv.COLOR_GRAY2RGB)
    #cv.imshow('frame',frame)
    #cv.waitKey(0)
    output.write(frame)

output.release()
