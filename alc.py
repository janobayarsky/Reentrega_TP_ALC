import numpy as np

__all__ = [
    "multiplicarMatrices",
    #"calculaLU",
    #"calculaLDV",
    "norma",
    "calcularAx",
    "prodVect",
    #"metpot2k",
    #"diagRH",
    "svd_reducida",
    "QR_con_GS",
    "res_tri",
    "inversa",
    "calculaCholesky"
]

def multiplicarMatrices(A,B):
    filasA,columnasA = A.shape
    filasB,columnasB = B.shape
    if columnasA != filasB:
        return None
    M = np.zeros((filasA,columnasB))
    for i in range(filasA):
        for j in range(columnasB):
            suma = 0
            for k in range(filasB):
                suma += A[i][k] * B[k][j]
            M[i][j] = suma
    return M

def calculaLU(A):
    if A is None:
        return None,None,0

    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    L = np.eye(n)

    if m!=n or A is None:
        return None,None,0
    cant_op = 0
    for i in range(n):
        if Ac[i,i] == 0:
          return None, None, 0
        for j in range(i+1,n):
            cant_op += 1+ 2*(n-i-1)
            pivot = Ac[j,i] / Ac[i,i]
            L[j,i] = pivot
            Ac[j] = Ac[j] - pivot*Ac[i]
    U = Ac

    return L, U, cant_op

def calculaLDV(A):
  n = A.shape[0]
  D = np.eye(n)
  D_inv = np.eye(n)
  L,U,cant = calculaLU(A)
  if L is None:
    return None, None, None
  for i in range(n):
    D[i,i] = U[i,i]
    D_inv[i,i] = 1/U[i,i]

  # Arriba calculo la inversa directamente, Me quede V = D^-1@V
  V = multiplicarMatrices(D_inv, U)
  return L,D,V

def norma(x,p):
    max = 0
    res = 0
    if x.ndim ==1:
      if p == 'inf':
        for i in range(len(x)):
          if max < abs(x[i]):
            max = abs(x[i])
        return max

      for i in range(len(x)):
          res+= abs(x[i])**p
      return res**(1/p)

    n = x.shape[1]
    resultado = np.zeros(n)

    for j in range(n):
      resultado[j] = norma(x[:,j],p)
    return resultado

def calcularAx(A, v):
    w = np.zeros((A.shape[0]))
    for i in range(A.shape[0]):
        row = v * A[i]
        sum_row = 0
        for elem in row:
            sum_row += elem
        w[i] = sum_row
    return w

def f(A, v):
    w = calcularAx(A, v)
    norm = norma(w,2)
    if norm > 0:
        w = w/norm
    return w

def f_k(A, v, k):
    for i in range(k):
        v = f(A, v)
    return v


def prodVect(v,w):
  n = len(v)
  suma = 0
  for i in range(n):
    suma += v[i]*w[i]
  return suma

def metpot2k(A,tol=1e-15,K=1000):
  fil,col = A.shape
  v = np.random.rand(fil)
  v_p = f_k(A,v,2)
  e = v_p.T @ v
  k = 0
  while abs(e - 1) > tol and k < K:
    v = v_p
    v_p = f_k(A,v,2)
    e = prodVect(v_p,v)
    k += 1
  lamb = prodVect(v_p.T, calcularAx(A,v_p))
  eps = e - 1
  return v, lamb, k, eps

def diagRH(A,tol=1e-15,K=1000):
  fil,col = A.shape
  v1, lamb1, _, _ = metpot2k(A,tol,K)
  I = np.eye(fil)
  u = (I[:,0]-v1).reshape(-1, 1)
  H_v1 = I-2*(u@u.T)/(u.T@u)
  if fil == 2:
    S = H_v1
    D = H_v1 @ A @ H_v1.T
    return S,D
  else:
    B = H_v1 @ A @ H_v1.T
    A_2 = B[1:fil,1:fil]
    S_2, D_2 =  diagRH(A_2,tol,K)
    D = np.zeros((fil,fil))
    D[0,0] = lamb1
    D[1:,1:] = D_2
    S = np.eye(fil)
    S[1:,1:] = S_2
    S = H_v1 @ S
    return S, D

def svd_reducida(A, k="max", tol=1e-15):

    fil, col = A.shape
    traspuesta = fil < col

    if traspuesta:
        A = A.T

    # Diagonalizar AtA
    At_A = multiplicarMatrices(A.T, A)

    S, D = diagRH(At_A, tol)
    V = S

    autoval = np.diag(D)

    # Ordenar autovalores de mayor a menor
    idx = sorted(range(len(autoval)),
                 key=lambda i: autoval[i],
                 reverse=True)

    autoval = autoval[idx]
    V = V[:, idx]

    # Eliminar autovalores nulos
    mask = autoval > tol

    autoval = autoval[mask]
    V = V[:, mask]

    # Limitar cantidad de valores singulares
    if k == "max":
        k = len(autoval)

    k = min(k, len(autoval))

    autoval = autoval[:k]
    V = V[:, :k]

    # Calcular valores singulares
    hatSig = autoval**(0.5)

    # Calcular U
    B = multiplicarMatrices(A, V)
    
    U_hat = np.zeros(B.shape)
    for j in range(k):
        U_hat[:, j] = B[:, j] / norma(B[:,j],2)


    # Deshacer transposición
    if traspuesta:
        U_hat, V = V, U_hat

    return U_hat, hatSig, V


def QR_con_GS(A):
  fil, col = A.shape
  if col != fil:
    return None
  Q = np.zeros((fil,col))
  R = np.zeros((fil,col))
  norm_a1 = norma(A[:,0],2)
  Q[:,0] = A[:,0]/norm_a1
  R[0,0] = norm_a1
  for j in range(1,fil):
    Q[:,j] = A[:,j]
    for k in range(j):
      R[k,j] = Q[:,k].T@Q[:,j]
      Q[:,j] -= R[k,j]*Q[:,k]
    norma_qj = norma(Q[:,j],2)
    Q[:,j] = Q[:,j]/norma_qj
    R[j,j] = norma_qj
  return Q,R

def res_tri(L,b, inferior=True):
  n = L.shape[0]
  x = np.zeros(n)
  bc = b.copy()
  if inferior:
    for i in range(n):
      for j in range(i):
        bc[i] -= L[i,j]*x[j]
      x[i] = bc[i] / L[i,i]
  else:
      for i in range(n-1, -1, -1):
        for j in range(i+1,n):
          bc[i] -= x[j]*L[i,j]
        x[i] = bc[i] / L[i,i]

  return x


def inversa(A):
  n = A.shape[0]
  I = np.eye(n)
  L,U,cant = calculaLU(A)
  if L is None:
    return None
  A_inv = np.zeros((n,n))
  for i in range(n):
    e_i = I[:,i]
    y = res_tri(L,e_i,True)
    x = res_tri(U,y,False)
    A_inv[:,i] = x
  return A_inv

def calculaCholesky(A):
    L,D,L = calculaLDV(A)
    D_raiz = np.diag(np.sqrt(np.diag(D)))
    return multiplicarMatrices(L,D_raiz)