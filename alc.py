import numpy as np
import cv2

__all__ = [
    "multiplicarMatrices",
    #"calculaLU",
    #"calculaLDV",
    "norma",
    "calcularAx",
    "productoPunto",
    #"metpot2k",
    #"diagRH",
    "svd_reducida",
    "QR_con_GS",
    "res_tri",
    "inversa",
    "calculaCholesky",
    "prodVectorial",
    "obtener_uv_con_M",
    "obtenerQ2d",
    "obtenerQ3d",
    "resolverDLT3D",
    "devolverMatricesCalibracion3d",
    "proyectar",
    "dibujar_circulo",
    "resolver2D",
    "distGolEnzo",
    "error_reproy",
    "armarA",
    "obtenerPose2D",
    "calcularMatrizK2D",
    "obtener_velocidad_kmh",
    "calculoLambdaMu",
    "calcularSalidaPOSIT",
    "evaluarPose",
    "aplicarPOSIT",
    "obtenerPoseObjetivo"    
    
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


def productoPunto(v,w):
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
    e = productoPunto(v_p,v)
    k += 1
  lamb = productoPunto(v_p.T, calcularAx(A,v_p))
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


def prodVectorial(v,u):
    p = np.array([v[1]*u[2]-v[2]*u[1],
                  v[2]*u[0]-v[0]*u[2],
                  v[0]*u[1]-v[1]*u[0]])
    return p


def obtener_uv_con_M(K,R,t,puntos):
    Rt = np.column_stack((R,t))
    M = multiplicarMatrices(K,Rt)
    uv = multiplicarMatrices(M,puntos).T
    uv = (uv / uv[:, -1:])[:,0:2]
    return uv


def obtenerQ2d(imgPts, objPts):
    n = len(objPts)
    uv = np.asarray(imgPts).reshape(n, 2)
    Q = np.zeros((n*2,9))

    for i in range(n):
        P = np.append(objPts[i], 1)
        u , v = uv[i]
        Q[2*i,0:3] = P
        Q[2*i,6:9] = -u*P

        Q[2*i+1,3:6] = P
        Q[2*i+1,6:9] = -v*P

    return Q


def resolver2D(Q):
    U, sigma, V = svd_reducida(Q)
    m = V.T[-1] #ultima fila de V traspuesta, seria la ultima columna de V
    return m.reshape(3,3)


def calcularV(H, i, j):
    v = np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ])
    return v


def armarA(Hlst):

    n = len(Hlst)
    A = np.zeros((2*n,6))
    j = 0
    for H in Hlst:
        v12 = calcularV(H,0,1)
        v11 = calcularV(H,0,0)
        v22 = calcularV(H,1,1)
        V = np.array([v12,v11-v22])

        A[2*j : 2*j+2,:] = V
        j = j + 1

    return A


def calcularMatrizK2D(A):
    U, S, V = svd_reducida(A, tol=1e-20)
    b = V[:,-1] # la transpongo para quedarme con la ultima columna de V
                 # el cual me da el valor singular minimo
    B = np.array([
                [b[0],b[1],b[3]],
                [b[1],b[2],b[4]],
                [b[3],b[4],b[5]]
                ])

    # Intentar corregir el signo de B

    #if np.all(autovalores < 0):
    #    B = -B
    autovalores = np.linalg.eigvalsh(B)

    if np.all(autovalores < 0):
        B = -B

    L = calculaCholesky(B)

    K = np.linalg.inv(L.T)
    K = K / K[2,2]
    return K


def obtenerPose2D(Hj,K):
    K_inv = inversa(K)
    lamb = 1/ norma(productoPunto(K_inv,Hj[:,0]),2)
    r1 = lamb*(productoPunto(K_inv,Hj[:,0]))
    r2 = lamb*(productoPunto(K_inv,Hj[:,1]))
    r3 = prodVectorial(r1,r2)
    R_t = np.array([r1,r2,r3])
    R = R_t.T

    tj = lamb*(productoPunto(K_inv,Hj[:,2]))

    return R, tj


def distGolEnzo(pt,H):
    p = np.array([pt[0],pt[1],1])
    U,V, W  = calcularAx(inversa(H), p)
    x, y = U / W, V/W
    return np.sqrt((105-x)**2+(34-y)**2)


def proyectar(H, x, y):
    q = calcularAx(H , np.array([x, y, 1.0]))
    return q[0] / q[2], q[1] / q[2]       # (u, v) en píxeles


def dibujar_circulo(img, H, cx, cy, color=(0, 255, 0), grosor=2):
    t = np.linspace(0,2*np.pi, 60)
    pts = [proyectar(H,cx+1*np.cos(a),cy+1*np.sin(a))for a in t]
    pts = np.array(pts,dtype=np.int32)
    cv2.polylines(img,[pts],True,color,grosor)


def resolverDLT3D(Q):
    U, sigma, V = svd_reducida(Q)
    m = V.T[-1] #ultima fila de V traspuesta, seria la ultima columna de V
    M = m.reshape(3,4)
    return M


def devolverMatricesCalibracion3d(M, misma_focal=False, sin_skew=False):
    J = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
    ])

    A = M[:,:3].copy()
    b = M[:,3].copy()
    B = multiplicarMatrices(J,multiplicarMatrices(A.T,J))

    Q, R = QR_con_GS(B)

    K = multiplicarMatrices(J,multiplicarMatrices(R.T,J))

    # Ajustamos signos de la diagonal de K
    D = np.diag(np.sign(np.diag(K)))
    K = multiplicarMatrices(K, D) # para que quede positiva

    # Normalizamos K[2,2] = 1
    escala = K[2, 2]
    K = K / escala
    A = A / escala
    b = b / escala

    # Restricciones sobre K
    if misma_focal:
        f = (K[0,0]+K[1,1])/2 # tomo la media para no alejarme de sus valores originales
        K[0,0] = K[1,1] = f

    if sin_skew:
        K[0,1] = 0

    # Teniendo la K definitiva, calculamos
    C = multiplicarMatrices(inversa(K), A)
    U, sigma, V = svd_reducida(C)
    R_hat = multiplicarMatrices(U, V.T)
    t = calcularAx(inversa(K), b)

    # Corregimos el signo pidiendo que el det sea <0, ya que solo puede ser -1 o 1. Pero los valores no son exactos, asi que si es <0 tiene det cercano a -1
    if np.linalg.det(R_hat) < 0:
        R_hat = -R_hat
        t = -t

    return R_hat, t, K


def error_reproy(uv_calc, u_real, v_real):
    uv_real = np.column_stack((u_real, v_real))
    d = np.sqrt(np.sum((uv_calc - uv_real)**2, axis=1))
    return d.mean(), d.max()


def obtener_velocidad_kmh(puntos_px, H, segs):
    inv_H = inversa(H)
    # Pasamos de puntos de la imagen a puntos reales, y agregamos 1 al final
    puntos_reales = np.array([calcularAx(inv_H, np.append(punto, 1)) for punto in puntos_px])
    puntos_reales = puntos_reales / puntos_reales[:,2:3]

    # Calculamos el vector desplazamiento entre un punto y el siguiente, para todos los puntos
    desplazamientos = np.diff(puntos_reales, axis=0)

    # Pasamos los vectores desplazamiento a un número a traves de la norma
    metros_recorridos = norma(desplazamientos.T, 2)

    # Calculamos la variación de tiempo entre una imagen y la otra, para todas las imagenes
    variaciones_tiempo = np.diff(segs)

    # Calculamos la velocidad como la distancia recorrida entre el tiempo transcurrido
    velocidades_ms = metros_recorridos / variaciones_tiempo
    velocidades_kmh = velocidades_ms * 3.6

    return velocidades_kmh


def calculoLambdaMu(I_plano, J_plano):
    dif_normas = norma(J_plano, 2)**2 - norma(I_plano, 2)**2

    menos_doble_producto = -2 * productoPunto(I_plano, J_plano)

    modulo = np.sqrt(dif_normas**2 + menos_doble_producto**2)

    lamb = np.sqrt(max(0, (modulo + dif_normas) / 2))
    mu = np.sqrt(max(0, (modulo - dif_normas) / 2))

    # Ajustamos el signo de mu
    if menos_doble_producto < 0:
        mu = -mu

    return float(lamb), float(mu)


def calcularSalidaPOSIT(I_completo, J_completo, fx, pixel_ref_centrado, punto_ref_real):
    Z0 = fx / norma(I_completo, 2)
    
    i = I_completo / norma(I_completo, 2)
    j = J_completo / norma(J_completo, 2)

    k = prodVectorial(i, j)

    R = np.array([i, j, k])

    x_ref_cam = (Z0 / fx) * pixel_ref_centrado[0]
    y_ref_cam = (Z0 / fx) * pixel_ref_centrado[1]

    punto_ref_cam = np.array([x_ref_cam, y_ref_cam, Z0])

    punto_ref_rotado = calcularAx(R, punto_ref_real)

    t = punto_ref_cam - punto_ref_rotado

    return i, j, k, Z0, R, t


def evaluarPose(R, t, puntos_reales, puntos_px, K):
    n = len(puntos_reales)

    profundidades = np.zeros(n)
    uv_calculados = np.zeros((n, 2))

    for p in range(n):
        punto_cam = calcularAx(R, puntos_reales[p]) + t
        profundidades[p] = punto_cam[2]

        if profundidades[p] > 0:
            pixel_homog = calcularAx(K, punto_cam)
            uv_calculados[p] = pixel_homog[:2] / pixel_homog[2]

    validas = int(np.sum(profundidades > 0))

    if validas < n:
        return validas, float("inf")

    error_cuadrado = np.sum((uv_calculados - puntos_px)**2, axis=1)

    rms = np.sqrt(np.sum(error_cuadrado) / n)

    return validas, float(rms)


def aplicarPOSIT(puntos_reales, puntos_px, K, MaxIt, atol):
    n = len(puntos_reales)
    fx = K[0, 0]
    fy = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]

    u_centrado = puntos_px[:, 0] - cu
    v_centrado = (puntos_px[:, 1] - cv) * fx / fy

    punto_ref_real = puntos_reales[0]

    pixel_ref_centrado = np.array([u_centrado[0], v_centrado[0]])

    vecs_relativos = puntos_reales - punto_ref_real

    A = vecs_relativos[1:]

    A_plano = A[:, :2]

    A_traspuesta = A_plano.T

    A_pinv = multiplicarMatrices(inversa(multiplicarMatrices(A_traspuesta, A_plano)),A_traspuesta)

    normal_plano = np.array([0.0, 0.0, 1.0])
    
    epsilon = np.zeros(n - 1)

    rms_anterior = float("inf")
    rms_min = float("inf")

    R_mejor = None
    t_mejor = None

    for iteracion in range(MaxIt):

        b_u = u_centrado[1:] * (1 + epsilon) - u_centrado[0]
        b_v = v_centrado[1:] * (1 + epsilon) - v_centrado[0]

        I_xy = calcularAx(A_pinv, b_u)
        J_xy = calcularAx(A_pinv, b_v)

        I_plano = np.array([I_xy[0], I_xy[1], 0.0])
        J_plano = np.array([J_xy[0], J_xy[1], 0.0])

        lamb, mu = calculoLambdaMu(I_plano, J_plano)

        rms_actual = float("inf")

        for signo in [1, -1]:

            I_completo = I_plano + signo * lamb * normal_plano
            J_completo = J_plano + signo * mu * normal_plano

            i, j, k, Z0, R, t = calcularSalidaPOSIT(
                I_completo,
                J_completo,
                fx,
                pixel_ref_centrado,
                punto_ref_real
            )

            validas, rms = evaluarPose(R, t, puntos_reales, puntos_px, K)

            if validas == n and rms < rms_actual:

                R_actual = R
                t_actual = t
                k_actual = k
                Z0_actual = Z0
                rms_actual = rms

        if rms_actual == float("inf"):
            break

        if rms_actual < rms_min:

            R_mejor = R_actual
            t_mejor = t_actual
            rms_min = rms_actual
            
        mejora_rms = rms_anterior - rms_actual

        if mejora_rms < atol:
            break

        for p in range(n - 1):
            epsilon[p] = productoPunto(A[p], k_actual) / Z0_actual

        rms_anterior = rms_actual

    if R_mejor is None:
        raise ValueError("No se encontro una pose valida")

    return R_mejor, t_mejor


def obtenerPoseObjetivo(R, t):
    centro_camara = -calcularAx(R.T, t)

    altura = centro_camara[2]

    centro_objetivo = np.array([0.0, 0.0, altura])

    R_objetivo = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    
    t_objetivo = -calcularAx(R_objetivo, centro_objetivo)

    return R_objetivo, t_objetivo


def obtenerQ3d(imgPts, objPts):
    n = len(objPts)
    uv = np.asarray(imgPts).reshape(n, 2)
    Q = np.zeros((n*2,12))

    for i in range(n):
        P = np.append(objPts[i], 1)
        u , v = uv[i]
        Q[2*i,0:4] = P
        Q[2*i,8:12] = -u*P

        Q[2*i+1,4:8] = P
        Q[2*i+1,8:12] = -v*P

    return Q
