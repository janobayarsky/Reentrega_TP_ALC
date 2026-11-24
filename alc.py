import numpy as np
import cv2

__all__ = [
    "multiplicarMatrices",
    "calcularAx",
    "svd_reducida",
    "QR_con_GS",
    "inversa",
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
    mask = abs(autoval) > tol

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
    L, D, V = calculaLDV(A)
    D_raiz = np.diag(np.sqrt(np.diag(D)))
    return multiplicarMatrices(L, D_raiz)


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
    # U, sigma, V = svd_reducida(Q)
    # m = V.T[-1] #ultima fila de V traspuesta, seria la ultima columna de V
    # return m.reshape(3,3)

    # como svd_reucida descarta autovalores de Q.T Q, 
    # mejor calculamos los autovalores de Q.T Q usando diagRH.
    # Este método, factoriza la matriz en S D S.T. la diag de D son los autovalores,
    # y S son los autovectores como columna. Nuestro método busca el autovector asociado al autovalor mas chico, 
    # así que la matriz S de diagRH cumple esto. No necesitamos la SVD. 
    # Es mejor evitar usarla para no descartar valores singulares (mask = abs(autoval) > tol).
    # Llamamos V a la S de diagRH ya que corresponde a los autovalores de Q.T Q, que coincide con V de la SVD.

    QtQ = multiplicarMatrices(Q.T, Q)
    V, D = diagRH(QtQ)
    # Buscamos el autovalor más cercano a cero en la diagonal de D. No están ordenados como en la SVD, así que recorremos todos
    pos = 0 
    for i in range(1, len(D)):
        if abs(D[i, i]) < abs(D[pos, pos]):
            pos = i
    # La columna correspondiente de V contiene los 9 coeficientes de H
    h = V[:, pos]
    return h.reshape(3, 3)

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
    # Tenemos que cada H = (c1|c2|c3) = sK(r1|r2|t), por lo que
    # c1=sKr1, c2=sKr2. Al despejarlas, queda que
    # r1 = 1/s K^-1 c1, r2 = 1/s K^-1 c2, siendo r1 y r2 columnas de rotacion.
    # Por esta prop, r1.T r2 = 0, r1.T r1 = r2.T r2 = 1
    # Se define B = K^-t K^-1, y las expresiones de antes se convierten en
    # c1.T B c1 - c2.T B c2 = 0, c1.T B c2 = 0.
    # Cada homografía aporta exactamente dos ecuaciones linelaes en las 6 componentes independientes de B.
    # Las incognitas dependen de K y los valores de las columnas de H.
    # Una vez obtenidas las Hj, se encuentra K planteando Ab=0, donde se expresa 
    # b = [B11 B12 B22 B13 B23 B33].T
    # Se define vij, para las columnas ci,cj de H tal que ci.T B cj = vij.T b, y se obtiene
    # v12.T b = 0, (v11 - v22).T b = 0, que corresponden a dos filas de A.
    # Las otras filas de A se arman con las homografias de todas las vistas.
    # Se apilan las 2n ecuaciones de n vistas para obtener A.

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

def esDefinidaPositiva(B):
    #Criterio de Sylvester: los 3 menores principales deben tener determinante mayor estricto a 0. Calculamos el determinante a mano:
    m1 = B[0,0]
    m2 = B[0,0]*B[1,1] - B[0,1]*B[1,0]
    m3 = (B[0,0]*(B[1,1]*B[2,2] - B[1,2]*B[2,1])
        - B[0,1]*(B[1,0]*B[2,2] - B[1,2]*B[2,0])
        + B[0,2]*(B[1,0]*B[2,1] - B[1,1]*B[2,0]))
    return m1 > 0 and m2 > 0 and m3 > 0

def calcularMatrizK2D(A):

    U, S, V = svd_reducida(A, tol=1e-20)
    b = V[:,-1] 

    # b es la sol que minimiza la norma de Ab, que al igual que en el caso de m, es la ultima columna
    # de la matriz V de la SVD de A. Se elije el signo de B que hace que B quede DP.

    B = np.array([
                [b[0],b[1],b[3]],
                [b[1],b[2],b[4]],
                [b[3],b[4],b[5]]
                ])

    autovalores = np.linalg.eigvalsh(B)

    if esDefinidaPositiva(B):
       pass
    # un criterio para ver que la matriz sea DP es que todos sus autovalores sean positivos. Si son todos negativos, cambiamos su signo
    elif esDefinidaPositiva(-B):
        B = -B
    else:
        raise ValueError("B no es definida positiva: revisar homografías / cantidad de vistas")

    # A partir de B, se usa la factorizacion de Cholesky B=L L.T, y como 
    # B = K^-T K^-1, con K^-1 triangular superior (ya que K era triangular superior), 
    # se tiene que L = K^-T, es decir K = L.T^-1. Se normaliza K dividiendo por K33 para que quede K33=1.
    # Verificamos que B sea definida positiva antes de aplicar Cholesky, para que la función avise si algo sale mal,
    # en vez de devolver una K inválida

    L = calculaCholesky(B)



    K = inversa(L.T)
    K = K / K[2,2]
    return K


def obtenerPose2D(Hj,K):
    # Una vez calculada la homografía de la vista y la matriz K para atodas las vistas,
    # podemos obtener los parametros extrínsecos de la vista j (Rj y tj).
    # Llamamos a las columnas de Hj c1,c2,c3.
    # Decimos que lambda = 1 / norma de K^c1 = 1 / s
    # Las columas de R las obtenemos de la siguiente forma: r1 = lambda K^-1 c1,
    # r2 = lambda K^-1 c2, y r3 = r1 x r2 (producto cruz, para obtener un vector ortogonal a r1 y r2).
    # Construimos Rj como las columnas que calculamos, y tj = lambda K^-1 c3.
    K_inv = inversa(K)

    Kinv_c1 = calcularAx(K_inv, Hj[:,0])
    Kinv_c2 = calcularAx(K_inv, Hj[:,1])
    Kinv_c3 = calcularAx(K_inv, Hj[:,2])


    lamb = 1/ norma(Kinv_c1,2)

    # La homografía se estima salvo un factor de escala, que puede ser negativo. 
    # Elegimos el signo de λ que deja al tablero delante de la cámara, 
    # porque la otra solución ubicaría al tablero detrás del plano imagen y no es físicamente válida.

    if Kinv_c3[2] < 0:
        lamb = -lamb


    r1 = lamb*(Kinv_c1)
    r2 = lamb*(Kinv_c2)
    r3 = prodVectorial(r1,r2)
    R = np.array([r1,r2,r3]).T

    # Al igual que antes, buscamos que R sea ortogonal, asi que buscamos la matriz ortogonal que mejor la paroxima.
    # Esto se resuelve calculando la SVD de R, obteniendo U Sigma V.T, e imponer que Sigma = Identidad.
    # La R nueva será U V.T, que al ser producto de ortogonales es ortogonal.
    # r1 x r2 no cambia si multiplicamos a r1 y a r2 por -1, por lo que R queda igual
    U, Sigma, V = svd_reducida(R)
    R = multiplicarMatrices(U,V.T)
    
    tj = lamb*(Kinv_c3)

    return R, tj


def distGolEnzo(pt, H):
    p = np.array([pt[0], pt[1], 1])
    U, V, W = calcularAx(inversa(H), p)
    x = U / W
    # La línea del arco está en x = 105 metros, y buscamos la distancia en linea recta desde enzo hacia ella. 
    return abs(105 - x)


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
    # Matriz de permutación que si se multiplica por derecha intercambia la primera col con la tercera, 
    # y si se multiplica por izquierda intercambia la primera fila con la tercera. Al hacer JAJ, se hacen ambas operaciones
    J = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
    ])

    A = M[:,:3].copy()
    b = M[:,3].copy()

    # Tenemos que M=(A|b)=K(R|t)=(KR|Kt), asi que A=KR, con K triangular superior y R ortogonal. Si factorizamos A con QR queda
    # QR = KR, pero Q y la segunda R son ortogonales, y la primera R y K son triangulares superiores. Queremos que KR aparezcan al reves.
    # Decimos que A.T=R.T K.T, y multiplicamos por matrices de permutacion ya que K.T es triangular inferior, y queremos que vuelva a ser
    # triangular superior. Al hacer estas operaciones, queda que J A.T J = J R.T K.T J = (J R.T J)(J K.T J), ya que como
    # J es una matriz de permutacion, es ortogonal (JJ=I). El primer parentesis es ortogonal, y el segundo triangular superior.
    # Factorizamos J A.T J = B, y queda B = Qgs Rgs. Qgs = (J R.T J), y Rgs = (J K.T J).
    # De esta forma, K.T = J Rgs J (ya que la inversa de J es J), y K = J Rgs.T J.
    # Calculada K, le hacemos las modificaciones correspondientes (skew y focal), y obtenemos R y t.
    # Se tiene que b = Kt, y obtenemos t multiplicando K^-1 b.

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

    # Teniendo la K definitiva, calculamos R y t.
    # Como originalmente R = K^-1 A era ortogonal, al modificar K se puede perder esta propiedad. 
    # Llamamos C = K^-1 A. Queremos la matriz R ortogonal que minimice la norma al cuadrado de C-R. Descomponemos C con una SVD, C= U Sigma V.T
    # La norma de C-R = norma (U sigma V.T - R) = norma (U.T(U sigma V.T - R)V), ya que al multiplicar una matriz ortogonal se mantiene la norma.
    # Volviendo a la norma nos queda la norma al cuadrado de Sigma - U.T R V, y llamo Z = U.T R V. Como U,R y V son ortogonales, Z tambien lo será,
    # Ya que el producto de ortogonales es ortogonal. 
    # Aplicando una norma Frobenius, que es igual a la raiz de la suma de cada elemento elevado al cuadrado de la matriz.
    # Cumple que la norma F al cuadrado de una matriz M es igual a la traza de M^t M.
    # De esta forma, la norma al cuadrado de Sigma - Z es igual a la traza de (Sigma - Z).T (Sigma - Z).
    # Distribuyendo, queda la traza de (Sigma.T Sigma) - 2 traza (Sigma.T Z) + traza de(Z Z.T).
    # El primer término, es tomar la traza de la matriz que tiene en la diagonal los valores singulares al cuadrado.
    # El tercer es la traza de la matriz identidad, ya que Z es ortogonal.
    # La segunda traza es igual a la sumatoria de valor singular i Zii, con i entre 1 y 3.
    # Como se quiere minimizar la norma, y el segundo término es negativo, se quiere maximizarlo, que seria achicar la norma.
    # Como Z es ortogonal, cada columna tiene norma 1, o sea que ningun elemento puede ser mayor a 1 en modulo.
    # Como queremos que sea lo mas grande posible positivo, y los sigma son positivos, entonces el max valor de la sumatoria se obtiene
    # tomando los elementos de la diagonal de Z iguales a 1 positivo. Dada esta elección, el resto de valores de la matriz Z deben ser iguales a 0,
    # ya que si alguno fuera mayor a 0 en modulo, ya no se cumpliría la norma 1. Determinamos que Z = I maximiza esta norma.
    # Volviendo a la expresión norma al cuadrado de Sigma - U.T R V, vimos que este segundo término (que habíamos llamado Z), debe ser igual al a identidad.
    # Despejando, se llega a que R = U I V.T = U V.T.
    # Concluimos que para obtener la matriz R de rotación mas cercana a K^-1 A, debemos hallar la SVD de esta expresión, y la R que buscamos será hacer U V.T.

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
