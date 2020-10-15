
import numpy as np
import time
import matplotlib.pyplot as plt
from math import *
from random import randint
from Methodes import *


def Inv_definie_positive(n): #Det(A) ≠ 0 ⇔ A inversible
    d=0
    while d == 0 :
        M = np.random.randn(n,n)
        A = np.dot(M.T,M) #M.T@M
        d = np.linalg.det(A)
    return A


def DecompositionGS(A):

    l, c = A.shape

    # Etape 1 : cas particulier où j = 0
    R = np.zeros((l, c))
    R[0, 0] = np.linalg.norm(A[:, 0])

    Q = np.zeros((l, c))
    Q[:, 0] = A[:, 0]/R[0, 0]

    W = np.zeros((l, c))
    for j in range(1, c):  # n-1 étapes restantes
        S = np.zeros((l, 1))
        for i in range(0, j):
            R[i, j] = Q[:, i]@A[:, j]  # 1) Calcul des coeff de R rij
            S[:, 0] += R[i, j] * Q[:, i]
        W[:, j] = A[:, j] - S[:, 0]  # 2) Calcul d'un vect intermédiaire W
        R[j, j] = np.linalg.norm(W[:, j])  # 3) Calcul de rjj
        Q[:, j] = W[:, j]/R[j, j]  # 4) Calcul de qj

    # print("Q = ", Q, "\n", "R = ", R)
    return Q, R


def Verif(A, Q, R):

    # Check if QR = A
    if (Q@R).all() == A.all():
        print("QR = A")
    else:
        print("QR != A, erreur.")

    # Check if R upper triangular
    if np.allclose(R, np.triu(R)):
        print("R triangulaire sup.")
    else:
        print("R n'est pas triangulaire sup., erreur.")

    # Check if Q orthogonale
    n = Q.shape[0]
    if (Q@Q.T).all() == (np.identity(n)).all():
        print("Q orthogonale.")
    else:
        print("Q non orthogonale, erreur.")


def ResolutionSystTriSup(Taug):  # Résolution d'un système triangulaire supérieur
    n, m = Taug.shape
    X = np.zeros(n)  # Création d'un vecteur solution de taille n
    for k in range(n-1, -1, -1):  # Par remontée
        S = 0
        for j in range(k+1, n):
            S += (Taug[k, j]*X[j])
        X[k] = (Taug[k, -1] - S)/Taug[k, k]
    return X


def ResolGS(A, b):
    [Q, R] = DecompositionGS(A)
    Taug = np.column_stack((R, Q.T@b))
    X = ResolutionSystTriSup(Taug)  # RX = Qtb
    return X


def Verif2(X, A, B):
    # Check if X = X_verif
    X_verif = np.linalg.solve(A,B)
    if (X).all() == (X_verif).all():
        print("X = X_verif, la fonction fonctionne :)")
    else:
        print("Erreur.")


def Comparer_temps():
    TC = [] #Temps de calcul décompo Cholesky
    TGSCP = [] #Temps de calcul GaussSansChoixPivot
    TGCPP = [] #Temps de calcul GaussChoixPivotPartiel
    TGCPT = [] #Temps de calcul GaussChoixPivotTotal
    TGS = [] #Temps de calcul Gram-Schmidt
    TLU = [] #Temps de calcul avec décompo LU
    TNP =[] #Temps de calcul numpy

    N = [] #Taille matrice
    for k in range (1, 500, 2):
        N.append(k)

        A = Inv_definie_positive(k)
        B = np.random.randn(k,1)

        t0 = time.perf_counter()
        X1 = ResolCholesky(A,B)
        t = time.perf_counter()

        a = t - t0

        t1 = time.perf_counter()
        X2 = Gauss(A,B)
        t2 = time.perf_counter()

        b = t2 - t1

        t3 = time.perf_counter()
        X3 = GaussChoixPivotPartiel(A,B)
        t4 = time.perf_counter()

        c = t4 - t3

        t5 = time.perf_counter()
        X4 = GaussChoixPivotTotal(A,B)
        t6 = time.perf_counter()

        d = t6 - t5

        t7 = time.perf_counter()
        X5 = ResolGS (A,B)
        t8 = time.perf_counter()

        e = t8 - t7

        t9 = time.perf_counter()
        X6 = ResolutionLU(A,B)
        t10 = time.perf_counter()

        f = t10 - t9

        t11 = time.perf_counter()
        X7 = np.linalg.solve(A,B)
        t12 = time.perf_counter()

        g = t12 - t11

        TC.append(a)
        TGSCP.append(b)
        TGCPP.append(c)
        TGCPT.append(d)
        TGS.append(e)
        TLU.append(f)
        TNP.append(g)

    plt.ylabel("Temps de calcul (s)")
    plt.yscale("log")

    plt.xlabel("Taille de la matrice")
    #plt.xscale("log")

    plt.plot(N,TC,".:",label = "Décomposition de Cholesky")
    plt.plot(N,TGSCP,".:",label = "Gauss sans choix pivot")
    plt.plot(N,TGCPP,".:",label = "Gauss choix pivot partiel")
    plt.plot(N,TGCPT,".:",label = "Gauss choix pivot total")
    plt.plot(N,TGS,".:",label = "Gram-Schmidt")
    plt.plot(N,TLU,".:",label = "Décomposition LU")
    plt.plot(N,TNP,".:",label = "Numpy")

    plt.legend(loc = "upper left")
    plt.title("Temps de résolution d'un système en fonction de la taille n de A \n", fontsize=12)

    plt.show()


if __name__ == '__main__':
    # partie1
    A = np.array([[6., 6., 16.], [-3., -9., -2.], [6., -6., -8.]])
    [Q, R] = DecompositionGS(A)
    V = Verif (A, Q, R)
    # Partie2
    n = int(input("Taille désirée : "))
    C = Inv_definie_positive(n)
    D = np.random.randn(n, 1)
    X = ResolGS(C, D)
    V_2 = Verif2(X, C, D)
    #Partie3
    Temps = Comparer_temps()
