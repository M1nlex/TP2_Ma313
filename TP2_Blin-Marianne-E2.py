
import numpy as np
import time
import matplotlib.pyplot as plt
from math import *
from random import randint


def Inv_definie_positive(n): #Det(A) ≠ 0 ⇔ A inversible
    # Le produit d'une matrice et sa tranposée est
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


def Verif (A, Q, R):

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


def ResolGS (A, b):
    [Q, R] = DecompositionGS(A)
    Taug = np.column_stack((R, Q.T@b))
    X = ResolutionSystTriSup(Taug)  # RX = Qtb
    X_verif = np.linalg.solve(R, Q.T@b)
    # print (X,X_verif)
    return X, X_verif

def Comparer_temps_erreur ():
    pass

def test_proba_definie_positive(n,m):
    liste_abscisse = []
    liste_ordonee = []
    liste_oui_ou_non = []
    for i in range (1,n+1):
        print('taille de matrice en cours :' + str(i))
        liste_abscisse.append(i)
        nbr = 0
        for j in range(0,m):
            M = np.random.randn(i,i)
            try:
                A = np.linalg.inv(M)
                nbr += 1
                liste_oui_ou_non.append(1)
            except:
                liste_oui_ou_non.append(0)
                pass
        print(liste_oui_ou_non)
        liste_ordonee.append((nbr/m)*100)
    print(liste_ordonee)
    plt.plot(liste_abscisse,liste_ordonee)
    plt.xlabel('Taille matrice')
    plt.ylabel('Probabilité de matrice inverse sur '+str(m)+' essais (en %)')
    plt.show()

def test_non_inversible():
    """
    M = [[ 0.15637397,-1.01061316,0.60579566,1.13659488,0.4419333,-0.82592954,0.8849149,-1.97848384,0.3218673 ],
    [-0.21343234,-0.81372672,-0.17761841,0.97168728,-0.07965538,-0.50431718,-0.86084873,1.2422463,-0.69006776],
    [ 0.21670775,0.34851508,-0.20891298,0.13664464,-0.91301186,-0.68450093,-0.03658012,0.41443061,-0.2433923 ],
    [ 1.47643636,-0.07832674,-1.06252504,-0.34907162,-1.19899256,-1.13744233,-0.07244393,0.88603961,-1.29260512],
    [-2.05330516,-1.64254612,-0.01507904,1.35373427,0.44322605,0.71443811,-0.56660003,-2.05713702,1.26809434],
    [-1.7657797,-0.03848966,0.13864455,0.13872329,0.47935171,-1.22328327,-0.22977099,0.20996,1.23478186],
    [-2.009099,-0.35644065,0.01716443,1.89857132,0.35092185,2.67075468,-0.25405959,-0.01604311,-0.25356851],
    [ 0.65616668,-0.13382444,0.39131677,2.14508992,1.390733,-0.28189053,0.01693905,1.09073418,0.05453428],
    [-0.76616023,0.07651195,1.49811264,-0.28072279,1.04737987,0.61194093,-0.11707616,-1.79272935,-0.49199432]]
    """
    M=[[0,0],[1,1]]
    try:
        A = np.linalg.inv(M)
        print(A)
    except:
        print("non")

if __name__ == '__main__':
    """
    # partie1
    B = np.array([[6., 6., 16.], [-3., -9., -2.], [6., -6., -8.]])
    [Q, R] = DecompositionGS(B)
    V = Verif (B, Q, R)
    # Partie2
    b = np.random.randn(3, 1)
    X, X_verif = ResolGS(B, b)
    """
    test_proba_definie_positive(30,1000)
    #test_non_inversible()
