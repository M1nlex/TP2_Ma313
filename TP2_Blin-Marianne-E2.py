
import numpy as np
import time
import matplotlib.pyplot as plt
from math import *
from random import randint


def Inv_definie_positive(n): #Det(A) ≠ 0 ⇔ A inversible
    d=0
    while d == 0 :
        M = np.random.randn(n,n)
        A = np.dot(M.T,M) #M.T@M
        d = np.linalg.det(A)
    return A


def DecompositionGS (A):

    l,c = A.shape

    #Etape 1 : cas particulier où j = 0
    R = np.zeros((l,c))
    R[0,0] = np.linalg.norm(A[:,0])

    Q = np.zeros((l,c))
    Q[:,0] = A[:,0]/R[0,0]

    W = np.zeros((l,c))
    for j in range (1,c): #n-1 étapes restantes
        S = np.zeros((l,1))
        for i in range (0,j):
            R[i,j] = Q[:,i]@A[:,j] #1) Calcul des coeff de R rij
            S[:,0] += R[i,j] * Q[:,i]
        W[:,j] = A[:,j] - S[:,0] #2) Calcul d'un vect intermédiaire W
        R[j,j] = np.linalg.norm(W[:,j]) #3) Calcul de rjj
        Q[:,j] = W[:,j]/R[j,j] #4) Calcul de qj

    # print("Q = ", Q, "\n", "R = ", R)
    return Q,R


def Verif (A,Q,R):

    #Check if QR = A
    if (Q@R).all() == A.all() :
        print ("QR = A")
    else:
        print ("QR != A, erreur.")

    #Check if R upper triangular
    if np.allclose(R, np.triu(R)) :
        print ("R triangulaire sup.")
    else:
        print ("R n'est pas triangulaire sup., erreur.")

    #Check if Q orthogonale
    n = Q.shape[0]
    if (Q@Q.T).all() == (np.identity(n)).all() :
        print ("Q orthogonale.")
    else:
        print ("Q non orthogonale, erreur.")


def ResolutionSystTriSup(Taug): #Résolution d'un système triangulaire supérieur
    n,m = Taug.shape
    X = np.zeros(n) #Création d'un vecteur solution de taille n
    for k in range (n-1,-1,-1): #Par remontée
        S = 0
        for j in range (k+1,n):
            S += (Taug[k,j]*X[j])
        X[k] = (Taug[k,-1] - S)/Taug[k,k]
    return X


def ResolGS (A,b):
    [Q,R] = DecompositionGS (A)
    Taug = np.column_stack((R,Q.T@b))
    X = ResolutionSystTriSup(Taug) #RX = Qtb
    X_verif = np.linalg.solve(R,Q.T@b)
    #print (X,X_verif)
    return X,X_verif

def Comparer_temps_erreur ():
    

if __name__ == '__main__':
    #partie1
    B = np.array([[6.,6.,16.],[-3.,-9.,-2.],[6.,-6.,-8.]])
    [Q,R] = DecompositionGS (B)
    V = Verif (B,Q,R)
    #Partie2
    b = np.random.randn(3,1)
    X,X_verif = ResolGS (B,b)
