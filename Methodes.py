import numpy as np
import time
import matplotlib.pyplot as plt
from math import *
from itertools import chain
from random import randint

def ReductionGauss(Aaug): #Renvoie la matrice triangulaire supérieure augmentée du système à résoudre
    A = np.copy(Aaug)
    n,m = A.shape
    for k in range (0,n-1):
        Piv = A[k,k]
        if Piv == 0 :
            raise Exeption("Un pivot est nul")
        else :
            for i in range (k+1,n):
                G = A[i,k]/Piv
                A[i,:] -= G*A[k,:]
    return A

def ResolutionSystTriSup(Taug): #Résolution d'un système triangulaire supérieur
    n,m = Taug.shape
    X = np.zeros(n)
    for k in range (n-1,-1,-1): #Par remontée
        S = 0
        for j in range (k+1,n):
            S += (Taug[k,j]*X[j])
        X[k] = (Taug[k,-1] - S)/Taug[k,k]
    return X


def Gauss (A,B):
    Aaug = np.column_stack((A,B))
    Taug = ReductionGauss(Aaug)
    X = ResolutionSystTriSup(Taug)
    return X

def ResolutionSystTriInf(Taug): #Résolution d'un système triangulaire inférieur
    n,m = Taug.shape
    Y = np.zeros(n)
    for k in range (0,n): #Par descente
        S = 0
        for j in range (0,k+1):
            S += (Taug[k,j]*Y[j])
        Y[k] = (Taug[k,-1] - S)/Taug[k,k]
    return Y

def PivotPartiel(A,k,n): #Recherche du plus grand coeff de A en valeur absolue 
    q = k
    for p in range (k,n):
        if abs(A[p][k]) > abs(A[q][k]):
            q = p
    return q

def Transposition(A,i,p): #Intervertir les lignes
    temp = A[i].copy()
    A[i] = A[p]
    A[p] = temp
    return A

def ReductionChoixPivotPartiel(Aaug):
    A = np.copy(Aaug)
    n,m = A.shape
    for k in range(0,n-1):
        p = PivotPartiel(A,k,n)
        Piv = A[p,k]
        A = Transposition(A,k,p)
        if Piv == 0 :
            raise Exeption("Un pivot est nul")
        else :
            for i in range (k+1,n):
                G = A[i,k]/Piv
                A[i,:] -= G*A[k,:]
    return A


def GaussChoixPivotPartiel(A,B): #Renvoie la solution X de AX = B par la méthode du pivot partiel
    Aaug = np.column_stack((A,B))
    Taug = ReductionChoixPivotPartiel(Aaug)
    X = ResolutionSystTriSup(Taug)
    return X

def TranspositionLigne(A,i,imax): #Intervertir les lignes
    temp = A[i].copy()
    A[i] = A[imax]
    A[imax] = temp
    return A

def TranspositionColonne(A,j,jmax): #Intervertir les colonnes
    temp = A[j].copy()
    A[j] = A[jmax]
    A[jmax] = temp
    return A

def PivotTotal(A,k,n): #Recherche du plus grand coeff en valeur absolue dans toute la matrice A
    V = 0
    for i in range (k,n):
        for j in range (k,n):
            if abs(A[i][j]) > V:
                V = abs(A[i][j])
                imax = i
                jmax = j
    return (imax,jmax)

def ReductionChoixPivotTotal(Aaug):
    A = np.copy(Aaug)
    n,m = A.shape
    for k in range (0,n-1):
        (imax,jmax) = PivotTotal(A,k,n)
        
        for i in range (k,n):
            A = TranspositionLigne(A,i,imax)
            
        for j in range (k,n):
            A = TranspositionColonne(A,j,jmax)
            
        Piv = A[k,k]
        
        if Piv == 0 :
            raise Exeption("Un pivot est nul")
        else :
            for i in range (k+1,n):
                G = A[i,k]/Piv
                A[i,:] -= G*A[k,:]
    return A

def GaussChoixPivotTotal(A,B): #Renvoie la solution X de AX = B par la méthode du pivot total
    Aaug = np.column_stack((A,B))
    Taug = ReductionChoixPivotTotal(Aaug)
    X = ResolutionSystTriSup(Taug)
    return X

def DecompositionLU(A): #Renvoie la décomposition LU d'une matrice
    U = np.copy(A)
    n,m = U.shape
    L = np.identity(n)
    for k in range (0,n-1):
        Piv = U[k,k]
        if Piv == 0 :
            raise Exeption("Un pivot est nul")
        else :
            for i in range (k+1,n):
                G = U[i,k]/Piv
                U[i,:] -= G*U[k,:]
                L[i][k] = G
    return L,U

def ResolutionLU(A,B):
    L,U = DecompositionLU(A)
    Laug = np.column_stack((L,B))
    Y = ResolutionSystTriInf(Laug) #LY = B
    Uaug = np.column_stack((U,Y))
    X = ResolutionSystTriSup(Uaug) #UX = Y
    return X

def Cholesky (A):
    n,m = A.shape
    if n != m :
        print ("A n'est pas carrée")
        return
    L = np.zeros((n,n))
    for k in range (0,n):
        S_diag = 0
        for j in range (0,k) :
            S_diag += L[k,j]**2
        #Calcul des coeff diagonaux de L
        L[k,k] = sqrt(A[k,k]-S_diag)        
        for i in range (k+1,n):
            S = 0
            for j in range (0,k) :
                S += L[i,j] * L[k,j]
            #Calcul des coeff non diag de la colonne k
            L[i,k] = 1/L[k,k] * (A[i,k] - S) 
    return L

def ResolCholesky(A,B):
    L = Cholesky(A)
    Lt = L.T #Transposée de L
    
    Laug = np.column_stack((L,B))
    Y = ResolutionSystTriInf(Laug) #LY = B
    Ltaug = np.column_stack((Lt,Y))
    X = ResolutionSystTriSup(Ltaug) #LtX = Y
    return X
