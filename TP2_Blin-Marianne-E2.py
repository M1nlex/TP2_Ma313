
import numpy as np
import time
import matplotlib.pyplot as plt
from math import *
from random import randint
from Methodes import *


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


def Comparer_temps(limTaille=100):
    TC = [] #Temps de calcul décompo Cholesky
    TGSCP = [] #Temps de calcul GaussSansChoixPivot
    TGCPP = [] #Temps de calcul GaussChoixPivotPartiel
    TGCPT = [] #Temps de calcul GaussChoixPivotTotal
    TGS = [] #Temps de calcul Gram-Schmidt
    TLU = [] #Temps de calcul avec décompo LU
    TNP =[] #Temps de calcul numpy

    N = [] #Taille matrice
    for k in range (1, limTaille, 1):
        N.append(k)
        # Génération matrice commune pour toutes les solutions
        A = Inv_definie_positive(k)
        B = np.random.randn(k,1)

        # Calcul temps de calcul pour chaque méthode
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
        # Ajout valeurs du temps de calcul pour cette itération
        TC.append(a)
        TGSCP.append(b)
        TGCPP.append(c)
        TGCPT.append(d)
        TGS.append(e)
        TLU.append(f)
        TNP.append(g)

    plt.ylabel("Temps de calcul (s)")
    #plt.yscale("log")

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

def Comparer_temps_moyenne(limTaille=100,NbrParTaille=10):
    TC = [] #Temps de calcul décompo Cholesky
    TGSCP = [] #Temps de calcul GaussSansChoixPivot
    TGCPP = [] #Temps de calcul GaussChoixPivotPartiel
    TGCPT = [] #Temps de calcul GaussChoixPivotTotal
    TGS = [] #Temps de calcul Gram-Schmidt
    TLU = [] #Temps de calcul avec décompo LU
    TNP =[] #Temps de calcul numpy

    TCE = [] #Erreur de calcul décompo Cholesky
    TGSCPE = [] #Erreur de calcul GaussSansChoixPivot
    TGCPPE = [] #Erreur de calcul GaussChoixPivotPartiel
    TGCPTE = [] #Erreur de calcul GaussChoixPivotTotal
    TGSE = [] #Erreur de calcul Gram-Schmidt
    TLUE = [] #Erreur de calcul avec décompo LU
    TNPE =[] #Erreur de calcul numpy

    N = [] #Taille matrice
    for k in range (1, limTaille, 1):

        # Création des listes pour moyenne temps / Supression des données des itérations précédentes
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        l5 = []
        l6 = []
        l7 = []

        # Création des listes pour moyenne erreur / Supression des données des itérations précédentes
        e1 = []
        e2 = []
        e3 = []
        e4 = []
        e5 = []
        e6 = []
        e7 = []

        N.append(k)
        print(k)

        for j in range(0,NbrParTaille):

            # Génération matrice commune pour toutes les solutions
            A = Inv_definie_positive(k)
            B = np.random.randn(k,1)

            # Calcul temps de calcul pour chaque méthode
            t0 = time.perf_counter()
            X1 = np.transpose([ResolCholesky(A,B)])
            t = time.perf_counter()

            a = t - t0
            ae = np.linalg.norm( abs((A@X1)-B) )

            t1 = time.perf_counter()
            X2 = np.transpose([Gauss(A,B)])
            t2 = time.perf_counter()

            b = t2 - t1
            be = np.linalg.norm( abs((A@X2)-B) )

            t3 = time.perf_counter()
            X3 = np.transpose([GaussChoixPivotPartiel(A,B)])
            t4 = time.perf_counter()

            c = t4 - t3
            ce = np.linalg.norm( abs((A@X3)-B) )

            t5 = time.perf_counter()
            X4 = np.transpose([GaussChoixPivotTotal(A,B)])
            t6 = time.perf_counter()

            d = t6 - t5
            de = np.linalg.norm( abs((A@X4)-B) )

            t7 = time.perf_counter()
            X5 = np.transpose([ResolGS (A,B)])
            t8 = time.perf_counter()

            e = t8 - t7
            ee = np.linalg.norm( abs((A@X5)-B) )

            t9 = time.perf_counter()
            X6 = np.transpose([ResolutionLU(A,B)])
            t10 = time.perf_counter()

            f = t10 - t9
            fe = np.linalg.norm( abs((A@X6)-B) )

            t11 = time.perf_counter()
            X7 = np.linalg.solve(A,B)
            t12 = time.perf_counter()

            g = t12 - t11
            ge = np.linalg.norm( abs((A@X7)-B) )

            # Ajout de la valeur dans la liste pour le calcul de moyenne de temps
            l1.append(a)
            l2.append(b)
            l3.append(c)
            l4.append(d)
            l5.append(e)
            l6.append(f)
            l7.append(g)
            # Ajout de la valeur dans la liste pour le calcul de moyenne d'erreur
            e1.append(ae)
            e2.append(be)
            e3.append(ce)
            e4.append(de)
            e5.append(ee)
            e6.append(fe)
            e7.append(ge)

        # Ajout valeurs du temps de calcul pour cette itération
        TC.append(sum(l1)/NbrParTaille)
        TGSCP.append(sum(l2)/NbrParTaille)
        TGCPP.append(sum(l3)/NbrParTaille)
        TGCPT.append(sum(l4)/NbrParTaille)
        TGS.append(sum(l5)/NbrParTaille)
        TLU.append(sum(l6)/NbrParTaille)
        TNP.append(sum(l7)/NbrParTaille)

        # Ajout valeurs du temps de calcul pour cette itération
        TCE.append(sum(e1)/NbrParTaille)
        TGSCPE.append(sum(e2)/NbrParTaille)
        TGCPPE.append(sum(e3)/NbrParTaille)
        TGCPTE.append(sum(e4)/NbrParTaille)
        TGSE.append(sum(e5)/NbrParTaille)
        TLUE.append(sum(e6)/NbrParTaille)
        TNPE.append(sum(e7)/NbrParTaille)

    # Premier graph (temps)
    plt.ylabel("Temps de calcul (s)")
    #plt.yscale("log")

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

    # Deuxième graph (erreur)
    plt.ylabel("Erreur relative")
    #plt.yscale("log")

    plt.xlabel("Taille de la matrice")
    #plt.xscale("log")

    plt.plot(N,TCE,".:",label = "Décomposition de Cholesky")
    plt.plot(N,TGSCPE,".:",label = "Gauss sans choix pivot")
    plt.plot(N,TGCPPE,".:",label = "Gauss choix pivot partiel")
    plt.plot(N,TGCPTE,".:",label = "Gauss choix pivot total")
    plt.plot(N,TGSE,".:",label = "Gram-Schmidt")
    plt.plot(N,TLUE,".:",label = "Décomposition LU")
    #plt.plot(N,TNPE,".:",label = "Numpy")

    plt.legend(loc = "upper left")
    plt.title("Erreur relative de X en fonction de la taille des matrices\n", fontsize=12)

    plt.show()

if __name__ == '__main__':
    """
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
    """
    Temps = Comparer_temps_moyenne(500,20)
