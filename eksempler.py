import numpy as np
import matplotlib.pyplot as plt

#General note: This is only the most important functions. If you need a specific function, you can google if it exists. Or make it yourself. Or ask Vasin for help.
#General note 2: Don't use numpy.matrix / numpy.mat. It's outdated.
#General note 3: For more thorough descriptions, see here: https://numpy.org/doc/stable/reference/routines.linalg.html?highlight=eigenval

#Create an m*n matrix using 0's
m=4 #rows
n=3 #columns 
a = np.zeros((m,n)) #pay attention to double parantheses
print("Real")
print(a)
a=np.zeros((m,n),dtype=complex) # In case you want complex numbers in your array (This is useful for KJM2601, on a side note)
print("Complex")
print(a)


#Create an n*m matrix writing in the numbers directly (here, n=m=3)
A = np.array([[0.1, 0.6, 0.51], [0.35, 0.2, 0.19], [0.55, 2.0, 0.3]])



#Accessing the matrix
A[2,1]=0.2; # 3. row, 2. column (don't forget that counting starts at 0)


#Matrix-vector-product:
A=A #Leave a unchanged, but you can really use any value.
v=np.random.random(3); # A random vector
#v=[0.64,0.43,0.63] # A random vector (Question: Is it random?)

print("Random vector:")
print(v)
print("Matrix-vector product")

print(np.dot(A,v)/np.linalg.norm(np.dot(A,v))); #Dot is the dot product, norm is...well...the norm.


#Matrix-matrix-product:
A_squared=np.dot(A,A)
print("A_squared:");
print(A_squared)
print("A_cubed:");
A_cubed=np.linalg.multi_dot([A,A,A]) #There's other (more efficient?) ways to take A^n
print(A_cubed)
A_3=np.linalg.matrix_power(A,3)
print("A to the power of 3:")
print(A_3)

#Finding eigenvalues
eigenvalue, eigenvector= np.linalg.eig(A)
print("Eigenvalues: ")
print(eigenvalue)
print("Eigenvectors: ") #The eigenvectors is itself a matrix. So, technically speaking, we got P of the matrix diagonalisation
print(eigenvector)

#Diagonalizing a matrix
eigenvalue, eigenvector= np.linalg.eig(A) # Get eigenvalues and eigenvectors
P=eigenvector #that's P, as I said before
D=np.diag(eigenvalue) 
P_invers=np.linalg.inv(P) # This is the inverse! This works 
print(A-np.linalg.multi_dot([P,D,P_invers])) # you can't write P*D*P_invers, that's where the multi_dot comes in.
#Pay attention to what is printed here. We get numbers around the size 10^-16. If you're dealing with numbers larger than 0.01, you can (usually) assume that 10^-16 is just a fancy zero. (Rounding error)


#Finding the determinant of a matrix:
determinant=np.linalg.det(A)
print("Determinant")
print(determinant)

#Checking if a matrix is orthogonal
A = np.array([[2/3,1/3,2/3], [-2/3,2/3,1/3], [1/3,2/3,-2/3]])
A_transpose=A.T # The transposed of A.
print(np.dot(A_transpose,A)) #This is supposed to be the identity matrix.


#Some other useful functions:
v=(1, 2, 3)
V=np.diag(v) #create a diagonal matrix of size len(v)*len(v) with all values equal to zero, except V[i,i] for all i <len(V), that is, the diagonal entries
print("Diagonal matrix usig np.diag:")
print(V)

#Some general "ideas" how to use coding for the exam:
#Example: Trial exam 2, task 3.
def T(v):
    return np.dot([[0,-1],[-1,0]],v)+ [1,1]
x=[5,5]
y=[-30,5]
z=[0.5,0.5] #Find this by trial and error
w=[5,-4]
points=[x,y,z,w]
counter=0;
colors=["b","g","r","c"]
for v in points:
    plt.scatter(v[0],v[1],color=colors[counter],label="v%d"%counter)
    plt.scatter(T(v)[0],T(v)[1],color=colors[counter],label="T(v%d)"%counter)
    plt.scatter(T(T(v))[0],T(T(v))[1],color=colors[counter],label="T(T(v%d))"%counter)
    plt.scatter(T(T(T(v)))[0],T(T(T(v)))[1],color=colors[counter],label="T(T(T(v%d)))"%counter)
    plt.legend()
    counter+=1;
plt.show()
#We can "see" that we get a reflection through some line (and that both 0.5/0.5 and 5,-4 lie on it). Also, we can, for some vectors x,y,z, test the "order":
print("Checking T(T(v))-v)")
for v in points:
    print(v-T(T(v))); #This should be [0, 0] as we "guess" the order to be 2 (that is, T^2=ID <==> T(T(v))=v))
