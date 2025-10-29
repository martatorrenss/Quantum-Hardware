import numpy as np
import matplotlib.pyplot as plt

#BASIS (it s the pauli matrix z basis!!)
state0 = np.array([1,0],dtype=complex)
state1 = np.array([0,1],dtype=complex)

#Initial state
initial_state = state0

#Variables

#Frequencies
gamma = 2*np.pi*42.58e6 #s-1 (son MHz)
B = 100e-6 #T (son microT)
w_R = gamma*B
B_values=[50e-6,100e-6,200e-6,500e-6]
w_R_values=[gamma*B_values[0],gamma*B_values[1],gamma*B_values[2],gamma*B_values[3]]

#Constants
h = 1
#eigenvalues for hamiltonian (pauli matrix x) E=hw_r/2 H=-hw_R/2\sigma_x
E_0 = -h*w_R/2
E_1 = h*w_R/2


#Pauli matrices
sigma_x = np.array([[0,1],[1,0]],dtype=complex)
sigma_y = np.array([[0,-1j],[1j,0]],dtype=complex)
sigma_z = np.array([[1,0],[0,-1]],dtype=complex)
sigma_xz = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]],dtype=complex) #basis change

#Hamiltonian FROM X!!
eigenvalues=[E_0,E_1]

#Time-evolution operator

def time_evolution(state, energies, time):
    #canvi de base de l'estat de z a x
    state = np.matmul(sigma_xz,state)
    new_state = np.zeros(len(state), dtype=complex)
    for i in range(len(state)):
        new_state[i] = state[i]*np.exp(-1j*energies[i]/h*time)
        
    #canvi base de l'estat de x a z
    new_state = np.matmul(sigma_xz,new_state)
    new_state = new_state/np.linalg.norm(np.matmul(sigma_xz,new_state))
    
    return new_state

#Time-evolution of the state
time=[]
dt = (2*np.pi)/(200*w_R)
t=0

#Bloch-sphere coordinates for each time, angles
theta_values = []
phi_values = []

#bloch-sphere coordinates for each time, vector
rx_values = []
ry_values = []
ry_theoryvalues = []
rz_values = []
rz_theoryvalues = []

#probabilities
p0_values = []
p0_theoryvalues = []
p1_values = []

#for w_R in w_R_values:


for i in range(0,750):

    initial_state = time_evolution(initial_state, eigenvalues, dt)
    time.append(t*1e3)
    
    
    #bloch sphere coordinates, vector: cartesian coordinates
    #rx = (np.matmul(np.matmul(initial_state.conj().T,sigma_x),initial_state))
    psi = initial_state.astype(complex)
    
    rx = np.vdot(psi, sigma_x @ psi).real
    ry = np.vdot(psi, sigma_y @ psi).real
    ry_theory = np.sin(w_R*t)
    rz = np.vdot(psi, sigma_z @ psi).real
    rz_theory = np.cos(w_R*t)
    

    #ry = (np.matmul(np.matmul(initial_state.conj().T,sigma_y),initial_state))
    #rz = (np.matmul(np.matmul(initial_state.conj().T,sigma_z),initial_state))

    rx_values.append(rx)
    ry_values.append(ry)
    ry_theoryvalues.append(ry_theory)
    rz_values.append(rz)
    rz_theoryvalues.append(rz_theory)

    #bloch sphere coordinates for each time, angles: spherical coordinates
    #theta = 2*np.arccos(initial_state[0])
    #theta = np.arctan(np.sqrt(r_x**2 + r_y**2)/r_z)
    theta = 2*np.arccos(np.clip(np.abs(initial_state[0]), 0, 1))
    theta_values.append(theta)
    
    #phi=-1j*np.log(initial_state[1]/np.sin(theta/2))
    #phi=np.arctan(r_y/r_x) #cal posar algo per quan divideix entre 0
    phi = np.arctan2(np.real(ry), np.real(rx))
    phi_values.append(phi)
    
    #probabilities
    p0 = float(np.real(psi[0]*np.conj(psi[0])))
    p0_theory = np.cos(w_R/2*t)**2
    p0_values.append(p0)
    p0_theoryvalues.append(p0_theory)
    
    p1 = float(np.real(psi[1]*np.conj(psi[1])))
    p1_values.append(p1)

    #sphere plot
        # --- DIBUJAR ESFERA Y VECTOR EN i == 50 ---
    if i == 0 or i == 250 or i == 500 or i == 750:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Esfera unitaria
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        Xs = np.outer(np.cos(u), np.sin(v))
        Ys = np.outer(np.sin(u), np.sin(v))
        Zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(Xs, Ys, Zs, alpha=0.1, linewidth=0, antialiased=True)

        # Ejes
        L = 1.0
        ax.plot([-L, L], [0, 0], [0, 0], lw=1, color='black', label='x')
        ax.plot([0, 0], [-L, L], [0, 0], lw=1, color='gray',  label='y')
        ax.plot([0, 0], [0, 0], [-L, L], lw=1, color='brown', label='z')

        # Vector Bloch en este instante
        ax.plot([0, rx], [0, ry], [0, rz], lw=3, color='blue', label='Bloch vector')
        ax.scatter([rx], [ry], [rz], s=50, color='blue')

        # Estética
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_xlabel(r'$\langle\sigma_x\rangle$'); 
        ax.set_ylabel(r'$\langle\sigma_y\rangle$'); 
        ax.set_zlabel(r'$\langle\sigma_z\rangle$')
        ax.set_title(f'Bloch vector at t={t*1e3:.3f} ms')
        ax.legend()
        plt.show()


    t += dt
    
print(ry_values)
print(rz_values)
    
plt.title('Bloch Sphere: spherical coordinates')   
plt.plot(time,theta_values,label='$\\theta$', color='red')
plt.plot(time,phi_values, label='$\phi$', color='blue')
plt.xlabel('Time (ms)')
plt.ylabel('Angles (rad)')
plt.legend()
plt.show()

plt.title('Bloch Sphere: cartesian coordinates')
plt.plot(time,rx_values,label='$r_{x}$', color='black')
plt.plot(time,ry_values,label='$r_{y}$', color='red')
plt.plot(time,ry_theoryvalues,label='$r_{y}$ theory', color='orange', linestyle='--')
plt.plot(time,rz_values, label='$r_{z}$', color='blue')
plt.plot(time,rz_theoryvalues, label='$r_{z}$ theory', color='lightblue', linestyle='--')
plt.axhline(1,color='gray',linestyle='--')
plt.axhline(-1,color='gray',linestyle='--')
plt.ylabel('Bloch vector components')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()

plt.title('Probabilities')
plt.plot(time,p0_values,label='p(|0>)', color='blue')
plt.plot(time,p0_theoryvalues,label='p(|0>) theory', color='lightblue',linestyle='--')
plt.plot(time,p1_values,label='p(|1>)', color='red')
plt.axhline(1,color='gray',linestyle='--')
plt.axvline(0.235, color='lightgreen', linestyle='--', label='$T_{R}$ theory')
plt.axhline(0,color='gray',linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('Probabilities')
plt.legend(loc='right')
plt.show()


#AFEGIR VALORS TEÒRICS DE COS ETC PER COMPROVAR!

#ESFERA BLOCH COORDENADES CARTESIANES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # activa 3D

# --- Bloch sphere (malla + ejes) ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Esfera unitaria
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
Xs = np.outer(np.cos(u), np.sin(v))
Ys = np.outer(np.sin(u), np.sin(v))
Zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(Xs, Ys, Zs, alpha=0.1, linewidth=0, antialiased=True)

# Ejes
L = 1
ax.plot([-L, L], [0,0], [0,0], lw=1, label='x axis', color='black')   # x
ax.plot([0,0], [-L, L], [0,0], lw=1, label='y axis', color='gray')   # y
ax.plot([0,0], [0,0], [-L, L], lw=1, label='z axis', color='brown')   # z

#ax.plot(rx_values, ry_values, rz_values, lw=2, label='spin')          # trayectoria
#ax.scatter([rx[0]], [ry[0]], [rz[0]], s=40)  # punto inicial


#amb coordenades cartesianas
theta = np.array(theta_values, dtype=float)
phi   = np.array(phi_values,   dtype=float)

# (opcional) suaviza saltos de fase
phi = np.unwrap(phi)

rx = np.sin(theta)*np.cos(phi)
ry = np.sin(theta)*np.sin(phi)
rz = np.cos(theta)

# Reutiliza el bloque 3D de arriba para dibujar esfera y luego:
ax.plot(rx, ry, rz, lw=2,label='trajectory', color='blue')
ax.scatter([rx[0]], [ry[0]], [rz[0]], s=40)


# Vista y aspecto
ax.set_box_aspect([1,1,1])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.view_init(elev=25, azim=45)
plt.title('Bloch Sphere: Trajectory of the spin')
plt.legend()
plt.show()
