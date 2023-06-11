#!/usr/bin/env python
# coding: utf-8

# In[9]:


#PENDULO INVERTIDO SIMPLE RK4 

import numpy as np
import matplotlib.pyplot as plt

#ECUACION DIFERENCIAL ESTABLECIDA EN MODELOS DE ESTADO:
#Fuerza propuesta: F(t)=t
def f(t,x):
  return np.array([x[1],(((t)-c*x[1]+m2*l*(x[3]**2)*np.sin(x[2]))*(I+m2*(l**2))-((m2**2)*(l**2)*g*np.sin(x[2])*np.cos(x[2])))/((I+m2*(l**2))*(m1+m2)-((m2*l*np.cos(x[2]))**2)),x[3],((m2*g*l*np.sin(x[2]))*(m1+m1)-(m2*l*np.cos(x[2]))*((t)-c*x[1]+m2*l*(x[3])**2*np.sin(x[2])))/((I+m2*l**2)*(m1+m2)-(m2*l*np.cos(x[2]))**2)])

#PARAMETROS CONOCIDOS
m1=1
m2=0.1
c=0.1
I=0.025/3
l=0.5
g=9.81

#Condiciones iniciales
t0=0
x0=np.array([0,0,0,0])

#Intervalo de solución y tamaño de paso
tn=5
h=0.0001

#Método numérico
n=round((tn-t0)/h)
t=np.linspace(t0,tn,n+1)
x=np.zeros((n+1,len(x0)))
for i in range(n):
  k1=h*f(t[i],x[i,:])
  k2=h*f(t[i]+1/2*h,x[i,:]+1/2*k1)
  k3=h*f(t[i]+1/2*h,x[i,:]+1/2*k2)
  k4=h*f(t[i]+h,x[i,:]+k3)
  x[i+1,:]=x[i,:]+(1/6)*(k1+2*k2+2*k3+k4)
  print(f"x{i+1} = {np.round(x[i+1,:],4)}")
  

#Gráficas
#SALIDA PRINCIPAL
plt.plot(t,x[:,0],"b",label="x1")
plt.xlabel ("t")
plt.ylabel("x1")
plt.title("Posición, x1")
plt.grid(True)
plt.legend()
plt.show()


#GRÁFICA 1 CON ZOOM PARA VISUALIZAR LAS PERTURBACIONES
#plt.plot(t, x[:,0], "b", label="x1")
#plt.xlabel("t")
#plt.ylabel("x1")
#plt.title("Posición, x1")
#plt.grid(True)
#plt.legend()
# Cambiar el dominio
#x_ticks = np.arange(0, 2.5, 0.125)  # Genera ticks en el rango de 0 a 5 con incrementos de 0.25
#plt.xticks(x_ticks)  # Establece los ticks en el eje x
#plt.show()

#SALIDA SECUNDARIA PARA ASEGURAR EL CORRECTO FUNCIONAMIENTO DEL SISTEMA
plt.plot(t,x[:,2],"g", label="θ")
plt.xlabel ("t")
plt.ylabel("θ")
plt.title("Desplazamiento angular, θ")
plt.grid(True)
plt.legend()
plt.show()

#GRÁFICA EN CONJUNTO DE SALIDAS
plt.plot(t,x[:,0],"b",label="x")
plt.plot(t,x[:,2],"g",label="θ")
plt.xlabel ("t")
plt.title("Variables de estado Modelo No Lineal")
plt.grid(True)
plt.legend()
plt.show()


#x3
plt.plot(t,x[:,1],"r", label="x''")
plt.xlabel ("t")
plt.ylabel("x''")
plt.title("x''")
plt.grid(True)
plt.legend()
plt.show()

#x4
plt.plot(t,x[:,3],"k", label="θ''")
plt.xlabel ("t")
plt.ylabel("θ''")
plt.title("θ''")
plt.grid(True)
plt.legend()
plt.show()

#GRÁFICA CON TODAS VARIABLES DE ESTADO
plt.plot(t,x[:,0],"b",label="x")
plt.plot(t,x[:,1],"r",label="x''")
plt.plot(t,x[:,2],"g",label="θ")
plt.plot(t,x[:,3],"k",label="θ''")
plt.xlabel ("t")
plt.title("Variables de estado Modelo No Lineal")
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




