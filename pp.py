import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from PIL import Image
st.set_page_config(page_title="LAB2")
class SignalConvolution:
    def __init__(self):
        self.time_delta = 1e-3  # Paso de tiempo para señales continuas
    def generate_signal(self, signal_type, domain="continuous"):
        """
        Genera una señal predefinida del tipo especificado
        """
        if domain == "continuous":
            return self.generate_continuous_signal(signal_type)
        else:
            return self.generate_discrete_signal(signal_type)
    def generate_continuous_signal(self, signal_type):
            Delta = self.time_delta
            t_full = np.arange(-10, 10, Delta)
            
            if signal_type == 1: 
                t11 = np.arange(0, 3, Delta)
                t12 = np.arange(3, 5, Delta)
                x_t1 = np.zeros_like(t_full)
                idx_start = np.searchsorted(t_full, 0)
                idx_end = np.searchsorted(t_full, 5)
                x_t1[idx_start:idx_start+len(t11)] = 2
                x_t1[idx_start+len(t11):idx_end] = -2
                return t_full, x_t1, (-10, 10, -2.5, 2.5)
            
            elif signal_type == 2:
                x_t2 = np.zeros_like(t_full)
                idx_start = np.searchsorted(t_full, -1)
                idx_end = np.searchsorted(t_full, 1)
                x_t2[idx_start:idx_end] = -t_full[idx_start:idx_end]
                return t_full, x_t2, (-10, 10, -1, 1)
            
            elif signal_type == 3:
                x_t3 = np.zeros_like(t_full)
                idx_start = np.searchsorted(t_full, -1)
                idx_mid1 = np.searchsorted(t_full, 1)
                idx_mid2 = np.searchsorted(t_full, 3)
                idx_end = np.searchsorted(t_full, 5)
                x_t3[idx_start:idx_mid1] = 2
                x_t3[idx_mid1:idx_mid2] = -2 * t_full[idx_mid1:idx_mid2] + 4
                x_t3[idx_mid2:idx_end] = -2
                return t_full, x_t3, (-10, 10, -2.5, 2.5)
            
            elif signal_type == 4:
                x_t4 = np.zeros_like(t_full)
                idx_start = np.searchsorted(t_full, -3)
                idx_mid = np.searchsorted(t_full, 0)
                idx_end = np.searchsorted(t_full, 3)
                x_t4[idx_start:idx_mid] = np.exp(t_full[idx_start:idx_mid])
                x_t4[idx_mid:idx_end] = np.exp(-t_full[idx_mid:idx_end])
                return t_full, x_t4, (-10, 10, 0, 1)

    def generate_discrete_signal(self, signal_type):
        if signal_type == 1:
            n = np.arange(-10, 11)
            x = np.where(np.abs(n) < 6, 6 - np.abs(n), 0)
            return n, x, (-10, 10, 0, 6)
        
        elif signal_type == 2:
            n = np.arange(-10, 11)
            h = np.where(np.abs(n) <= 5, 1, 0) - np.where(n-5 >= 6, 1, 0)
            return n, h, (-10, 10, -1, 1)
        
        elif signal_type == 3:
            n = np.arange(-10, 11)
            x = np.where(n >= -2, 1, 0) - np.where( n >= 9, 1, 0)
            return n, x, (-10, 10, -1, 1)
        
        elif signal_type == 4:
            n = np.arange(-10, 11)
            h = (9/11)**n * (np.where(n >= -1, 1, 0) - np.where(n >= 10, 1, 0))
            return n, h, (-10, 10, -1, 1)
    def perform_convolution(self, time1, signal1, time2, signal2, domain="continuous"):
        """
        Realiza la convolución entre dos señales y visualiza el proceso
        """
        if domain == "continuous":
            delta = self.time_delta
            step = 200
        else:
            delta = 1
            step = 1
        # Reflejar y desplazar la primera señal
        time1_reflected = -time1[::-1]
        signal1_reflected = signal1[::-1]
        
        # Calcular límites del vector de tiempo final
        time_min = min(time1_reflected[0] + time2[0], time2[0])
        time_max = max(time1_reflected[-1] + time2[-1], time2[-1])
        
        # Crear vector de tiempo final
        time_full = np.arange(time_min, time_max + delta, delta)
        
        # Preparar señales para convolución
        signal1_full = np.zeros(len(time_full))
        signal2_full = np.zeros(len(time_full))
        
        # Llenar las señales en el vector de tiempo completo
        for i, t in enumerate(time_full):
            if time2[0] <= t <= time2[-1]:
                signal2_full[i] = np.interp(t, time2, signal2)
        
        # Realizar convolución
        convolution_result = []
        time_result = []
        
        # Configurar la visualización       
        with st.empty():
            for i in range(0, len(time_full), step):
                # Graficar señales originales
                plt.clf()
                if domain == "continuous":
                    fig,(ax1,ax2,ax3) =plt.subplots(3,1)                  
                    ax1.plot(time1, signal1, 'b-', label='Señal 1')
                    ax1.plot(time2, signal2, 'r-', label='Señal 2')                
                else:
                    fig,(ax1,ax2,ax3) =plt.subplots(3,1) 
                    ax1.stem(time1, signal1, 'b-', label='Señal 1')
                    ax1.stem(time2, signal2, 'r-', label='Señal 2')
                plt.grid(True)
                plt.legend()
                plt.title('Señales Originales')            
                # Graficar proceso de convolución
                signal1_shifted = np.zeros(len(time_full))
                for j, t in enumerate(time_full):
                    t_orig = t - time_full[i]
                    if time1[0] <= t_orig <= time1[-1]:
                        signal1_shifted[j] = np.interp(t_orig, time1, signal1_reflected)
                
                if domain == "continuous":
                    ax2.plot(time_full, signal2_full, 'r-', label='Señal 2')
                    ax2.plot(time_full, signal1_shifted, 'b-', label='Señal 1 (Reflejada y Desplazada)')

                else:
                    ax2.stem(time_full, signal2_full, 'r-', label='Señal 2')
                    ax2.stem(time_full, signal1_shifted, 'b-', label='Señal 1 (Reflejada y Desplazada)')
                plt.grid(True)
                plt.legend()
                plt.title('Proceso de Convolución')
                
                # Calcular punto de convolución
                conv_point = np.sum(signal1_shifted * signal2_full) * delta
                time_result.append(time_full[i])
                convolution_result.append(conv_point)
                
                # Graficar resultado de convolución
                if domain == "continuous":                     
                    ax3.plot(time_result, convolution_result, 'g-', label='Resultado')                
                    st.pyplot(fig,clear_figure=True)
                    
                else:
                    ax3.stem(time_result, convolution_result, 'g-', label='Resultado')
                    st.pyplot(fig,clear_figure=True)
                plt.grid(True)
                plt.legend()
                plt.title('Resultado de la Convolución')
                
                plt.tight_layout()
                plt.pause(0.01)
            
        return time_result, convolution_result


def main():
    st.title("Segundo laboratorio")
    st.subheader(" Juan Polo C    Jesus Carmona   Samir Albor")
    st.error("PRIMER PUNTO")
    opcion=st.selectbox("Elige el domino del tiempo",["Ninguno","Continuo","Discreto"])
    st.write("Ha elegido",opcion)
    img1=Image.open("Continuo.png")
    img2=Image.open("Discreto.png")
    img3=Image.open("Punto2.png")

    if opcion=="Ninguno":
        opcion2=None
        opcion3=None
    if opcion =="Continuo":
        domain="continuous"
        st.image(img1)
        opcion2=st.selectbox("Elige la funcion a trasladar",["Ninguno","grafica[a]","grafica[b]","grafica[c]","grafica[d]"])
        if opcion2=="grafica[a]":
            signal1_type=1
        elif opcion2=="grafica[b]":
            signal1_type=2
        elif opcion2=="grafica[c]":
            signal1_type=3
        elif opcion2=="grafica[d]":
            signal1_type=4
        opcion3=st.selectbox("Elige la funcion fija",["Ninguno","grafica[a]","grafica[b]","grafica[c]","grafica[d]"])
        if opcion3=="grafica[a]":
            signal2_type=1
        elif opcion3=="grafica[b]":
            signal2_type=2
        elif opcion3=="grafica[c]":
            signal2_type=3
        elif opcion3=="grafica[d]":
            signal2_type=4
    if opcion=="Discreto":
        domain="discrete"
        st.image(img2)
        opcion2=st.selectbox("Elige la convolucion discreta",["Ninguno","[a]","[b]"])
        if opcion2=="[a]":
            sig1=1
            sig2=2
        elif opcion2=="[b]":
            sig1=3
            sig2=4
        opcion3=st.selectbox("Elige la funcion a trasladar",["Ninguno","X[n]","H[n]"])
        if opcion3=="X[n]":
            signal1_type=sig1
            signal2_type=sig2
        elif opcion3=="H[n]":
            signal1_type=sig1 + 1
            signal2_type=sig2 - 1
    if opcion!="Ninguno" and opcion2!="Ninguno" and opcion3!="Ninguno":
        convolution = SignalConvolution()
        time1, signal1, limits1 = convolution.generate_signal(signal1_type, domain)
        time2, signal2, limits2 = convolution.generate_signal(signal2_type, domain)
        result_time, result_conv = convolution.perform_convolution(
        time1, signal1, time2, signal2, domain)
    st.error("SEGUNDO PUNTO")
    opcion4=st.selectbox("Elige la convolucion",["Ninguno","[a]","[b]","[c]"])
    st.image(img3)
    t = np.linspace(-10, 20, 1000)
    dt = t[1] - t[0]

    def plot_convolution_results(t, x, h, y, title):
  

        # Graficar señal de entrada x(t)
        fig, ax= plt.subplots()
        ax.plot(t, x )
        plt.grid(True)
        plt.title('Señal de entrada x(t)')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        st.pyplot(fig)
        # Graficar respuesta al impulso h(t)
        fig, ax= plt.subplots()
        ax.plot(t, h)
        plt.grid(True)
        plt.title('Respuesta al impulso h(t)')
        plt.xlabel('t')
        plt.ylabel('h(t)')
        st.pyplot(fig)
        # Graficar resultado de la convolución y(t)
        fig, ax= plt.subplots()        
        ax.plot(t, y)
        plt.grid(True)
        plt.title('Resultado de la convolución y(t)')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        st.pyplot(fig)
    if opcion4=="[a]":
            
        # h(t) = e^(-4t/5)u(t)
        h_a = np.exp(-4*t/5) * (t >= 0)

        # x(t) = e^(-3t/4)[u(t+1) - u(t-5)]
        x_a = np.exp(-3*t/4) * ((t >= -1) & (t <= 5))

        # Calcular la convolución
        y_a = signal.convolve(x_a, h_a) * dt
        t_conv = np.linspace(-10, 20, len(y_a))

        # Recortar para que coincida con el tiempo original
        mid = len(y_a) // 2
        start = mid - len(t) // 2
        y_a = y_a[start:start+len(t)]

        plot_convolution_results(t, x_a, h_a, y_a, 'Caso (a)')
    if opcion4=="[b]":
        # h(t) = e^(-5t/7)u(t+1)
        h_b = np.exp(-5*t/7) * (t >= -1)

        # x(t) = e^t[u(-t) - u(-t-3)] + e^(-t)[u(t) - u(t-3)]
        x_b1 = np.exp(t) * ((t <= 0) & (t >= -3))  # Primera parte
        x_b2 = np.exp(-t) * ((t >= 0) & (t <= 3))  # Segunda parte
        x_b = x_b1 + x_b2

        # Calcular la convolución
        y_b = signal.convolve(x_b, h_b) * dt
        t_conv = np.linspace(-10, 20, len(y_b))

        # Recortar para que coincida con el tiempo original
        mid = len(y_b) // 2
        start = mid - len(t) // 2
        y_b = y_b[start:start+len(t)]

        plot_convolution_results(t, x_b, h_b, y_b, 'Caso (b)')
    if opcion4=="[c]":
        # h(t) = e^t u(1-t)
        h_c = np.exp(t) * (t <= 1)

        # x(t) = u(t+1) - u(t-3)
        x_c = ((t >= -1) & (t <= 3)).astype(float)

        # Calcular la convolución
        y_c = signal.convolve(x_c, h_c) * dt
        t_conv = np.linspace(-10, 20, len(y_c))

        # Recortar para que coincida con el tiempo original
        mid = len(y_c) // 2
        start = mid - len(t) // 2
        y_c = y_c[start:start+len(t)]

        plot_convolution_results(t, x_c, h_c, y_c, 'Caso (c)')
main() 