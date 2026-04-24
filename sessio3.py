import cv2
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import numpy as np
import metrikz
import eines_sessio3

quantization_matrix =  [[16.,11.,10.,16.,24.,40.,51.,61.],
                        [12.,12.,14.,19.,26.,58.,60.,55.],
                        [14.,13.,16.,24.,40.,57.,69.,56.],
                        [14.,17.,22.,29.,51.,87.,80.,62.],
                        [18.,22.,37.,56.,68.,109.,103.,77.],
                        [24.,35.,55.,64.,81.,104.,113.,92.],
                        [49.,64.,78.,87.,103.,121.,120.,101.],
                        [72.,92.,95.,98.,112.,100.,103.,99.]]


#transformada DCT de matrius
def dct2(block):
    return dct(dct(block, axis=0, norm = 'ortho'), axis=1, norm = 'ortho'); 


#inversa de la transformada DCT de matrius
def idct2(block):
    return np.round(idct(idct(block, axis=1, norm = 'ortho'), axis=0, norm = 'ortho'));

    
#funcio de quantitzacio
def quantit(block):
    quant = np.round(block / quantization_matrix);
    return quant


# funcio inversa de de quantitzacio
def iquantit(block):
    inverse = (block * quantization_matrix);
    return inverse





if __name__ == '__main__':

    #frame anterior
    img1 = cv2.imread("frame0_1.png")

    #frame actual
    img2 = cv2.imread("frame0_2.png")

    # convertim a gris per simplicitat
    frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #mostra les dimensions de la imatge
    print("dimensions de la imatge= ")
    print(frame1.shape)
    dim=frame1.shape

    #mostra els dos frames per separat en mida real, es pot canviar per matplotlib per veure més gran.
    cv2.imshow('frame 1',frame1)
    cv2.imshow('frame 2', frame2)


    #vectors finals
    actual_position=[]
    motion_vector=[]
    errors_prediction=[]

    # matriu per generar els blocs 
    bk=np.zeros((8, 8))


    # GENERAR AQUI EL CODI PER FER EL MOTION VECTORS
    # ######################
    # afegir al vector actual_positions totes les coordenades dels blocks de les imatges.
    # afegir al vector motion_vector les coordenades el block que menor error te del frame anterior.
    # afegir al vector errors_prediccio l'error generat.
    # quantitzar i aplicar DCT a la matriu d'errors (func_quantized).
    # crear el vector final amb l'algoritme zigzag dels error.


    # GENERAR PER ULTIM EL CODI DE VISUALITZACIO 
    # ######################
    # sobre la imatge del frame 2, crear una línia entre cada posició dels elements dels 
    # vectors actual_position i motion_vector que siguin diferents. ex. línia origen (3,2) a (14,9)
    # podeu fer servir  el seguent codi on podeu posar origen i final de la linea, 
    # en el cas que siguin diferents.
    #
    # podeu fer servir cv2.imshow o matplotlib amb imshow 
    # aquí teniu un exemple per una única línea
    

    #### Fent servir CV2 #########
    # Crear una còpia en color per dibuixar la línia
    img_gris_color = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

    # Dibuixa una línia vermella entre les posicions (3,2) i (14,9)
    cv2.line(img_gris_color, (3, 2), (14, 9), (0, 0, 255), 1)

    # Mostra la imatge amb cv2.imshow amb les línies vermelles
    cv2.imshow('Imatge amb moviments marcats', img_gris_color)


    #### Fent servir Matplotlib #########
    # Crear una còpia en color per dibuixar la línia
    img_gris_color = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

    # Convertir la imatge de BGR a RGB per visualitzar-la correctament amb Matplotlib
    img_gris_color_rgb = cv2.cvtColor(img_gris_color, cv2.COLOR_BGR2RGB)

    # Dibuixa una línia vermella entre les posicions (3,2) i (14,9)
    cv2.line(img_gris_color_rgb, (3, 2), (14, 9), (255, 0, 0), 1)

    plt.imshow(img_gris_color_rgb)
    plt.title("Imatge amb moviments marcats")
    plt.axis('off')
    plt.show()
