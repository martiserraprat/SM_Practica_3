import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import dct, idct
import numpy as np
import time
import metrikz
import eines_sessio3

# ─────────────────────────────────────────────
# Matriu de quantització estàndard JPEG/MPEG
# ─────────────────────────────────────────────
quantization_matrix = np.array([
    [16., 11., 10., 16., 24., 40., 51., 61.],
    [12., 12., 14., 19., 26., 58., 60., 55.],
    [14., 13., 16., 24., 40., 57., 69., 56.],
    [14., 17., 22., 29., 51., 87., 80., 62.],
    [18., 22., 37., 56., 68., 109., 103., 77.],
    [24., 35., 55., 64., 81., 104., 113., 92.],
    [49., 64., 78., 87., 103., 121., 120., 101.],
    [72., 92., 95., 98., 112., 100., 103., 99.]
])

BLOCK_SIZE = 8  # mida del bloc en píxels


# ─────────────────────────────────────────────
# Funcions auxiliars DCT i quantització
# ─────────────────────────────────────────────
def dct2(block):
    """DCT 2D d'un bloc 8x8."""
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(block):
    """Inversa DCT 2D d'un bloc 8x8."""
    return np.round(idct(idct(block, axis=1, norm='ortho'), axis=0, norm='ortho'))


def quantit(block):
    """Quantització d'un bloc DCT."""
    return np.round(block / quantization_matrix)


def iquantit(block):
    """Quantització inversa."""
    return block * quantization_matrix


def calcular_mse_bloc(bloc1, bloc2):
    """MSE entre dos blocs 8x8."""
    return np.mean((bloc1.astype(float) - bloc2.astype(float)) ** 2)


# ─────────────────────────────────────────────
# Algoritme de Block Matching
# ─────────────────────────────────────────────
def block_matching(frame1, frame2, block_size=8, search_mode='full', search_range=12):
    """
    Algorisme de block matching entre dos frames.

    Paràmetres
    ----------
    frame1      : frame anterior (referència), en grisos
    frame2      : frame actual, en grisos
    block_size  : mida del bloc (per defecte 8x8)
    search_mode : 'full'      → V1: cerca en tota la imatge
                  'restricted'→ V2: cerca en una regió limitada (search_range)
    search_range: radi de cerca per al mode restringit (en píxels)

    Retorna
    -------
    actual_position  : llista de (row, col) de cada bloc al frame2
    motion_vector    : llista de (dy, dx) per a cada bloc
    errors_prediction: llista de vectors zigzag dels errors quantitzats
    """

    h, w = frame2.shape
    actual_position = []
    motion_vector = []
    errors_prediction = []

    # Recorrem tots els blocs del frame actual (frame2)
    for row in range(0, h - block_size + 1, block_size):
        for col in range(0, w - block_size + 1, block_size):

            # Bloc actual del frame2
            bloc_actual = frame2[row:row + block_size, col:col + block_size].astype(float)

            best_mse = float('inf')
            best_pos = (row, col)  # per defecte, mateixa posició

            # ── V1: cerca en tota la imatge ──────────────────────────────
            if search_mode == 'full':
                for r in range(0, h - block_size + 1, block_size):
                    for c in range(0, w - block_size + 1, block_size):
                        bloc_ref = frame1[r:r + block_size, c:c + block_size].astype(float)
                        mse_val = calcular_mse_bloc(bloc_actual, bloc_ref)
                        if mse_val < best_mse:
                            best_mse = mse_val
                            best_pos = (r, c)

            # ── V2: cerca en una regió restringida ───────────────────────
            elif search_mode == 'restricted':
                r_min = max(0, row - search_range)
                r_max = min(h - block_size, row + search_range)
                c_min = max(0, col - search_range)
                c_max = min(w - block_size, col + search_range)

                for r in range(r_min, r_max + 1):
                    for c in range(c_min, c_max + 1):
                        bloc_ref = frame1[r:r + block_size, c:c + block_size].astype(float)
                        mse_val = calcular_mse_bloc(bloc_actual, bloc_ref)
                        if mse_val < best_mse:
                            best_mse = mse_val
                            best_pos = (r, c)

            # ── Guardem resultats ────────────────────────────────────────
            actual_position.append((row, col))

            dy = best_pos[0] - row
            dx = best_pos[1] - col
            motion_vector.append((dy, dx))

            # Error de predicció = bloc_actual - bloc_més_semblant
            bloc_best = frame1[best_pos[0]:best_pos[0] + block_size,
                                best_pos[1]:best_pos[1] + block_size].astype(float)
            error_matrix = bloc_actual - bloc_best

            # DCT + Quantització de l'error
            error_dct = dct2(error_matrix)
            error_quant = quantit(error_dct)

            # Zigzag del error quantitzat
            error_zigzag = eines_sessio3.zigzag(error_quant)
            errors_prediction.append(error_zigzag)

    return actual_position, motion_vector, errors_prediction


# ─────────────────────────────────────────────
# Visualització dels vectors de moviment
# ─────────────────────────────────────────────
def visualitzar_vectors(frame2, actual_position, motion_vector, titol="Vectors de moviment", block_size=8):
    """
    Dibuixa els vectors de moviment sobre el frame2.
    Línies vermelles des de la posició actual fins a la posició del bloc de referència.
    """
    img_color = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    for i, ((row, col), (dy, dx)) in enumerate(zip(actual_position, motion_vector)):
        if dy != 0 or dx != 0:
            # Centre del bloc actual
            cx_actual = col + block_size // 2
            cy_actual = row + block_size // 2
            # Centre del bloc de referència (on apunta el motion vector)
            cx_ref = cx_actual + dx
            cy_ref = cy_actual + dy
            cv2.line(img_color,
                     (cx_actual, cy_actual),
                     (cx_ref, cy_ref),
                     (0, 0, 255), 1)

    # Convertim BGR→RGB per matplotlib
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(titol)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{titol.replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"  → Imatge guardada com '{titol.replace(' ', '_')}.png'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # Parells d'imatges disponibles
    parells = [
        ("frame0_1.png", "frame0_2.png"),
        ("frame1_1.png", "frame1_2.png"),
        ("frame2_1.png", "frame2_2.png"),
    ]

    print("=" * 65)
    print("  BLOCK MATCHING - Pràctica 3 Sistemes Multimèdia")
    print("=" * 65)

    resultats = []  # per a la taula resum

    for nom1, nom2 in parells:
        img1 = cv2.imread(nom1)
        img2 = cv2.imread(nom2)

        if img1 is None or img2 is None:
            print(f"\n[!] No s'han trobat {nom1} o {nom2}, saltem aquest parell.")
            continue

        frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        print(f"\nParell: {nom1}  ←→  {nom2}  |  Dimensions: {frame1.shape}")

        for mode, label, s_range in [('full', 'V1 (cerca completa)', None),
                                      ('restricted', 'V2 (cerca restringida ±12px)', 12)]:

            kwargs = {'search_mode': mode}
            if s_range:
                kwargs['search_range'] = s_range

            t0 = time.time()
            actual_pos, mov_vec, errors_pred = block_matching(frame1, frame2, **kwargs)
            t1 = time.time()
            elapsed = t1 - t0

            # MSE global: reconstruïm el frame predito i calculem MSE
            h, w = frame2.shape
            frame_predito = np.zeros_like(frame2, dtype=float)
            for i, ((row, col), (dy, dx)) in enumerate(zip(actual_pos, mov_vec)):
                r_ref = row + dy
                c_ref = col + dx
                r_ref = np.clip(r_ref, 0, h - BLOCK_SIZE)
                c_ref = np.clip(c_ref, 0, w - BLOCK_SIZE)
                frame_predito[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE] = \
                    frame1[r_ref:r_ref + BLOCK_SIZE, c_ref:c_ref + BLOCK_SIZE]

            mse_global = metrikz.mse(frame2.astype(float), frame_predito)

            print(f"\n  [{label}]")
            print(f"    Temps d'execució : {elapsed:.2f} s")
            print(f"    MSE global       : {mse_global:.4f}")
            print(f"    Blocs processats : {len(actual_pos)}")

            resultats.append({
                'parell': f"{nom1}/{nom2}",
                'versio': label,
                'mse': mse_global,
                'temps': elapsed
            })

            # Guardem vectors per a la pròxima sessió
            prefix = f"{nom1.split('_')[0]}_{nom1.split('_')[1].split('.')[0]}_{mode}"
            np.save(f"actual_position_{prefix}.npy", np.array(actual_pos))
            np.save(f"motion_vector_{prefix}.npy", np.array(mov_vec))
            np.save(f"errors_prediction_{prefix}.npy", np.array(errors_pred, dtype=object))
            print(f"    Vectors guardats com '{prefix}_*.npy'")

            # Visualització
            titol = f"Vectors de moviment - {nom1.split('.')[0]} ({label})"
            visualitzar_vectors(frame2, actual_pos, mov_vec, titol=titol)

    # ─── Taula resum ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TAULA RESUM")
    print("=" * 65)
    print(f"{'Parell':<25} {'Versió':<30} {'MSE':>10} {'Temps(s)':>10}")
    print("-" * 65)
    for r in resultats:
        print(f"{r['parell']:<25} {r['versio']:<30} {r['mse']:>10.4f} {r['temps']:>10.2f}")
    print("=" * 65)