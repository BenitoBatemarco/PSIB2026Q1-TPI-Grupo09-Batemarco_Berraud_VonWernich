# -*- coding: utf-8 -*-
"""
============================================================================
 Interfaz grafica - TP Retinopatia Diabetica (Grado 0 vs Grado 4)
============================================================================
 GUI de escritorio (Tkinter) para presentar el trabajo de forma estetica.
 Reutiliza el pipeline de 'retinopatia_diabetica.py'.
"""

import os
import numpy as np

# Pipeline del TP (define procesar, listar_imagenes, umbral_textura, etc.)
import retinopatia_diabetica as R


def cargar_dataset(progreso=None):
    """
    Procesa todas las imagenes una vez y cachea sus features escalares
    (incluye la textura: entropia local y homogeneidad GLCM).
    
    """
    items = R.listar_imagenes()
    n = len(items)
    data = []
    for i, (path, grado) in enumerate(items):
        if progreso:
            progreso(i, n, os.path.basename(path))
        f = R.procesar(path, con_textura=True)
        data.append(dict(
            path=path, nombre=os.path.basename(path), grado=grado,
            f_exud=f["f_exud"] * 100, n_exud=f["n_exud"],
            f_hemo=f["f_hemo"] * 100, n_hemo=f["n_hemo"],
            ent_local=f.get("ent_local", float("nan")),
            glcm_homog=f.get("glcm_homog", float("nan")),
        ))
    return data


def umbral_textura(data):
    """Umbral de entropia local por Otsu (no supervisado) sobre el dataset."""
    return R.umbral_textura([d["ent_local"] for d in data])


def predecir(ent_local, umbral):
    """Criterio de textura: Grado 4 si la entropia local <= umbral."""
    return 4 if ent_local <= umbral else 0


def comparar_criterios(data):
    """
    Compara, de forma NO supervisada (umbral por Otsu y direccion fijada por
    conocimiento del dominio), el criterio de lesiones vs el de texturas.
    """
    ys = np.array([d["grado"] for d in data])

    # criterio LESIONES: mas area de lesiones (exudados+hemorragias) => Grado 4
    les = np.array([d["f_exud"] + d["f_hemo"] for d in data], float)
    pred_les = np.where(les >= R.umbral_textura(les), 4, 0)
    les_full = (pred_les == ys).mean()

    # criterio TEXTURAS: menos entropia local (textura mas lisa) => Grado 4
    ent = np.array([d["ent_local"] for d in data], float)
    pred_tex = np.where(ent <= R.umbral_textura(ent), 4, 0)
    tex_full = (pred_tex == ys).mean()

    baseline = max((ys == 0).mean(), (ys == 4).mean())
    return dict(les_full=les_full, tex_full=tex_full, baseline=baseline)


# ===========================================================================
#  INTERFAZ GRAFICA (Tkinter)
# ===========================================================================
def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # ---- paleta ----
    COL_BG = "#0f1620"; COL_PANEL = "#18222e"; COL_TXT = "#e6edf3"
    COL_G0 = "#2ecc71"; COL_G4 = "#e74c3c"; COL_ACC = "#3498db"
    COL_OK = "#2ecc71"; COL_ERR = "#e74c3c"

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Diagnostico de Retinopatia Diabetica - TP Procesamiento de Imagenes")
            self.geometry("1280x820")
            self.configure(bg=COL_BG)

            # estado (el procesamiento con texturas tarda unos segundos)
            def _prog(i, n, nombre):
                print(f"  Procesando dataset {i+1}/{n}  {nombre}", end="\r", flush=True)
            print("Cargando y procesando imagenes")
            self.data = cargar_dataset(progreso=_prog)
            print("\nListo." )
            self.comp = comparar_criterios(self.data)
            self.umbral_tex = umbral_textura(self.data)   # Otsu (no supervisado)
            self.idx = 0
            self.imagen_externa = None  # (feats, etapas, nombre) si se cargo una

            self._estilo()
            self._header()
            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.tab_img = ttk.Frame(nb); nb.add(self.tab_img, text="  Analisis por imagen  ")
            self.tab_cmp = ttk.Frame(nb); nb.add(self.tab_cmp, text="  Comparacion de criterios  ")
            self.tab_teo = ttk.Frame(nb); nb.add(self.tab_teo, text="  Metodologia  ")
            self._build_img(); self._build_cmp(); self._build_teo()
            self._refresh_img()

        # ---------- estilo ----------
        def _estilo(self):
            s = ttk.Style(self)
            try: s.theme_use("clam")
            except Exception: pass
            s.configure("TFrame", background=COL_BG)
            s.configure("Panel.TFrame", background=COL_PANEL)
            s.configure("TLabel", background=COL_BG, foreground=COL_TXT, font=("Segoe UI", 10))
            s.configure("Panel.TLabel", background=COL_PANEL, foreground=COL_TXT, font=("Segoe UI", 10))
            s.configure("H.TLabel", background=COL_BG, foreground=COL_TXT, font=("Segoe UI", 16, "bold"))
            s.configure("Big.TLabel", background=COL_PANEL, foreground=COL_TXT, font=("Segoe UI", 13, "bold"))
            s.configure("TButton", font=("Segoe UI", 10), padding=6)
            s.configure("TNotebook", background=COL_BG, borderwidth=0)
            s.configure("TNotebook.Tab", background=COL_PANEL, foreground=COL_TXT, padding=(14, 8))
            s.map("TNotebook.Tab", background=[("selected", COL_ACC)])

        def _header(self):
            h = ttk.Frame(self); h.pack(fill="x", padx=10, pady=10)
            ttk.Label(h, text="Retinopatia Diabetica  -  Clasificador Grado 0 / Grado 4",
                      style="H.TLabel").pack(side="left")
            ttk.Label(h, text="Procesamiento de imagenes",
                      style="TLabel").pack(side="right")

        def _fig_canvas(self, parent, figsize):
            fig = Figure(figsize=figsize, facecolor=COL_PANEL)
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.get_tk_widget().configure(bg=COL_PANEL, highlightthickness=0)
            return fig, canvas

        # ====================================================================
        #  TAB 1 - ANALISIS POR IMAGEN
        # ====================================================================
        def _build_img(self):
            left = ttk.Frame(self.tab_img, style="Panel.TFrame")
            left.pack(side="left", fill="y", padx=8, pady=8)
            ttk.Label(left, text="Imagenes del dataset", style="Big.TLabel").pack(padx=10, pady=(10, 4))

            self.lst = tk.Listbox(left, width=26, height=26, bg=COL_PANEL, fg=COL_TXT,
                                  selectbackground=COL_ACC, highlightthickness=0,
                                  borderwidth=0, font=("Consolas", 10), activestyle="none")
            for d in self.data:
                self.lst.insert("end", f"  G{d['grado']}   {d['nombre']}")
            self.lst.pack(padx=10, pady=4, fill="y", expand=True)
            self.lst.bind("<<ListboxSelect>>", self._on_select)
            self.lst.selection_set(0)

            nav = ttk.Frame(left, style="Panel.TFrame"); nav.pack(pady=8)
            ttk.Button(nav, text="◀  Anterior", command=self._prev).grid(row=0, column=0, padx=3)
            ttk.Button(nav, text="Siguiente  ▶", command=self._next).grid(row=0, column=1, padx=3)
            ttk.Button(left, text="Cargar imagen externa...", command=self._cargar).pack(pady=(2, 12))

            right = ttk.Frame(self.tab_img, style="Panel.TFrame")
            right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
            self.fig_img, self.canvas_img = self._fig_canvas(right, (9.2, 6.6))
            self.canvas_img.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

            self.diag = ttk.Label(right, text="", style="Big.TLabel", anchor="center")
            self.diag.pack(fill="x", padx=8, pady=(0, 8))

        def _on_select(self, _evt=None):
            sel = self.lst.curselection()
            if sel:
                self.idx = sel[0]; self.imagen_externa = None; self._refresh_img()

        def _prev(self):
            self.idx = (self.idx - 1) % len(self.data)
            self.lst.selection_clear(0, "end"); self.lst.selection_set(self.idx)
            self.imagen_externa = None; self._refresh_img()

        def _next(self):
            self.idx = (self.idx + 1) % len(self.data)
            self.lst.selection_clear(0, "end"); self.lst.selection_set(self.idx)
            self.imagen_externa = None; self._refresh_img()

        def _cargar(self):
            path = filedialog.askopenfilename(
                title="Elegir imagen de fondo de ojo",
                filetypes=[("Imagenes", "*.jpeg *.jpg *.png *.tif *.tiff"), ("Todos", "*.*")])
            if not path:
                return
            try:
                feats, etapas = R.procesar(path, devolver_etapas=True, con_textura=True)
                self.imagen_externa = (feats, etapas, os.path.basename(path))
                self._refresh_img()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo procesar:\n{e}")

        def _dibujar_pipeline(self, fig, feats, etapas):
            pred = predecir(feats["ent_local"], self.umbral_tex)
            overlay = R.overlay_lesiones(etapas["img"], etapas["exud"], etapas["hemo"])
            paneles = [
                (etapas["img"], "1. Original (RGB)", None),
                (etapas["fov"], "2. Campo de vision", "gray"),
                (etapas["canal"], f"3. Canal mayor contraste ({feats['canal']})", "gray"),
                (etapas["suave"], "4. Filtro Gaussiano", "gray"),
                (etapas["realzada"], "5. Realce CLAHE", "gray"),
                (etapas["vasos"], "6. Vasos (Black-Top-Hat)", "gray"),
                (etapas["exud"], f"7. Exudados (n={feats['n_exud']})", "gray"),
                (etapas["mapa_ent"], "8. Entropia local (textura)", "magma"),
                (overlay, "9. Lesiones detectadas", None),
            ]
            fig.clear()
            for k, (im, tit, cmap) in enumerate(paneles):
                ax = fig.add_subplot(3, 3, k + 1)
                ax.imshow(im, cmap=cmap); ax.set_title(tit, color=COL_TXT, fontsize=9)
                ax.axis("off")
            fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.18)
            return pred

        def _refresh_img(self):
            if self.imagen_externa is not None:
                feats, etapas, nombre = self.imagen_externa
                grado_real = None
            else:
                d = self.data[self.idx]
                feats, etapas = R.procesar(d["path"], devolver_etapas=True, con_textura=True)
                nombre = d["nombre"]; grado_real = d["grado"]
            pred = self._dibujar_pipeline(self.fig_img, feats, etapas)
            self.canvas_img.draw()
            txt = (f"{nombre}      entropia local = {feats['ent_local']:.3f} "
                   f"(umbral Otsu <= {self.umbral_tex:.3f})      "
                   f"homogeneidad GLCM = {feats['glcm_homog']:.3f}\n")
            if grado_real is None:
                txt += f"Prediccion: GRADO {pred}   (imagen externa, sin etiqueta real)"
                self.diag.configure(text=txt, foreground=COL_ACC)
            else:
                ok = (pred == grado_real)
                txt += (f"Real: GRADO {grado_real}      Prediccion: GRADO {pred}      "
                        f"{'CORRECTO' if ok else 'ERROR'}")
                self.diag.configure(text=txt, foreground=COL_OK if ok else COL_ERR)

        # ====================================================================
        #  TAB 3 - COMPARACION DE CRITERIOS
        # ====================================================================
        def _build_cmp(self):
            top = ttk.Frame(self.tab_cmp, style="Panel.TFrame")
            top.pack(fill="both", expand=True, padx=8, pady=8)
            fig, canvas = self._fig_canvas(top, (9.5, 5.2))
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
            c = self.comp
            ax = fig.add_subplot(111, facecolor="#101a26")
            etiquetas = ["Criterio lesiones\n(exudados/hemorragias)", "Criterio de texturas\n(entropia local)"]
            vals = [c["les_full"]*100, c["tex_full"]*100]
            cols = ["#8e9aa6", COL_ACC]
            bars = ax.bar(etiquetas, vals, color=cols, edgecolor="white", linewidth=0.6, width=0.55)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center",
                        color=COL_TXT, fontweight="bold")
            ax.set_ylim(0, 100); ax.set_ylabel("Accuracy (%)", color=COL_TXT)
            ax.set_title("Criterio de lesiones  vs  criterio de texturas (ambos con umbral Otsu, no supervisado)",
                         color=COL_TXT, fontsize=11)
            ax.tick_params(colors=COL_TXT)
            for sp in ax.spines.values(): sp.set_color("#33414f")
            fig.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.12)
            canvas.draw()

            txt = (
                f"Ambos criterios se evaluan de forma no supervisada: el umbral sale de la distribucion de "
                f"la feature por el metodo de Otsu (sin usar las etiquetas) y la direccion se fija por "
                f"conocimiento del dominio.\n\n"
                f"Clasificar por area de lesiones (exudados + hemorragias) rinde {c['les_full']*100:.1f}% "
                f"(la clase mayoritaria ya da {c['baseline']*100:.0f}%): muchas Grado 4 son de bajo "
                f"contraste, por lo que sus lesiones se segmentan peor que en una retina Grado 0 nitida.\n\n"
                f"El criterio de texturas (entropia local) alcanza {c['tex_full']*100:.1f}%: una retina sana "
                f"nitida tiene una textura mas irregular (mayor entropia) que una Grado 4 difusa."
            )
            lbl = ttk.Label(self.tab_cmp, text=txt, style="Panel.TLabel",
                            wraplength=1180, justify="left", font=("Segoe UI", 10))
            lbl.pack(fill="x", padx=14, pady=(0, 12))

        # ====================================================================
        #  TAB 4 - METODOLOGIA
        # ====================================================================
        def _build_teo(self):
            txt = (
                "PIPELINE DE PROCESAMIENTO\n"
                "\n"
                "1) Contextualizacion\n"
                "   - Carga de la imagen RGB y mascara del campo de vision (umbral + morfologia).\n"
                "\n"
                "2) Pre-procesamiento\n"
                "   - Seleccion del canal RGB de mayor contraste (desvio estandar de intensidades).\n"
                "   - Filtro pasa-bajos Gaussiano: atenua el ruido de alta frecuencia (convolucion).\n"
                "   - Realce CLAHE: ecualizacion de histograma adaptativa y limitada en contraste.\n"
                "\n"
                "3) Segmentacion y extraccion de caracteristicas (morfologia matematica)\n"
                "   - Vasos: Black-Top-Hat (resalta estructuras oscuras finas) + umbral de Otsu.\n"
                "   - Disco optico: deteccion del maximo brillo, se excluye de los exudados.\n"
                "   - Exudados brillantes: White-Top-Hat + umbral absoluto.\n"
                "   - Hemorragias: Black-Top-Hat + umbral + componentes conexos filtrados por forma.\n"
                "   - Texturas: entropia local y homogeneidad GLCM (matriz de co-ocurrencia).\n"
                "\n"
                "4) Diagnostico\n"
                "   - Criterio de TEXTURA: GRADO 4 si la entropia local <= umbral.\n"
                "   - El umbral se obtiene por Otsu sobre la distribucion de entropias igual que el umbral de los vasos\n"
                "NOTA: las lesiones (exudados/vasos) se segmentan y se visualizan. El\n"
                "diagnostico final usa la textura: una retina sana nitida es mas irregular (mayor\n"
                "entropia) que una Grado 4 difusa (ver'Comparacion de criterios')."
            )
            ttk.Label(self.tab_teo, text=txt, style="Panel.TLabel",
                      justify="left", font=("Consolas", 11)).pack(
                      fill="both", expand=True, padx=18, pady=18, anchor="nw")

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
