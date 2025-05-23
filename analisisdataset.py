import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
from tkinter.scrolledtext import ScrolledText
import numpy as np
import json

class DatasetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis DataSet")

        self.df = None
        self.targets = None
        self.fdr_results = {}
        self.cross_correlation = None
        self.pearson = None

        # Frame para los botones
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.open_button = tk.Button(button_frame, text="Abrir CSV", command=self.load_csv)
        self.open_button.grid(row=0, column=0, padx=5)

        self.drop_row_button = tk.Button(button_frame, text="Eliminar Fila ➡️", command=self.drop_row)
        self.drop_row_button.grid(row=0, column=2, padx=5)

        self.drop_column_button = tk.Button(button_frame, text="Eliminar Columna ⬇️", command=self.drop_column)
        self.drop_column_button.grid(row=0, column=3, padx=5)

        self.target_button = tk.Button(button_frame, text="Seleccionar Target", command=self.select_target)
        self.target_button.grid(row=0, column=1, padx=5)

        self.fdr_button = tk.Button(button_frame, text="Calcular FDR", command=self.compute_fdr)
        self.fdr_button.grid(row=1, column=0, padx=5)

        self.pearson_button = tk.Button(button_frame, text="Calcular Coeficiente de Pearson", command=self.compute_pearson_coef)
        self.pearson_button.grid(row=1, column=1, padx=5)

        self.cross_button = tk.Button(button_frame, text="Calcular Correlación Cruzada", command=self.compute_cross_correlation)
        self.cross_button.grid(row=1, column=2, padx=5)

        # Frame para visualización de datos
        text_frame = tk.Frame(root)
        text_frame.pack(pady=10)

        # Dataset
        dataset_frame = tk.LabelFrame(text_frame, text="Dataset (Características)")
        dataset_frame.grid(row=0, column=0, padx=10)
        self.text = ScrolledText(dataset_frame, width=100, height=10)
        self.text.pack()

        # Targets
        target_frame = tk.LabelFrame(text_frame, text="Targets Seleccionados")
        target_frame.grid(row=1, column=0, padx=10)
        self.text_targets = ScrolledText(target_frame, width=40, height=10)
        self.text_targets.pack()

        # FDR Results
        fdr_frame = tk.LabelFrame(text_frame, text="Resultados del FDR")
        fdr_frame.grid(row=1, column=1, padx=10)
        self.text_fdr_results = ScrolledText(fdr_frame, width=40, height=10)
        self.text_fdr_results.pack()

        # Pearson Results
        pearson_frame = tk.LabelFrame(text_frame, text="Resultados de Pearson")
        pearson_frame.grid(row=2, column=0, padx=10)
        self.text_pearson_results = ScrolledText(pearson_frame, width=40, height = 10)
        self.text_pearson_results.pack()

        #Cross Correlation Results
        cross_frame = tk.LabelFrame(text_frame, text="Resultados de Correlación Cruzada")
        cross_frame.grid(row=2,column=1,padx=10)
        self.text_cross_results = ScrolledText(cross_frame, width=40, height=10)
        self.text_cross_results.pack()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            dialog = CSVOptionsDialog(self.root)
            if dialog.result:
                sep, has_header = dialog.result
                try:
                    header = 0 if has_header else None
                    self.resetAll()
                    self.df = pd.read_csv(file_path, sep=sep, header=header)
                    self.display_dataframe()
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo abrir el archivo: {e}")

    def select_target(self):
        if self.df is None:
            messagebox.showwarning("Advertencia", "No hay ningún DataFrame cargado.")
            return

        try:
            column_input = simpledialog.askstring("Seleccionar target", "Nombre o índice de la columna target:")

            if column_input is None or column_input.strip() == "":
                return  # El usuario canceló

            # Intentar convertir a entero
            try:
                column_key = int(column_input)
            except ValueError:
                column_key = column_input.strip()

            if column_key not in self.df.columns:
                messagebox.showerror("Error", f"La columna '{column_key}' no existe en el DataFrame.")
                return

            self.targets = self.df[column_key]
            self.df.drop(column_key, axis=1, inplace=True)
            self.display_dataframe()

        except Exception as e:
            messagebox.showerror("Error", f"Error al seleccionar el target: {e}")

    def display_dataframe(self):
        if self.df is not None:
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, self.df.to_string())
        if self.targets is not None:
            self.text_targets.delete("1.0", tk.END)
            self.text_targets.insert(tk.END, self.targets.to_string())
        else:
            self.text_targets.delete("1.0", tk.END)
            self.text_fdr_results.delete("1.0", tk.END)
            self.text_cross_results.delete("1.0", tk.END)
            self.text_pearson_results.delete("1.0", tk.END)
        if self.fdr_results:
            self.text_fdr_results.delete("1.0", tk.END)
            df_fdr = pd.DataFrame(self.fdr_results, columns=["Característica", "FDR"])
            df_fdr["FDR"] = df_fdr["FDR"].round(4)
            self.text_fdr_results.insert(tk.END, df_fdr.to_string())
        if self.pearson:
            self.text_pearson_results.delete("1.0", tk.END)
            df_pearson = pd.DataFrame(self.pearson, columns=["Característica", "Pearson"])
            df_pearson["Pearson"] = df_pearson["Pearson"].round(4)
            self.text_pearson_results.insert(tk.END, df_pearson.drop_duplicates().to_string())
        else:
            self.text_pearson_results.delete("1.0", tk.END)
        if self.cross_correlation is not None:
            self.text_cross_results.delete("1.0", tk.END)
            df_cross = pd.DataFrame({
                "Característica": self.df.columns,
                "Correlación Cruzada": np.round(self.cross_correlation, 4)
            })
            self.text_cross_results.insert(tk.END, df_cross.to_string(index=False))
        else:
            self.text_cross_results.delete("1.0", tk.END)

    def drop_row(self):
        if self.df is None:
            return
        try:
            index = simpledialog.askinteger("Eliminar Fila", "Índice de la fila a eliminar:")
            if index is not None or index.strip() != "":
                self.df.drop(index, axis=0, inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                if self.targets is not None:
                    self.targets.drop(index, axis=0, inplace=True)
                    self.targets.reset_index(drop=True, inplace=True)
                self.display_dataframe()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar la fila: {e}")

    def drop_column(self):
        if self.df is None:
            return
        try:
            column_input = simpledialog.askstring("Seleccionar target", "Nombre o índice de la columna target:")

            if column_input is None or column_input.strip() == "":
                return  # El usuario canceló

            # Intentar convertir a entero
            try:
                column_key = int(column_input)
            except ValueError:
                column_key = column_input.strip()

            if column_key not in self.df.columns:
                messagebox.showerror("Error", f"La columna '{column_key}' no existe en el DataFrame.")
                return
            
            self.df.drop(column_key, axis=1, inplace=True)
            self.display_dataframe()

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar la columna: {e}")

    def compute_fdr(self):
        if self.df is None or self.targets is None:
            messagebox.showwarning("Advertencia", "Dataset o targets no inicializados.")
            return
        try: 
            classes = np.unique(self.targets)
            n_classes = len(classes)
            n_features = self.df.shape[1]
            fdr_values = np.zeros(n_features)

            for f in range(n_features):
                fdr_f = 0.0
                means = []
                variances = []
                for c in classes:
                    X_c = self.df.loc[self.targets == c].iloc[:, f]
                    means.append(np.mean(X_c))
                    variances.append(np.var(X_c) + 1e-8)

                for i in range(n_classes):
                    for j in range(n_classes):
                        if i != j:
                            numerator = (means[i] - means[j]) ** 2
                            denominator = variances[i] + variances[j]
                            fdr_f += numerator / denominator

                fdr_values[f] = fdr_f

            feature_names = self.df.columns.tolist()
            self.fdr_results = sorted(zip(feature_names, fdr_values), key=lambda x: x[1], reverse=True)
            self.display_dataframe()
 
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular FDR: {e}")            

    def compute_cross_correlation(self):
        if self.df is None:
            messagebox.showwarning("Advertencia", "DataSet no inicializado")
            return
        try:
            corr = np.abs(np.corrcoef(self.df,rowvar=False))
            np.fill_diagonal(corr,0)
            self.cross_correlation = np.sum(corr,axis=0)
            self.display_dataframe()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo calcular la correlación cruzada: {e}")

    def compute_pearson_coef(self):
        if self.df is None:
            messagebox.showwarning("Advertencia", "DataSet no inicializado")
            return
        try:
            coef = simpledialog.askfloat("Coeficiente de Pearson", "Ingrese el coeficiente de Pearson")
            if coef is None:
                return

            corr = np.abs(np.corrcoef(self.df, rowvar=False))
            columns = self.df.columns
            selected_pairs = []

            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[1]):
                    if corr[i, j] >= coef:
                        selected_pairs.append((columns[i], corr[i, j]))
                        selected_pairs.append((columns[j], corr[i, j]))

            # Eliminar duplicados manteniendo el mayor coeficiente encontrado por característica
            pearson_dict = {}
            for name, value in selected_pairs:
                if name not in pearson_dict or pearson_dict[name] < value:
                    pearson_dict[name] = value

            # Convertir a lista ordenada
            self.pearson = sorted(pearson_dict.items(), key=lambda x: x[1], reverse=True)

            self.display_dataframe()

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo calcular Pearson: {e}")

    def resetAll(self):
        self.df = None
        self.targets = None
        self.fdr_results = {}
        self.pearson = None
        self.cross_correlation = None
        return

class CSVOptionsDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Opciones de carga CSV")
        self.result = None

        tk.Label(self.top, text="Separador:").pack(padx=10, pady=(10, 0))
        self.sep_entry = tk.Entry(self.top)
        self.sep_entry.insert(0, ",")
        self.sep_entry.pack(padx=10, pady=5)

        self.has_header_var = tk.BooleanVar(value=True)
        self.header_check = tk.Checkbutton(self.top, text="El archivo tiene encabezado", variable=self.has_header_var)
        self.header_check.pack(pady=5)

        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Aceptar", command=self.on_accept).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancelar", command=self.top.destroy).pack(side=tk.LEFT, padx=5)

        self.top.grab_set()
        parent.wait_window(self.top)

    def on_accept(self):
        sep = self.sep_entry.get()
        has_header = self.has_header_var.get()
        self.result = (sep, has_header)
        self.top.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetApp(root)
    root.mainloop()
