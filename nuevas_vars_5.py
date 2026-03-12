import pandas as pd

archivo_ventas = "t_pr_venta_con_pred_todo.csv"
archivo_nuevas = "nuevasvars4.csv"
archivo_salida = "nuevasvars5.csv"

ventas = pd.read_csv(archivo_ventas, sep=None, engine="python", encoding="utf-8")
nuevas = pd.read_csv(archivo_nuevas, sep=None, engine="python", encoding="utf-8")

print("Columnas ventas:", ventas.columns.tolist())
print("Columnas nuevas:", nuevas.columns.tolist())

ventas_grouped = (
    ventas.groupby("co_cliente", as_index=False)["outcome_sin_con_pred"]
    .max()
)

resultado = nuevas.merge(ventas_grouped, on="co_cliente", how="left")
resultado.to_csv(archivo_salida, index=False)

print(f"Archivo guardado como {archivo_salida}")