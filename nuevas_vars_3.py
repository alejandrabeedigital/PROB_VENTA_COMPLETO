import pandas as pd

archivo_ventas = "nuevasvars2.csv"
archivo_salida = "nuevasvars3.csv"

ventas = pd.read_csv(
    archivo_ventas,
    low_memory=False,
    dtype={"co_cliente": str}
)

ventas.columns = ventas.columns.str.strip()

if "co_cliente" not in ventas.columns:
    raise ValueError(f"No existe la columna 'co_cliente' en {archivo_ventas}")

#if "outcome_sin_con_pred" not in ventas.columns:
    raise ValueError(f"No existe la columna 'outcome_sin_con_pred' en {archivo_ventas}")

# -------------------------------
# Crear variable origen_sc_o_no
# -------------------------------
if "origen" not in ventas.columns:
    raise KeyError("No existe la columna 'origen' en el CSV de entrada.")

origen_norm = (
    ventas["origen"]
    .astype(str)
    .str.strip()
    .str.lower()
)

ventas["origen_sc_o_no"] = origen_norm.str.startswith("sc").astype(int)

# Guardar archivo
ventas.to_csv(archivo_salida, index=False)

print(f"Proceso terminado. Archivo generado: {archivo_salida}")
print(f"Filas ventas: {len(ventas)}")
#print(f"Clientes con outcome informado: {ventas['outcome_sin_con_pred'].notna().sum()}")
print(f"origen_sc_o_no = 1: {ventas['origen_sc_o_no'].sum()}")