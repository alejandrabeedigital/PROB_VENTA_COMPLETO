import pandas as pd

df_resultados = pd.read_csv("todo_con_resultados_17.csv", low_memory=False)
df_descuelgue = pd.read_csv("todo_con_prob_descuelgue.csv", low_memory=False)

df_resultados["prob_descuelgue_modelo"] = df_descuelgue["prob_descuelgue_modelo"]

df_resultados["prob_final_venta_precontacto"] = (
    df_resultados["prob_descuelgue_modelo"] * df_resultados["prob_venta_modelo"]
)

df_resultados.to_csv(
    "todo_con_resultados_prob_final_venta_precontacto.csv",
    index=False
)

print("CSV generado: todo_con_resultados_prob_final_venta_precontacto.csv")