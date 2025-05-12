import pandas as pd
import os 

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mall_Customers-Missing Values.xlsx")

archivo_leido = pd.read_excel(path)

age = archivo_leido["Age"]
annualIncome = archivo_leido["Annual Income (k$)"]
spendingCore = archivo_leido["Spending Score (1-100)"]
loan = archivo_leido["Suitable for a bank loan"]

promedioEdad = round(age.mean())
promedioIncome = round(annualIncome.median())
promedioSpendingCore = round(spendingCore.median())

defaultValues = {
  "Age": promedioEdad,
  "Annual Income (k$)": promedioIncome,
  "Spending Score (1-100)": promedioSpendingCore
}

archivo_leido.fillna(value=defaultValues, inplace=True)

print("\nTÉCNICA DE IMPUTACIÓN | IMPUTACIÓN CON MEDIDA DE TENDENCIA CENTRAL")
print(f"\n{archivo_leido}")




