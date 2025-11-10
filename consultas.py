import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///Proyecto_Accidentalidad_Vial_Antioquia.db")

# Estructura de la tabla
df_info = pd.read_sql("SELECT ANIO, NOM_MES, GRAVEDAD_ACCIDENTE, COUNT (*) FROM Accidentalidad_Vial_Antioquia GROUP BY ANIO, NOM_MES, GRAVEDAD_ACCIDENTE ORDER BY ANIO ASC ;", con=engine)
print(df_info)




