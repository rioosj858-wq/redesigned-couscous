from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# =========================
# ARCHIVOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

archivo = os.path.join(BASE_DIR, "datos.csv")
historial_file = os.path.join(BASE_DIR, "historial.csv")
banca_file = os.path.join(BASE_DIR, "banca.txt")
learning_file = os.path.join(BASE_DIR, "learning.csv")

API_KEY = "TU_API_KEY_AQUI"

# =========================
# CREAR ARCHIVOS
# =========================
for f, cols in [
    (archivo, "local,visitante,goles_local,goles_visitante"),
    (historial_file, "local,visitante,pick,score,monto,decision"),
    (learning_file, "local,visitante,prediccion,real,acierto")
]:
    if not os.path.exists(f):
        with open(f, "w") as x:
            x.write(cols + "\n")

if not os.path.exists(banca_file):
    open(banca_file, "w").write("100")

# =========================
# BANCA
# =========================
def cargar_banca():
    return float(open(banca_file).read())

# =========================
# API PARTIDOS REALES
# =========================
def partidos_reales():
    try:
        url = "https://api.football-data.org/v4/matches"
        headers = {"X-Auth-Token": API_KEY}
        r = requests.get(url, headers=headers)
        data = r.json()

        partidos = []
        for m in data.get("matches", [])[:5]:
            home = m["homeTeam"]["name"]
            away = m["awayTeam"]["name"]
            partidos.append((home, away))

        return partidos
    except:
        return []

# =========================
# DATASET IA
# =========================
def preparar_datos():
    df = pd.read_csv(archivo)
    if len(df) < 10:
        return None

    X = df[["goles_local","goles_visitante"]]

    y = []
    for _, r in df.iterrows():
        if r["goles_local"] > r["goles_visitante"]:
            y.append(1)
        elif r["goles_local"] < r["goles_visitante"]:
            y.append(2)
        else:
            y.append(0)

    return X, y

def entrenar_modelo():
    data = preparar_datos()
    if data is None:
        return None

    X, y = data

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    return model

# =========================
# IA PRINCIPAL
# =========================
def analizar(local, visitante):

    df = pd.read_csv(archivo)
    if len(df) < 5:
        return "⚠️ Falta datos"

    model = entrenar_modelo()
    banca = cargar_banca()

    # promedio simple
    exp_local = df["goles_local"].mean() + 0.3
    exp_visit = df["goles_visitante"].mean()

    # simulación
    sims = []
    for _ in range(1200):
        l = np.random.poisson(max(exp_local, 0.3))
        v = np.random.poisson(max(exp_visit, 0.3))
        sims.append((l,v))

    df_sim = pd.DataFrame(sims, columns=["L","V"])
    conteo = df_sim.value_counts()

    top = conteo.index[0]
    prob = (conteo.iloc[0]/len(df_sim))*100

    # IA ML si existe
    if model:
        pred = model.predict([[exp_local, exp_visit]])[0]
    else:
        pred = 0

    if pred == 1:
        pick = "1"
    elif pred == 2:
        pick = "2"
    else:
        pick = "X"

    score = prob

    if score >= 65:
        decision = "🟢 FUERTE"
    elif score >= 50:
        decision = "🟡 MEDIA"
    else:
        decision = "🔴 NO ENTRAR"

    monto = round(banca * (score / 200), 2)

    hist = pd.read_csv(historial_file)
    hist.loc[len(hist)] = [local, visitante, pick, score, monto, decision]
    hist.to_csv(historial_file, index=False)

    return f"""
    🎯 {top[0]}-{top[1]}<br>
    📊 Prob: {prob:.2f}%<br>
    📈 Score: {score:.2f}<br>
    🎲 Pick: {pick}<br>
    🚨 {decision}<br>
    💰 {monto} Bs
    """

# =========================
# PARTIDOS REALES
# =========================
def mejores_reales():

    partidos = partidos_reales()

    txt = "🔥 PARTIDOS REALES<br><br>"

    for p in partidos:
        txt += f"{p[0]} vs {p[1]}<br>"

    return txt

# =========================
# UI (NO TOCADA)
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{background:#0f172a;color:white;text-align:center;font-family:Arial}
.card{background:#1f2937;margin:10px;padding:10px;border-radius:10px}
button{padding:10px;background:#00ff99;border:none}
input{padding:8px}
</style>
</head>
<body>

<h2>🏦 BET AI PRO</h2>

<div class="card">
<form method="post">
<input name="local" placeholder="Local">
<input name="visitante" placeholder="Visitante"><br><br>
<button name="accion" value="analizar">Analizar</button>
</form>
</div>

<div class="card">
<form method="post">
<button name="accion" value="real">Partidos reales</button>
</form>
</div>

<div class="card">
<pre>{{r}}</pre>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():

    r = ""

    if request.method == "POST":

        if request.form["accion"] == "analizar":
            r = analizar(request.form["local"], request.form["visitante"])

        elif request.form["accion"] == "real":
            r = mejores_reales()

    return render_template_string(HTML, r=r)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
