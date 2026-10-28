"""
Barrido para encontrar el JSON de CF-06 (score entre 25 y 50 -> DESAFIO_2FA).
Uso (con la API levantada en otra terminal):
    pip install requests
    python probar_scores_cf06.py
Marca con <<< CF-06 las combinaciones que caen en el rango buscado.
Nota: cada llamada queda registrada en el log y en /v1/reporte (son 20 evaluaciones).
"""
import requests

URL = "http://127.0.0.1:8000/v1/evaluar-transaccion"

BASE = {
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 8400000,
    "dob": "1990-03-22",
    "merch_lat": 40.9000,
    "merch_long": -74.2000,
    "gender": "M",
    "state": "NY",
}

CATEGORIAS = ["shopping_net", "misc_net"]
HORAS = ["22:15:00", "23:30:00"]
MONTOS = [400, 500, 600, 700, 800]

print(f"{'categoria':<14}{'hora':<10}{'amt':>6}  {'score':>6}  decision")
print("-" * 55)
for cat in CATEGORIAS:
    for hora in HORAS:
        for amt in MONTOS:
            payload = {
                **BASE,
                "category": cat,
                "amt": amt,
                "trans_date_trans_time": f"2026-10-20 {hora}",
            }
            r = requests.post(URL, json=payload, timeout=10)
            if r.status_code == 200:
                d = r.json()
                marca = "  <<< CF-06" if 25 <= d["score_riesgo"] < 50 else ""
                print(f"{cat:<14}{hora:<10}{amt:>6}  {d['score_riesgo']:>6}  {d['decision']}{marca}")
            else:
                print(f"{cat:<14}{hora:<10}{amt:>6}  HTTP {r.status_code}")
