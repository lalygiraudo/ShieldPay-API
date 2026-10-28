import json
import logging
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

# 1. Inicializamos la aplicación FastAPI
app = FastAPI(
    title="FraudShield API",
    description="Motor predictivo de scoring y detección de fraude en tiempo real",
    version="1.1.0"
)

# 2. Cargamos el pipeline entrenado
MODEL_PATH = "pipeline_fraud_rf.joblib"
pipeline = joblib.load(MODEL_PATH)

# 3. Trazabilidad: log en consola + archivo, y registro persistente de evaluaciones
BASE_DIR = Path(__file__).parent
ARCHIVO_LOG = BASE_DIR / "shieldpay.log"
ARCHIVO_REGISTRO = BASE_DIR / "evaluaciones.jsonl"

logger = logging.getLogger("shieldpay")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _formato = logging.Formatter("%(asctime)s | %(message)s")
    for _handler in (logging.StreamHandler(), logging.FileHandler(ARCHIVO_LOG, encoding="utf-8")):
        _handler.setFormatter(_formato)
        logger.addHandler(_handler)
    logger.propagate = False

_lock = threading.Lock()
_registro: list = []


def _cargar_registro():
    """Recupera las evaluaciones previas para que el reporte sobreviva a reinicios."""
    if ARCHIVO_REGISTRO.exists():
        with open(ARCHIVO_REGISTRO, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    _registro.append(json.loads(linea))


_cargar_registro()


# 4. Función Haversine para cálculo geodésico
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radio terrestre en kilómetros
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))


# 5. Estructura del payload JSON recibido (con validaciones)
Categoria = Literal[
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]


class TransaccionInput(BaseModel):
    amt: float = Field(..., gt=0, allow_inf_nan=False)                 # Monto en USD (> 0)
    lat: float = Field(..., ge=-90, le=90, allow_inf_nan=False)        # Latitud del cliente
    long: float = Field(..., ge=-180, le=180, allow_inf_nan=False)     # Longitud del cliente
    city_pop: int = Field(..., ge=0)                                    # Población de la ciudad
    dob: str                                                            # Fecha de nacimiento "AAAA-MM-DD"
    merch_lat: float = Field(..., ge=-90, le=90, allow_inf_nan=False)  # Latitud del comercio
    merch_long: float = Field(..., ge=-180, le=180, allow_inf_nan=False)  # Longitud del comercio
    category: Categoria                                                 # Categoría del comercio
    gender: Literal["M", "F"]                                           # "M" o "F"
    state: str = Field(..., pattern=r"^[A-Z]{2}$")                      # Código de estado (ej. "NC", "NY")
    # Opcional: si no se envía se usa el reloj del servidor (UTC). Permite pruebas reproducibles.
    trans_date_trans_time: Optional[datetime] = None

    @field_validator("dob")
    @classmethod
    def validar_dob(cls, v: str) -> str:
        try:
            nacimiento = datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("dob debe tener formato AAAA-MM-DD y ser una fecha válida")
        if nacimiento.date() > date.today():
            raise ValueError("dob no puede ser una fecha futura")
        return v


# 6. Endpoint de scoring y predicción
@app.post("/v1/evaluar-transaccion")
def evaluar_transaccion(trx: TransaccionInput):
    # Fecha/hora de la transacción: la del payload si viene, si no el reloj del servidor (UTC)
    if trx.trans_date_trans_time is not None:
        ahora = trx.trans_date_trans_time
        if ahora.tzinfo is not None:
            ahora = ahora.astimezone(timezone.utc).replace(tzinfo=None)
        origen_fecha = "payload"
    else:
        ahora = datetime.now(timezone.utc).replace(tzinfo=None)
        origen_fecha = "reloj_servidor_utc"

    fecha_nac = datetime.strptime(trx.dob, "%Y-%m-%d")
    if ahora < fecha_nac:
        raise HTTPException(
            status_code=422,
            detail="trans_date_trans_time no puede ser anterior a la fecha de nacimiento (dob)"
        )

    # Feature Engineering dinámico
    hora = ahora.hour
    dia_semana = ahora.weekday()
    edad = (ahora - fecha_nac).days // 365
    distancia_km = float(haversine(trx.lat, trx.long, trx.merch_lat, trx.merch_long))

    # Armado del DataFrame con las columnas esperadas por el ColumnTransformer
    input_data = pd.DataFrame([{
        'amt': trx.amt,
        'city_pop': trx.city_pop,
        'lat': trx.lat,
        'long': trx.long,
        'merch_lat': trx.merch_lat,
        'merch_long': trx.merch_long,
        'hora': hora,
        'dia_semana': dia_semana,
        'edad': edad,
        'distancia_km': distancia_km,
        'category': trx.category,
        'gender': trx.gender,
        'state': trx.state
    }])

    # Inferencia de probabilidades: clase 0 (legítima) vs clase 1 (fraude)
    probabilidad_fraude = float(pipeline.predict_proba(input_data)[0][1])
    score_riesgo = round(probabilidad_fraude * 100.0, 2)

    # Reglas de negocio calibradas según la distribución predictiva del modelo
    if score_riesgo >= 50.0:
        decision = "RECHAZAR"
        accion = "Bloqueo preventivo de la operación por alto riesgo de contracargo"
    elif score_riesgo >= 25.0:
        decision = "DESAFIO_2FA"
        accion = "Exigir validación biométrica o código OTP por SMS/email"
    else:
        decision = "APROBAR"
        accion = "Transacción legítima procesada exitosamente"

    # Trazabilidad: ID, timestamp, score y decisión en log y en el registro persistente
    id_transaccion = str(uuid.uuid4())
    timestamp_evaluacion = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.info(
        "EVALUACION | id=%s | score=%s | decision=%s | monto=%s | distancia_km=%s | hora=%s",
        id_transaccion, score_riesgo, decision, trx.amt, round(distancia_km, 2), hora
    )
    registro = {
        "id_transaccion": id_transaccion,
        "timestamp_evaluacion": timestamp_evaluacion,
        "score_riesgo": score_riesgo,
        "decision": decision,
    }
    with _lock:
        _registro.append(registro)
        with open(ARCHIVO_REGISTRO, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro) + "\n")

    return {
        "id_transaccion": id_transaccion,
        "timestamp_evaluacion": timestamp_evaluacion,
        "score_riesgo": score_riesgo,
        "decision": decision,
        "accion_sugerida": accion,
        "features_calculadas": {
            "distancia_km": round(distancia_km, 2),
            "edad_titular": edad,
            "hora_transaccion": hora,
            "dia_semana": dia_semana,
            "origen_fecha_hora": origen_fecha
        }
    }


# 7. Endpoint de reporte de indicadores de fraude
@app.get("/v1/reporte")
def reporte(
    desde: Optional[date] = Query(None, description="Fecha inicial AAAA-MM-DD (opcional)"),
    hasta: Optional[date] = Query(None, description="Fecha final AAAA-MM-DD (opcional)"),
):
    if desde and hasta and desde > hasta:
        raise HTTPException(status_code=422, detail="'desde' no puede ser posterior a 'hasta'")

    with _lock:
        registros = list(_registro)

    def en_periodo(r):
        dia = date.fromisoformat(r["timestamp_evaluacion"][:10])
        return (desde is None or dia >= desde) and (hasta is None or dia <= hasta)

    seleccion = [r for r in registros if en_periodo(r)]
    total = len(seleccion)
    aprobadas = sum(r["decision"] == "APROBAR" for r in seleccion)
    a_revisar = sum(r["decision"] == "DESAFIO_2FA" for r in seleccion)
    bloqueadas = sum(r["decision"] == "RECHAZAR" for r in seleccion)

    return {
        "periodo": {
            "desde": desde.isoformat() if desde else None,
            "hasta": hasta.isoformat() if hasta else None,
        },
        "total_evaluadas": total,
        "aprobadas": aprobadas,
        "a_revisar": a_revisar,
        "bloqueadas": bloqueadas,
        "score_promedio": round(sum(r["score_riesgo"] for r in seleccion) / total, 2) if total else 0.0,
        "tasa_bloqueo_pct": round(bloqueadas * 100.0 / total, 2) if total else 0.0,
    }
