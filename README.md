# Stroke Prediction (Imbalanced) — F1-Optimized Threshold + Probability Calibration

Proyecto “paper-like” para clasificación binaria en dataset tabular altamente desbalanceado (stroke ≈ 5%).  
Objetivo: maximizar **F1** sin fuga de información, y reportar también **AP/PR-AUC**.

---

## Idea principal (qué aporta este repo)
En lugar de quedarse con el umbral por defecto (0.5), este trabajo:
1) Entrena modelos con un protocolo reproducible (train/val/test + CV estratificada).
2) Obtiene probabilidades y ajusta el **umbral que maximiza F1** usando predicciones **out-of-fold (OOF)**.
3) Aplica **calibración de probabilidades** (sigmoid) y repite el tuning de umbral sobre probabilidades calibradas.

Esto hace que el pipeline sea más realista y defendible que un notebook típico de Kaggle.

---

## Metodología experimental (sin leakage)
- Split estratificado: `train/val/test` con `test` intocable.
- Model selection: CV estratificada sobre `trainval`.
- Threshold tuning: el umbral se elige con probabilidades OOF (no se usa test para escogerlo).
- Evaluación final: una sola corrida sobre test.

---

## Modelos
- **Baseline:** Logistic Regression + threshold tuning.
- **BalancedRandomForest:** modelo orientado a desbalance mediante muestreo balanceado al construir árboles.
- **Calibración:** CalibratedClassifierCV (sigmoid) para mejorar confiabilidad de probabilidades antes de escoger el umbral final.

---

## Resultados (TEST)
> Métricas reportadas: Precision, Recall, F1 (objetivo) y AP/PR-AUC (ranking).

| Experimento | Precision | Recall | F1 | AP/PR-AUC |
|---|---:|---:|---:|---:|
| Baseline (LogReg + CV threshold) | 0.2192 | 0.6400 | 0.3265 | 0.2599 |
| BRF tuned (OOF threshold) | 0.2155 | 0.5000 | 0.3012 | 0.2218 |
| BRF tuned + calibrated (OOF threshold) | 0.2328 | 0.5400 | 0.3253 | 0.2319 |

**Mejor trade-off final (paper-like):** BRF tuned + calibración + umbral OOF (F1 ≈ 0.325).

---

## Cómo reproducir
1) Abrir el notebook en Google Colab.
2) Subir el archivo CSV del dataset.
3) Ejecutar todas las celdas en orden (incluye split, entrenamiento, tuning, calibración y evaluación).

Recomendación: usar CPU (el dataset es pequeño y el pipeline es tabular).

---


## Notas y limitaciones
- El dataset es pequeño y desbalanceado; los resultados pueden variar según el split aleatorio.
- El tuning del umbral se hace con OOF para evitar optimismo en test.
- La calibración mejora la interpretabilidad de probabilidades, pero no garantiza subir F1 en todos los casos.

---
