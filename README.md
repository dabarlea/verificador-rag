# Verificador de respuestas RAG

Proyecto final de MMIA 6012 Aprendizaje Supervisado (USFQ, 2026).

Clasifica si una respuesta generada por un sistema RAG está respaldada por sus pasajes o contiene una alucinación, usando el corpus público RAGTruth (tarea QA).

- Features: NLI por oración (`cross-encoder/nli-deberta-v3-small`), similitud coseno (`all-MiniLM-L6-v2`) y features simples del texto.
- Modelos: regresión logística y XGBoost en pipelines de scikit-learn con `SelectKBest`, optimizados con Optuna y comparados con Wilcoxon (10 × 10 folds agrupados por pregunta).
- Resultado en test: F1 = 0,49 (prompt a GPT-4-turbo en el paper de RAGTruth: 0,456).

## Cómo correrlo

**Google Colab:** [abrir en Colab](https://colab.research.google.com/github/dabarlea/verificador-rag/blob/main/verificador_rag.ipynb) y ejecutar todo. La primera celda clona este repositorio e instala lo que falta.

**Local (uv):**

```
uv sync
uv run jupyter nbconvert --to notebook --execute verificador_rag.ipynb --inplace
```

`data/features.parquet` guarda las features ya extraídas. Si se borra, el notebook las vuelve a calcular (tarda bastante sin GPU). El corpus RAGTruth se descarga solo desde su repositorio oficial.

## Archivos

- `verificador_rag.ipynb`: el proyecto completo.
- `verificador_rag.pdf`: el notebook ejecutado, en PDF.
- `data/features.parquet`: features extraídas.
