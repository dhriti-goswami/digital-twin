
from datetime import datetime
from src.data.ingestion import DataIngestion
import logging
logging.basicConfig(level=logging.INFO)
ingestion = DataIngestion()
res = ingestion.ingest_cgm_reading(patient_id=1, timestamp=datetime.now(), glucose_mg_dl=120.0)
print('Result of injection:', res)

