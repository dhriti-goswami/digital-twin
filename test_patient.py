
from src.data.ingestion import DataIngestion
import logging
logging.basicConfig(level=logging.INFO)
ingestion = DataIngestion()
try:
    patient_id = ingestion.create_patient(
        external_id='DEMO001',
        age=35,
        weight_kg=75.0,
        height_cm=175.0,
        diabetes_type='Type 1'
    )
    print('Patient ID:', patient_id)
except Exception as e:
    print('Error creating patient:', e)

