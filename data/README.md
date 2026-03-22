# Real Diabetes Datasets

**This system uses 100% REAL patient data - NO simulated or synthetic data.**

## Data Sources

### 1. UCI Diabetes Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)
- **Citation:** Kahn, M. (1994). UCI Machine Learning Repository
- **Patients:** 70 Type 1 Diabetes patients
- **Duration:** Multiple weeks per patient
- **Content:**
  - Blood glucose measurements (multiple times daily)
  - Insulin doses (Regular, NPH, UltraLente)
  - Meal information (typical, large, small meals)
  - Exercise records
  - Hypoglycemic symptoms
  - Special events

### 2. PIMA Indians Diabetes Dataset
- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Citation:** Smith, J.W., et al. (1988). "Using the ADAP learning algorithm"
- **Patients:** 768 Pima Indian women
- **Content:**
  - Plasma glucose concentration (oral glucose tolerance test)
  - Diastolic blood pressure (mm Hg)
  - Triceps skin fold thickness (mm)
  - 2-Hour serum insulin (mu U/ml)
  - Body mass index
  - Diabetes pedigree function
  - Age
  - Diabetes diagnosis outcome

### 3. Diabetes 130-US Hospitals Dataset
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Citation:** Strack, B., et al. (2014). "Impact of HbA1c Measurement"
- **Records:** 101,766 patient encounters
- **Duration:** 10 years (1999-2008)
- **Hospitals:** 130 US hospitals
- **Content:**
  - 50+ clinical features
  - Diagnoses (ICD-9 codes)
  - Procedures
  - Medications and dosage changes
  - HbA1c measurements
  - Laboratory results
  - Hospital readmission outcomes

### 4. CGM Trace Samples
- **Source:** Research repositories (Digital Biomarker Discovery Pipeline)
- **Content:**
  - Real continuous glucose monitoring traces
  - 5-minute interval readings
  - Full daily glucose curves

## Data Format (After Parsing)

### glucose_real.csv
```
patient_id,timestamp,glucose_mg_dl
1,1991-04-21 08:00,120.0
1,1991-04-21 12:00,185.0
...
```

### insulin_real.csv
```
patient_id,timestamp,dose_units,insulin_type
1,1991-04-21 07:45,8.0,regular
1,1991-04-21 22:00,15.0,NPH
...
```

### meals_real.csv
```
patient_id,timestamp,carbs_grams,meal_type
1,1991-04-21 08:00,50.0,typical
1,1991-04-21 12:30,75.0,large
...
```

## Data Download

The data is downloaded automatically when you run:

```bash
python scripts/setup.py
```

Or download manually:

```bash
python scripts/download_real_data.py
```

## Ethical Use

This data contains anonymized medical records from real patients. Please:

1. **Do not attempt to re-identify patients**
2. **Use only for research/educational purposes**
3. **Cite the original data sources in publications**
4. **Follow all applicable data protection regulations**

## Statistics

After parsing all datasets, you should have approximately:

- **Glucose readings:** 10,000+ measurements
- **Insulin doses:** 5,000+ records
- **Meals:** 3,000+ records
- **Clinical profiles:** 768+ patients (PIMA)
- **EHR encounters:** 100,000+ records (130-Hospitals)

All of this is **REAL** data from **REAL** patients.
