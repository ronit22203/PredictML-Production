# Preprocessing Pipeline Key Points

This document highlights the technical and data science capabilities demonstrated in `src/preprocessing.py` for the PredictML no-show prediction feature engineering pipeline.

## 1. Modular, Object-Oriented Design

- Implemented as a `HealthcarePreprocessor` class with clear, single-responsibility methods:
  - `validate_input_data`
  - `clean_initial_data`
  - `process_age_features`
  - `process_appointment_features`
  - `process_billing_features`
  - `process_booking_features`
  - `process_categorical_features`
  - `process_target_variable`
  - `final_cleanup`
- Configuration-driven via a centralized `CONFIG` dictionary for easy tunability (e.g., bin thresholds, labels, columns to drop).

## 2. Robust Data Validation and Error Handling

- Checks for required schema columns and raises informative errors if missing.
- Ensures non-empty input to prevent downstream failures.
- Wraps CSV loading in try/except to catch I/O issues gracefully.

## 3. Comprehensive Feature Engineering

### a. Date and Time Features
- Parses dates (`DOB`, `AppointmentDate`, `Booked_Date_Time`, `Previous_Bill_Date`) with `pd.to_datetime` and error coercion.
- Derives calendar features (year, month, day, quarter, week-of-year, day-of-week).
- Calculates custom flags: weekend appointments, week-of-month.
- Determines season (`hot`, `warm`, `mild`, `unknown`) from month.

### b. Age Features
- Computes `age_at_visit` in years from `DOB` to `AppointmentDate`.
- Extracts birth date components and day-of-week.
- Bins age into meaningful cohorts using configurable bins & labels.

### c. Billing and Recency
- Calculates `days_since_prev_bill` as time since last bill.
- Groups recency into buckets (e.g., ≤30d, 31–90d) for recency-based analysis.

### d. Booking Lead Time
- Computes lead time in days between booking and appointment.
- Bins lead time into categories: same-day, 1–7d, 8–30d, etc.
- Flags same-day bookings for no-show risk modeling.

### e. Categorical Encoding
- Cleans and normalizes text columns (e.g., lowercase `Location`).
- Groups rare categories into "Other" for high cardinality features.
- Converts object columns to `category` dtype for memory efficiency.

## 4. Target Variable and Pipeline Flow

- Uniformly labels target as ‘No Show’ after filtering relevant statuses.
- Orchestrates end-to-end pipeline in `preprocess_data` method for seamless execution.

## 5. Production-Ready CLI Integration

- Provides a `main()` entrypoint with command-line arguments for input/output paths.
- Outputs success logs and final dataset dimensions.
- Uses minimal, open-source dependencies (Pandas, NumPy) for portability.

## 6. Memory and Performance Considerations

- Utilizes pandas nullable integer types (`Int64`, `Int8`) and `category` dtype.
- Drops unnecessary intermediate columns to reduce memory footprint.
- Suppresses warnings to maintain clean logs during batch runs.

---
*This pipeline demonstrates expertise in data cleaning, feature engineering, modular software design, and preparing data for machine learning workflows.*
