# Analysis of Hospital Appointment No-Shows (Anonymized Data)

## 1. Project Overview

This project analyzes factors contributing to patient no-shows for hospital appointments. The goal is to build a predictive model that can help the hospital administration reduce revenue loss and improve resource allocation by identifying patients who are at a high risk of missing their appointments.

## 2. Data Source & NDA Compliance

**CRITICAL:** The original dataset contains protected health information (PHI) and is subject to a strict Non-Disclosure Agreement (NDA). The raw data cannot and will not be shared or uploaded to any public repository.

*To facilitate collaboration and showcase the analytical methodology without compromising privacy, a fully anonymized and synthetic dataset has been generated. This new dataset (`noshow_data_anonymized.csv`) retains the statistical properties and predictive features of the original data but contains no real patient information.*

## 3. Data Anonymization Strategy

The following multi-step process was executed to de-identify the original dataset and generate the public version. This ensures that re-identification of any individual is computationally infeasible.

### Step 1: Removal of Direct Identifiers

- All columns that directly identify an individual were completely removed:
  - *Patient-ID*
  - *PatientName*
  - *PhoneNumber*
  - *Address*
  - *Email*
  - *AppointmentId*

### Step 2: Pseudonymization

- A new, non-identifiable `PatientId` was created. Original patient identifiers were mapped to a new set of randomly generated, non-sequential IDs (e.g., `P000001`, `P000002`).
- This preserves the ability to track repeat appointments for a single (but anonymous) patient without exposing their real identity.

### Step 3: Data Generalization & Binning

- Continuous or highly specific data points were grouped into broader categories to reduce granularity:
  - **Age:** Converted from a specific number (e.g., 47) to an age group (e.g., 40-50).
  - **Neighbourhood:** High-cardinality location data was mapped to broader, anonymized regions (e.g., Region-A, Region-B). This prevents identification through geographic profiling.

### Step 4: Date & Time Obfuscation

- All date and time information was shifted to protect the timeline of events:
  - **ScheduledDay & AppointmentDay:** All dates were shifted backward by a random, but consistent, number of days for each patient. This preserves the crucial `DaysInAdvance` calculation (`AppointmentDay - ScheduledDay`) while making the actual appointment dates meaningless.
  - **Time:** The time of the appointment was removed, as it was not a significant feature in the initial analysis.

### Step 5: Feature Engineering & Transformation

- New features were derived from the original data before anonymization to retain predictive power:
  - **DaysInAdvance:** Calculated from the original `ScheduledDay` and `AppointmentDay`.
  - **DayOfWeek:** Extracted from the `AppointmentDay`.

## 4. Anonymized Dataset

- `noshow_data_anonymized.csv`