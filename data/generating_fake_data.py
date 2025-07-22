## We are going to use Synthetic Data Vault (SDV) to generate synthetic data.
## SDV is a library that allows you to generate synthetic data based on real data.
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

#CTGAN is a type of generative model that can learn the distribution of a dataset and generate new samples that are similar to the original data.

#Loading the real data
real_data = pd.read_csv('placeholder.csv')

# Drop high cardinality categorical columns to reduce dimensionality
threshold = 50
categorical_cols = real_data.select_dtypes(include=['object', 'category']).columns
high_cardinality_cols = [col for col in categorical_cols if real_data[col].nunique() > threshold]
if high_cardinality_cols:
    print(f"Dropping high cardinality columns: {high_cardinality_cols}")
    real_data = real_data.drop(columns=high_cardinality_cols)

# Detect metadata from your dataframe
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
# Optionally specify primary key if needed:
# metadata.set_primary_key(column_name='id')

model = CTGANSynthesizer(metadata)
model.fit(real_data)
synthetic_data = model.sample(num_rows=100)


# Save the synthetic data to a CSV file
synthetic_data.to_csv('data/synthetic_data.csv', index=False)