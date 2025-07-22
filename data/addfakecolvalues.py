# Now we are going to add fake values (100) to this columns: Location, and DoctorName to existing data

import pandas as pd
import numpy as np
import random
import string

# Read the CSV file into a DataFrame    

df = pd.read_csv('/Users/ronitsaxena/Developer/Personal/predictml-production/data/synthetic_data.csv')

df['Location'] = df['Location'].astype(str)
# Function to generate random strings but meaningful values

def generate_random_string(length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

# Apply the function to the 'Location' column
df['Location'] = df['Location'].apply(lambda x: generate_random_string(10))


# now for Random DoctorNames

def generate_random_doctor_name():
    first_name = generate_random_string(10)
    last_name = generate_random_string(10)
    return f"{first_name} {last_name}"

df['DoctorName'] = df['DoctorName'].apply(lambda x: generate_random_doctor_name())

df.to_csv('/Users/ronitsaxena/Developer/Personal/predictml-production/data/synthetic_data.csv', index=False)