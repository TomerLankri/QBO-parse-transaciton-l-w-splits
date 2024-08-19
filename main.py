import numpy as np
import pandas as pd
import json
import uuid

with open("../pythonScript/pythonScript/input.json", "r") as file:
    data = json.load(file)

# Extract column headers
columns = [col["ColTitle"] for col in data["Columns"]["Column"]]
columns.insert(0, "id")
# Extract rows
rows = []
for section in data["Rows"]["Row"]:
    # Add the header row first
    header_row = [col.get("id", "") if not isinstance(columns, str) else col for col in section["Header"]["ColData"]]

    # Add the data rows
    for row in section["Rows"]["Row"]:
        row_data = [col for index, col in enumerate(row["ColData"])]
        row_data.insert(0, header_row[0])
        # print(row_data[6])
        rows.append(row_data)

# Create DataFrame
df = pd.DataFrame(rows, columns=columns)


def transform_row(row):
    transformed_row = {}
    for key, value in row.items():
        if isinstance(value, dict) and "value" in value and "id" in value:
            transformed_row[f"{key} value"] = value["value"]
            transformed_row[f"{key} id"] = value["id"]
        elif isinstance(value, dict) and "value" in value:
            transformed_row[key] = value["value"]
        else:
            transformed_row[key] = value
    return pd.Series(transformed_row)


# Apply the transformation to each row in the DataFrame
transformed_df = df.apply(transform_row, axis=1)
arr = ["Name value", "date", "Memo/Description", "Account id", 'Account value', "Date", "Transaction Type id",
       "Transaction Type value", "Department id", "Amount"]
df = transformed_df[[col for col in arr if col in transformed_df.columns]]
df = df.fillna("")

rename = {
    'Name value': 'name',
    'Memo/Description': 'memo',
    'Account value': 'account',
    'Account id': 'from_account_id',
    'Transaction Type id': 'original_transaction_id',
    'Transaction Type value': 'transaction_type',
    'Date': 'date',
    'Department id': 'department_id',
    'Amount': 'amount'
}
df = df.rename(columns=rename)
df['to_account_id'] = ""
previous_index = None

# Iterate through the DataFrame
for index, row in df.iterrows():
    # Check if the current row is not entirely null
    if not row.isnull().all():
        if previous_index is not None:
            # Assign the 'from_account_id' of the current row to the 'to_account_id' of the previous non-null row
            df.at[previous_index, 'to_account_id'] = row['from_account_id']

        # Update previous_index to current index
        previous_index = index

df.loc[df['from_account_id'] == "", 'to_account_id'] = ""


# Add 'split_id' column for the final output
def should_assign_uuid(row):
    return not all(cell == '' for cell in row)


# Apply function and assign UUIDs where appropriate
df['id'] = df.apply(lambda row: str(uuid.uuid4()) if should_assign_uuid(row) else "", axis=1)
df['split_id'] = ""


# Function to check if all values in a row are empty strings
def is_row_empty(row):
    return all(value == '' for value in row)


# Initialize variables
current_uuid = None
assign_uuid = False

for index, row in df.iterrows():
    if is_row_empty(row):
        assign_uuid = False
    else:
        if not assign_uuid:
            current_uuid = row['id']
            assign_uuid = True
        df.at[index, 'split_id'] = current_uuid

df['split_id'] = df.apply(lambda row: "" if row['id'] == row['split_id'] else row['split_id'], axis=1)
df['to_account_id'] = df.apply(lambda row: "" if row['split_id'] != "" else row['to_account_id'], axis=1)

# Ensure columns are of string type for exact matching
df['split_id'] = df['split_id'].astype(str)
df['id'] = df['id'].astype(str)

# Filter out rows with empty split_id
filtered_df = df[(df['split_id'] == '') & (df['id'] != '')]

# Merge DataFrame with itself based on split_id and id
merged_df = df.merge(filtered_df[['id', 'name', 'date', 'transaction_type']],
                     left_on='split_id',
                     right_on='id',
                     suffixes=('', '_parent'),
                     how='left')
merged_df = merged_df.fillna('')

df['name'] = np.where(df['name'] == '', merged_df['name_parent'], df['name'])
df['date'] = np.where(df['date'] == '0-00-00', merged_df['date_parent'], df['date'])
df['transaction_type'] = np.where(df['transaction_type'] == '', merged_df['transaction_type_parent'],
                                  df['transaction_type'])

split_id_counts = df['split_id'].value_counts()

# Filter out split_id that only appears once
split_id_to_keep = split_id_counts[split_id_counts > 1].index

# Keep only rows with split_id that appear more than once
df = df[df['split_id'].isin(split_id_to_keep)]
df_filtered = df[df['id'] != '']

print(df)
