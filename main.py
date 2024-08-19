import numpy as np
import pandas as pd
import json
import uuid

with open("input.json", "r") as file:
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
arr = ["id", "Name value", "date", "Memo/Description", "Account id", 'Account value', "Date", "Transaction Type id",
       "Transaction Type value", "Department id", "Amount"]
df = transformed_df[[col for col in arr if col in transformed_df.columns]]
df = df.fillna("")

rename = {'id': 'original_transaction_id',
          'Name value': 'name',
          'Memo/Description': 'memo',
          'Account value': 'account',
          'Account id': 'Account id',
          'Transaction Type id': 'transaction_id',
          'Transaction Type value': 'transaction_type',
          'Date': 'date',
          'Department id': 'department_id',
          'Amount': 'amount'
          }
df = df.rename(columns=rename)

# Forward-fill the id column to propagate the previous id to subsequent rows
df['Previous id'] = df['original_transaction_id'].replace('', pd.NA)
df['original_transaction_id'] = df['original_transaction_id'].replace('', pd.NA).fillna(method='ffill')
df = df.dropna(subset=['original_transaction_id'])
df = df.drop(columns=['Previous id'])
mask = df.drop(columns=['original_transaction_id']).applymap(lambda x: x == "" or pd.isna(x)).all(axis=1) & df[
    'original_transaction_id'].notna()
df = df[~mask]

# Reset the index if needed
df.reset_index(drop=True, inplace=True)
df["original_transaction_id"] = df["original_transaction_id"].replace('', pd.NA).fillna(method='ffill')
df['original_transaction_id'] = df['transaction_id']

# Iterate through rows to set 'from_account_id' and 'to_account_id'
df['from_account_id'] = None
df['to_account_id'] = None
for i in range(len(df)):
    if i < len(df) - 1:
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row['transaction_type'] and not next_row['transaction_type']:
            df.at[i, 'to_account_id'] = next_row['Account id']
            df.at[i, 'from_account_id'] = current_row['Account id']
        elif not current_row['transaction_type'] and i > 0 and df.iloc[i - 1]['transaction_type']:
            prev_row = df.iloc[i - 1]
            df.at[i, 'from_account_id'] = prev_row['Account id']
            df.at[i, 'to_account_id'] = current_row['Account id']
        elif not current_row['transaction_type'] and i > 0 and not df.iloc[i - 1]['transaction_type']:
            df.at[i, 'to_account_id'] = None

arr = ["original_transaction_id", "name", "memo", "date", "account", "department_id", "from_account_id",
       "to_account_id", "transaction_type", "amount"]
df = df[[col for col in arr if col in df.columns]]

# Iterate name, memo, date from previous
for i in range(1, len(df)):
    if pd.isna(df.at[i, 'name']) or df.at[i, 'name'] == "":
        df.at[i, 'name'] = df.at[i - 1, 'name']
    if pd.isna(df.at[i, 'memo']) or df.at[i, 'memo'] == "":
        df.at[i, 'memo'] = df.at[i - 1, 'memo']
    # Handle date
    if pd.isna(df.at[i, 'date']) or df.at[i, 'date'] == "" or df.at[i, 'date'] == "0-00-00":
        df.at[i, 'date'] = df.at[i - 1, 'date']
df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

df['from_account_id'] = df['from_account_id'].fillna(method='ffill')
df['to_account_id'] = df['to_account_id'].fillna(method='ffill')

# Variable to keep track of the current split_id
current_split_id = None

# Iterate over DataFrame rows
for index, row in df.iterrows():
    if row['transaction_type'] != "":
        current_split_id = row['id']
    if current_split_id is not None:
        df.at[index, 'split_id'] = current_split_id

for i in range(1, len(df)):
    if pd.isna(df.at[i, 'transaction_type']) or df.at[i, 'transaction_type'] == "":
        df.at[i, 'transaction_type'] = df.at[i - 1, 'transaction_type']
df.reset_index(drop=True, inplace=True)

def transform_data(df):
    # Create a dictionary to keep track of the new 'from_id' values for each split_id
    new_from_id_map = {}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        if row['id'] != row['split_id']:
            # If split_id not seen before, use the 'to_id' from the first entry
            if row['split_id'] not in new_from_id_map:
                # Store the initial 'from_id' of the current split_id
                new_from_id_map[row['split_id']] = row['to_account_id']

            # Update 'from_id' to the last seen 'to_id' for this split_id
            df.at[index, 'from_account_id'] = new_from_id_map[row['split_id']]
            # Set 'to_id' to None
            df.at[index, 'to_account_id'] = None

    return df


# Transform the data
df_transformed = transform_data(df)
grouped = df.groupby('split_id')
filtered = grouped.apply(lambda x: x.iloc[[0]] if len(x) == 2 else x).reset_index(drop=True)
filtered.loc[filtered['id'] == filtered['split_id'], 'split_id'] = np.nan


# Reset index to flatten the DataFrame
filtered.to_json('output.json', orient='records')

print(df)
