from typing import Optional, List

import numpy as np
import pandas as pd
import uuid

from app.db.schema import BaseModel

# Rename columns to match output specification
rename_map = {
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
# Columns of interest for the final output
target_columns = [
    "Name value", "Memo/Description", "Account id", 'Account value',
    "Date", "Transaction Type id", "Transaction Type value",
    "Department id", "Amount"
]


class Transaction(BaseModel):
    name: str
    memo: str
    from_account_id: str
    account: str
    date: str
    original_transaction_id: str
    transaction_type: str
    department_id: Optional[str] = ""
    amount: str
    to_account_id: Optional[str] = ""
    id: str
    split_id: Optional[str] = ""


def df_to_transactions(df_filtered) -> List[dict]:
    """
    Converts the filtered DataFrame to a list of Transaction objects.

    Args:
        df_filtered (pd.DataFrame): The filtered DataFrame containing transaction data.

    Returns:
        List[Transaction]: A list of Transaction objects.
    """

    transactions = df_filtered.to_dict(orient='records')
    return transactions
    # return [Transaction.parse_obj(transaction) for transaction in transactions]


def transform_row(row):
    """
    Transforms each row to include only necessary information.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        pd.Series: The transformed row with selected key-value pairs.
    """
    transformed_row = {}
    for key, value in row.items():
        if isinstance(value, dict) and "value" in value:
            # Handle columns with both 'value' and 'id' keys
            if "id" in value:
                transformed_row[f"{key} value"] = value["value"]
                transformed_row[f"{key} id"] = value["id"]
            else:
                transformed_row[key] = value["value"]
        else:
            transformed_row[key] = value
    return pd.Series(transformed_row)


def process_transaction_data(data):
    """
    Processes transaction data from a JSON input and returns it in a specified format.

    Args:
        data (dict): The input data parsed from a JSON file.

    Returns:
        list[dict]: A list of dictionaries, each representing a transaction.
    """

    # Extract column headers and insert 'id' at the start
    columns = [col["ColTitle"] for col in data["Columns"]["Column"]]
    columns.insert(0, "id")

    # Extract rows from the data
    rows = []
    for section in data["Rows"]["Row"]:
        # Extract the header row
        header_row = [col.get("id", "") for col in section["Header"]["ColData"]]

        # Extract each row's data and prepend the header ID
        for row in section["Rows"]["Row"]:
            row_data = [col for col in row["ColData"]]
            row_data.insert(0, header_row[0])  # Insert the header ID at the beginning
            rows.append(row_data)

    # Create DataFrame from extracted rows and columns
    df = pd.DataFrame(rows, columns=columns)

    # Apply transformation to each row in the DataFrame
    transformed_df = df.apply(transform_row, axis=1)

    df = transformed_df[[col for col in target_columns if col in transformed_df.columns]]
    df = df.fillna("")

    df = df.rename(columns=rename_map)
    df['to_account_id'] = ""
    previous_index = None

    # Update 'to_account_id' with the 'from_account_id' of the next non-empty row
    for index, row in df.iterrows():
        if not row.isnull().all():  # Check if the current row is not entirely null
            if previous_index is not None:
                df.at[previous_index, 'to_account_id'] = row['from_account_id']
            previous_index = index

    # Clear 'to_account_id' if 'from_account_id' is empty
    df.loc[df['from_account_id'] == "", 'to_account_id'] = ""

    def should_assign_uuid(row):
        """Determines if a row should be assigned a UUID."""
        return not all(cell == '' for cell in row)

    # Assign unique 'id' and initialize 'split_id'
    df['id'] = df.apply(lambda row: str(uuid.uuid4()) if should_assign_uuid(row) else "", axis=1)
    df['split_id'] = ""

    def is_row_empty(row):
        """Checks if a row is entirely empty."""
        return all(value == '' for value in row)

    # Manage 'split_id' assignment for rows that are not empty
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

    # Correct 'split_id' and 'to_account_id' based on conditions
    df['split_id'] = df.apply(lambda row: "" if row['id'] == row['split_id'] else row['split_id'], axis=1)
    df['to_account_id'] = df.apply(lambda row: "" if row['split_id'] != "" else row['to_account_id'], axis=1)

    # Ensure 'split_id' and 'id' are strings for exact matching
    df['split_id'] = df['split_id'].astype(str)
    df['id'] = df['id'].astype(str)

    # Filter out rows with empty 'split_id' but non-empty 'id'
    filtered_df = df[(df['split_id'] == '') & (df['id'] != '')]

    # Merge with filtered DataFrame to fill in missing parent information
    merged_df = df.merge(filtered_df[['id', 'name', 'date', 'transaction_type']],
                         left_on='split_id',
                         right_on='id',
                         suffixes=('', '_parent'),
                         how='left')
    merged_df = merged_df.fillna('')

    # Update original DataFrame based on merged information
    df['name'] = np.where(df['name'] == '', merged_df['name_parent'], df['name'])
    df['date'] = np.where(df['date'] == '0-00-00', merged_df['date_parent'], df['date'])
    df['transaction_type'] = np.where(df['transaction_type'] == '', merged_df['transaction_type_parent'],
                                      df['transaction_type'])

    # Filter to keep only rows with 'split_id' that appear more than once
    split_id_counts = df['split_id'].value_counts()
    split_id_to_keep = split_id_counts[split_id_counts > 1].index
    df = df[df['split_id'].isin(split_id_to_keep)]
    df_filtered = df[df['id'] != '']
    # rename id to apideck_uid
    df_filtered = df_filtered.rename(columns={'id': 'uid'})

    # Return the final processed data
    return df_filtered
