import pandas

# Name of the original dataset excel spreadsheet
xlsx_file_name = 'alberta-post-secondary-graduate-earnings-by-field-of-study.xlsx'

# Name of the CSV file to be saved/read
csv_file_name = 'derived_data.csv'

# The names of all of the important sheets in the original dataset
def get_sheet_names():
    return [
        'One Year After Graduation',
        'Two Years After Graduation',
        'Five Years After Graduation',
        'Ten Years After Graduation'
    ]

# The names of the years each sheet pertains to
def get_sheet_year_names():
    return ['1', '2', '5', '10']

# The indices of the salary positions in the CSV file
def csv_salaries_indices():
    return [2, 4, 6, 8]

# Get the name of the average income column based on the sheet that it is/was on
def sheet_name_income(sheet_name):
    return f'Average Income {sheet_name}'

# Get the name of the cohort size column based on the sheet that it is/was on
def sheet_name_size(sheet_name):
    return f'Cohort Size {sheet_name}'

def get_original_dataset(sheet_names):
    global xlsx_file_name

    # Read the excel spreadsheet with the provided sheet names
    return pandas.read_excel(xlsx_file_name, sheet_name=sheet_names)

# Generate and save the CSV file from the original dataset
def generate_data_csv():
    # Declare the names of each sheet to grab
    sheet_names = get_sheet_names()

    # Get the dataframes for each sheet of the original dataset
    sheet_dataframes = get_original_dataset(sheet_names)
    
    # Augment each of the read dataframes
    for sheet_name in sheet_dataframes:
        # Remove NA values
        sheet_dataframes[sheet_name] = sheet_dataframes[sheet_name].dropna(axis=0)
        # Convert median income values from strings to numeric values (from strings)
        sheet_dataframes[sheet_name]['Median Income'] = pandas.to_numeric(sheet_dataframes[sheet_name]['Median Income'].replace('[\$,]', '', regex=True))
        
        # Get the name if the income and size for this sheet
        income_name = sheet_name_income(sheet_name)
        size_name = sheet_name_size(sheet_name)

        # Generate a set of groups by 'Credential' and 'Field of Study (CIP code)'
        groups = sheet_dataframes[sheet_name].groupby(['Credential', 'Field of Study (CIP code)'])

        # Go through the list of groups, and generate a list of rows with the data we want
        rows_list = []
        for key, item in groups:
            current_group = groups.get_group(key)

            # Create a single row for each group, to eliminate the use of 'Graduating Cohort'
            rows_list.append({
                'Credential': current_group.iloc[0]['Credential'],
                'Field of Study (CIP code)': current_group.iloc[0]['Field of Study (CIP code)'],
                income_name: current_group['Median Income'].mean().astype(int),
                size_name: current_group['Cohort Size'].sum().astype(int)
            })
        
        # Finally, rebuild the dataframe based on the list of rows
        rebuild_df = pandas.DataFrame(rows_list)
        sheet_dataframes[sheet_name] = rebuild_df

    # Build a better dataframe for our uses out of the above info
    # Start by copying the first dataframe
    data_frame = sheet_dataframes[sheet_names[0]].copy(deep=True)
    # Merge the remainder of the dataframes
    common_columns = ['Credential', 'Field of Study (CIP code)']
    for i in range(1, len(sheet_names)):
        data_frame = pandas.merge(data_frame, sheet_dataframes[sheet_names[i]], how='left', left_on=common_columns, right_on=common_columns)

    # For some reason, the diploma credential type is represented with the string literal "Diploma " (note the space after).
    # The following line should remove it...
    data_frame['Credential'] = data_frame['Credential'].str.replace('Diploma ', 'Diploma', regex=False)

    # Save the file
    global csv_file_name
    data_frame.to_csv(csv_file_name, index=False, na_rep='n/a')

    # Return the combined dataframes
    return data_frame

# Read the dataset if it exists, or recreate it if it does not.
def get_or_generate_dataset(force_regenerate = False):
    global csv_file_name

    # If we want to forcefully regenerate it, regenerate and return
    if force_regenerate:
        return generate_data_csv()

    # If we do not want to regenerate it, attempt to read the file first
    try:
        # If we can read the CSV file, return the data it contains
        return pandas.read_csv(csv_file_name)
    except FileNotFoundError as ex:
        # Otherwise, generate the CSV and return the data it contains
        return generate_data_csv()

if __name__ == '__main__':
    data_frame = get_or_generate_dataset(force_regenerate=False)