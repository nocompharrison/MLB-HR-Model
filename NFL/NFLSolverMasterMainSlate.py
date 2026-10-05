import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import xlwings as xw  # Import xlwings

# Load the Excel file
file_path = r'C:\Users\hlee145\Documents\FanDuel Spreadsheets\FanDuel NFL Spreadsheet 2026.xlsm'
sheet_name = 'Optimizer BR'

# Function to read the necessary data from Excel
def load_filtered_data_from_excel():
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]
    
    # Read only the necessary range and columns
    df = sheet.range("A1:K" + str(sheet.cells.last_cell.row)).options(pd.DataFrame, header=1, index=False).value
    
    # Filter rows where 'Slate' column is 'Sunday' and 'PROJ' and 'SALARY' are not zero or empty
    df_filtered = df[(df['Slate'] == 'Sunday') & (df['PROJ'].notna()) & (df['PROJ'] > 0) & 
                     (df['SALARY'].notna()) & (df['SALARY'] > 0)]
    
    return df_filtered

# Function to write results back to Excel
def write_results_to_excel(results):
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]
    
    # Clear previous data if necessary
    sheet.range("M8:P16").clear_contents()
    
    # Write results to Excel without headers
    if results:
        sheet.range("M8").value = results

# Load and filter the data
df = load_filtered_data_from_excel()

# Ensure the necessary columns are present
required_columns = ['PLAYER', 'POS', 'PROJ', 'SALARY']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing one or more required columns: {required_columns}")

# Convert data to numeric and handle missing values
df['PROJ'] = pd.to_numeric(df['PROJ'], errors='coerce').fillna(0)
df['SALARY'] = pd.to_numeric(df['SALARY'], errors='coerce').fillna(0)

# Number of rows in the DataFrame
num_rows = len(df)

# Define the problem
prob = LpProblem("FantasyFootball", LpMaximize)

# Define variables
A = [LpVariable(f"A_{i}", cat="Binary") for i in range(num_rows)]

# Define the objective function: maximize total projection
prob += lpSum(A[i] * df['PROJ'].iloc[i] for i in range(num_rows)), "Total Projection"

# Define constraints

# Constraint: Total salary <= 60,000
prob += lpSum(A[i] * df['SALARY'].iloc[i] for i in range(num_rows)) <= 60000, "Total Salary Constraint"

# Constraint: Exactly 9 players
prob += lpSum(A[i] for i in range(num_rows)) == 9, "Number of Players Constraint"

# Constraints by position
prob += lpSum(A[i] for i in range(num_rows) if df['POS'].iloc[i] == 'QB') == 1, "QB Constraint"
prob += lpSum(A[i] for i in range(num_rows) if df['POS'].iloc[i] == 'TE') >= 1, "TE Constraint"
prob += lpSum(A[i] for i in range(num_rows) if df['POS'].iloc[i] == 'DST') == 1, "DST Constraint"
prob += lpSum(A[i] for i in range(num_rows) if df['POS'].iloc[i] == 'RB') >= 2, "RB Constraint (min 2)"
prob += lpSum(A[i] for i in range(num_rows) if df['POS'].iloc[i] == 'WR') >= 3, "WR Constraint (min 3)"

# Additional constraints to ensure that exactly one FLEX player (RB, WR, or TE) is used
# and that exactly 9 players are selected
prob += lpSum(A[i] for i in range(num_rows) if df['POS'].iloc[i] in ['RB', 'WR', 'TE']) >= 1, "At least one FLEX Constraint"

# Solve the problem
try:
    prob.solve()

    # Output the results
    total_projection = prob.objective.value()
    total_salary = sum(A[i].varValue * df['SALARY'].iloc[i] for i in range(num_rows))
    selected_players = [(df['PLAYER'].iloc[i], df['POS'].iloc[i], df['SALARY'].iloc[i], df['PROJ'].iloc[i]) 
                        for i in range(num_rows) if A[i].varValue == 1]

    # Ensure to have at least 1 QB and 1 DST
    if not any(player[1] == 'QB' for player in selected_players):
        raise ValueError("The lineup does not include a QB.")
    if not any(player[1] == 'DST' for player in selected_players):
        raise ValueError("The lineup does not include a DST.")

    # Assign FLEX to the 3rd RB, 4th WR, or 2nd TE
    rb_count = wr_count = te_count = 0
    for i in range(len(selected_players)):
        pos = selected_players[i][1]
        if pos == 'RB':
            rb_count += 1
            if rb_count == 3:
                selected_players[i] = (selected_players[i][0], 'FLEX', selected_players[i][2], selected_players[i][3])
        elif pos == 'WR':
            wr_count += 1
            if wr_count == 4:
                selected_players[i] = (selected_players[i][0], 'FLEX', selected_players[i][2], selected_players[i][3])
        elif pos == 'TE':
            te_count += 1
            if te_count == 2:
                selected_players[i] = (selected_players[i][0], 'FLEX', selected_players[i][2], selected_players[i][3])

    # Define the order of positions
    position_order = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']

    # Sort players by the defined order
    def get_position_rank(pos):
        return position_order.index(pos) if pos in position_order else len(position_order)
    
    sorted_players = sorted(selected_players, key=lambda x: get_position_rank(x[1]))

    # Prepare results without headers
    results = [[player[0], player[1], player[2], player[3]] for player in sorted_players]

    # Write results back to Excel
    write_results_to_excel(results)

    print(f"Total Projection: {total_projection}")
    print(f"Total Salary: {total_salary}")
    print("Selected Players (Ordered by position):")
    for player in sorted_players:
        print(f"Player: {player[0]}, Position: {player[1]}, Salary: {player[2]}, Projection: {player[3]}")

except Exception as e:
    print(f"An error occurred: {e}")
