import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import xlwings as xw

# Load the Excel file
file_path = r'C:\Users\hlee145\Documents\FanDuel Spreadsheets\FanDuel NFL Spreadsheet 2026.xlsm'
sheet_name = 'Optimizer BR'

# Open the workbook and load the sheet
wb = xw.Book(file_path)
sheet = wb.sheets[sheet_name]

# Load the data
full_df = pd.DataFrame(sheet.range('B2:K500').value)
full_df.columns = ['PLAYER', 'SALARY', 'PROJ', 'POS', 'QB', 'RB', 'WR', 'TE', 'DST', 'Slate']
df = full_df[['PLAYER', 'SALARY', 'PROJ', 'POS', 'Slate']]

# Filter valid rows
df = df[(df['PROJ'].notna()) & (df['SALARY'].notna()) & (df['Slate'].notna())]
df = df[(df['PROJ'] > 0) & (df['SALARY'] > 0) & ((df['Slate'] == 'Thursday') | (df['Slate'] == 'Friday'))]
df['PROJ'] = pd.to_numeric(df['PROJ'], errors='coerce').fillna(0)
df['SALARY'] = pd.to_numeric(df['SALARY'], errors='coerce').fillna(0)

num_rows = len(df)

# Define problem
prob = LpProblem("FantasyFootball", LpMaximize)

# Define variables: MVP[i], UTIL[i]
MVP = [LpVariable(f"MVP_{i}", cat="Binary") for i in range(num_rows)]
UTIL = [LpVariable(f"UTIL_{i}", cat="Binary") for i in range(num_rows)]

# Objective: maximize projections
prob += lpSum(1.5 * MVP[i] * df['PROJ'].iloc[i] + UTIL[i] * df['PROJ'].iloc[i]
              for i in range(num_rows)), "Total Projection"

# Salary constraint (MVP salary counts 1.5x)
prob += lpSum(1.5 * MVP[i] * df['SALARY'].iloc[i] + UTIL[i] * df['SALARY'].iloc[i]
              for i in range(num_rows)) <= 60000, "Total Salary"

# Roster size: exactly 6 players
prob += lpSum(MVP[i] + UTIL[i] for i in range(num_rows)) == 6, "Total Players"

# Exactly 1 MVP
prob += lpSum(MVP[i] for i in range(num_rows)) == 1, "Exactly One MVP"

# Cannot be both MVP and UTIL
for i in range(num_rows):
    prob += MVP[i] + UTIL[i] <= 1, f"RoleConstraint_{i}"

# Solve
try:
    prob.solve()

    total_projection = prob.objective.value()
    total_salary = sum(1.5 * MVP[i].varValue * df['SALARY'].iloc[i] +
                       UTIL[i].varValue * df['SALARY'].iloc[i]
                       for i in range(num_rows))

    selected_players = []
    for i in range(num_rows):
        if MVP[i].varValue == 1:
            selected_players.append((df['PLAYER'].iloc[i], "MVP",
                                     int(1.5 * df['SALARY'].iloc[i]),
                                     round(1.5 * df['PROJ'].iloc[i], 2)))
        elif UTIL[i].varValue == 1:
            selected_players.append((df['PLAYER'].iloc[i], df['POS'].iloc[i],
                                     int(df['SALARY'].iloc[i]),
                                     round(df['PROJ'].iloc[i], 2)))

    # MVP first, then sort by salary
    sorted_players = sorted(selected_players, key=lambda x: (x[1] != 'MVP', -x[2]))

    print(f"Total Projection: {total_projection}")
    print(f"Total Salary: {total_salary}")
    print("Selected Players:")
    for p in sorted_players:
        print(f"Player: {p[0]}, Position: {p[1]}, Salary: {p[2]}, Projection: {p[3]}")

    # Output to Excel
    results = [[p[0], p[1], p[2], p[3]] for p in sorted_players]
    sheet.range("M8").expand('down').clear_contents()
    sheet.range("M8").value = results

except Exception as e:
    print(f"An error occurred: {e}")
