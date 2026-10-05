import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import xlwings as xw  # Import xlwings

# Load the Excel file
file_path = r'C:\Users\hlee145\Documents\FanDuel Spreadsheets\FanDuel NFL Spreadsheet 2026.xlsm'
sheet_name = 'Optimizer BR'

# Function to read the necessary data from Excel and apply the checkbox filters
def load_filtered_data_from_excel():
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]
    
    # Read the data from the sheet
    df = sheet.range("A1").expand('table').options(pd.DataFrame, header=1, index=False).value

    # --- existing checkbox groups (unchanged) ---
    # Get the checkbox values for conditions on column '%'
    checkbox_73 = sheet.api.CheckBoxes("Check Box 73").Value  # 1 for checked, 0 for unchecked
    checkbox_74 = sheet.api.CheckBoxes("Check Box 74").Value
    checkbox_75 = sheet.api.CheckBoxes("Check Box 75").Value

    # Get the checkbox values for including 1, 2, or 3 players from the filtered group
    checkbox_76 = sheet.api.CheckBoxes("Check Box 76").Value  # 1 for checked, 0 for unchecked
    checkbox_77 = sheet.api.CheckBoxes("Check Box 77").Value
    checkbox_78 = sheet.api.CheckBoxes("Check Box 78").Value

    # ✅ NEW: get value of Check Box 1397 (stack QB + WR same team)
    try:
        checkbox_1397 = sheet.api.CheckBoxes("Check Box 1397").Value
    except Exception:
        checkbox_1397 = 0  # fallback if checkbox is missing

    # Apply filters based on column '%' (for checkboxes 73-75)
    condition_players = pd.DataFrame()
    if checkbox_73 == 1:
        condition_players = df[df['%'] < 5]  # Players meeting condition of checkbox 73
    elif checkbox_74 == 1:
        condition_players = df[df['%'] < 10]  # Players meeting condition of checkbox 74
    elif checkbox_75 == 1:
        condition_players = df[df['%'] < 15]  # Players meeting condition of checkbox 75

    # Further filter where 'PROJ' and 'SALARY' are valid
    df_filtered = df[(df['PROJ'].notna()) & (df['PROJ'] > 0) &
                     (df['SALARY'].notna()) & (df['SALARY'] > 0)]

    # If checkboxes 76-78 are checked, include players based on the specific requirement
    num_players_to_include = 0
    if checkbox_76 == 1:
        num_players_to_include = 1
    elif checkbox_77 == 1:
        num_players_to_include = 2
    elif checkbox_78 == 1:
        num_players_to_include = 3

    # If any of checkboxes 76-78 are checked, limit the inclusion of players
    if num_players_to_include > 0 and not condition_players.empty:
        condition_players = condition_players.head(num_players_to_include)
        df_filtered = pd.concat([df_filtered, condition_players]).drop_duplicates()

    # Check if checkbox 1039 is checked
    checkbox_1039 = sheet.api.CheckBoxes("Check Box 1039").Value
    
    players_to_include = []

    # List of checkbox numbers
    checkbox_numbers = list(range(82, 433))

    # Proceed to checkboxes 82 to 1036 only if checkbox 1039 is checked
    if checkbox_1039 == 1:
        for i in checkbox_numbers:
            checkbox_name = f"Check Box {i}"
            try:
                checkbox = sheet.api.CheckBoxes(checkbox_name)
                checkbox_value = checkbox.Value  # 1 for checked, 0 for unchecked
                if checkbox_value == 1:
                    # Get the row of the checkbox
                    row = checkbox.TopLeftCell.Row
                    # Get the player name from column B at that row
                    player_name = sheet.range(f'B{row}').value
                    players_to_include.append(player_name)
            except Exception as e:
                # Skip if the checkbox does not exist
                continue

    # --- NEW: day-based exclusion checkboxes (45-48) ---
    # Map of checkboxes to their T-cell (where the day text lives)
    cb_cell_map = {
        "Check Box 45": "T5",  # Friday
        "Check Box 46": "T2",  # Thursday
        "Check Box 47": "T3",  # Sunday
        "Check Box 48": "T4",  # Monday
    }

    days_checked = []  # list of day strings (e.g. "Sunday", "Thursday") from the T cells
    for cb_name, cell_addr in cb_cell_map.items():
        try:
            if sheet.api.CheckBoxes(cb_name).Value == 1:
                day_text = sheet.range(cell_addr).value
                if day_text is not None:
                    days_checked.append(str(day_text).strip())
        except Exception:
            # If a checkbox/cell is missing, skip silently
            continue

    # Detect slate column (prefer common names then fall back to column K)
    slate_col = None
    for candidate in ['Slate', 'SLATE', 'slate', 'GameDay', 'Day']:
        if candidate in df.columns:
            slate_col = candidate
            break
    if slate_col is None:
        # use column K (11th column) if available
        if len(df.columns) >= 11:
            slate_col = df.columns[10]
        else:
            slate_col = None

    players_to_exclude_by_day = []
    if slate_col is not None and days_checked:
        # match by startswith (case-insensitive) so "Sunday" matches "Sunday GPP" etc.
        slate_ser = df[slate_col].astype(str).fillna('').str.strip()
        for day in days_checked:
            if not day:
                continue
            mask = slate_ser.str.lower().str.startswith(str(day).lower())
            players_found = df.loc[mask, 'PLAYER'].tolist()
            players_to_exclude_by_day.extend(players_found)

    # Deduplicate
    players_to_exclude_by_day = list(dict.fromkeys(players_to_exclude_by_day))

    return (df_filtered, players_to_include, players_to_exclude_by_day,
            checkbox_73, checkbox_74, checkbox_75, checkbox_76, checkbox_77, checkbox_78, checkbox_1397)


# Function to write results back to Excel (excluding the % value)
def write_results_to_excel(results):
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]
    
    # Clear previous data if necessary
    sheet.range("M8:P16").clear_contents()
    
    # Write results to Excel without headers and excluding % value
    if results:
        # Only include player name, position, salary, and projection in the results
        results_to_write = [(player[0], player[1], player[2], player[3]) for player in results]
        sheet.range("M8").value = results_to_write

# Load and filter the data
(df_filtered, players_to_include, players_to_exclude_by_day,
 checkbox_73, checkbox_74, checkbox_75, checkbox_76, checkbox_77, checkbox_78, checkbox_1397) = load_filtered_data_from_excel()

# Ensure the necessary columns are present
required_columns = ['PLAYER', 'POS', 'PROJ', 'SALARY', '%']
if not all(col in df_filtered.columns for col in required_columns):
    raise ValueError(f"Missing one or more required columns: {required_columns}")

# Convert data to numeric and handle missing values
df_filtered['PROJ'] = pd.to_numeric(df_filtered['PROJ'], errors='coerce').fillna(0)
df_filtered['SALARY'] = pd.to_numeric(df_filtered['SALARY'], errors='coerce').fillna(0)

# Reset index
df_filtered = df_filtered.reset_index(drop=True)
num_rows = len(df_filtered)

# ✅ bring in DFF sheet to map players → team
wb = xw.Book(file_path)
dff_sheet = wb.sheets['DFF']
dff_df = dff_sheet.range("A1").expand('table').options(pd.DataFrame, header=1, index=False).value
name_to_team = dict(zip(dff_df['Name'], dff_df['team']))
df_filtered['TEAM'] = df_filtered['PLAYER'].map(name_to_team)

# Define the problem
prob = LpProblem("FantasyFootball", LpMaximize)

# Define variables
A = [LpVariable(f"A_{i}", cat="Binary") for i in range(num_rows)]

# Define the objective function: maximize total projection
prob += lpSum(A[i] * df_filtered['PROJ'].iloc[i] for i in range(num_rows)), "Total Projection"

# Define constraints

# Constraint: Total salary <= 60,000
prob += lpSum(A[i] * df_filtered['SALARY'].iloc[i] for i in range(num_rows)) <= 60000, "Total Salary Constraint"

# Constraint: Exactly 9 players
prob += lpSum(A[i] for i in range(num_rows)) == 9, "Number of Players Constraint"

# Constraints by position
prob += lpSum(A[i] for i in range(num_rows) if df_filtered['POS'].iloc[i] == 'QB') == 1, "QB Constraint"
prob += lpSum(A[i] for i in range(num_rows) if df_filtered['POS'].iloc[i] == 'TE') >= 1, "TE Constraint"
prob += lpSum(A[i] for i in range(num_rows) if df_filtered['POS'].iloc[i] == 'DST') == 1, "DST Constraint"
prob += lpSum(A[i] for i in range(num_rows) if df_filtered['POS'].iloc[i] == 'RB') >= 2, "RB Constraint (min 2)"
prob += lpSum(A[i] for i in range(num_rows) if df_filtered['POS'].iloc[i] == 'WR') >= 3, "WR Constraint (min 3)"

# Additional constraints to ensure that exactly one FLEX player (RB, WR, or TE) is used
prob += lpSum(A[i] for i in range(num_rows) if df_filtered['POS'].iloc[i] in ['RB', 'WR', 'TE']) >= 1, "At least one FLEX Constraint"

# ✅ NEW: Stacking QB + WR constraint if 1397 is checked
if checkbox_1397 == 1:
    teams = df_filtered['TEAM'].dropna().unique()
    for team in teams:
        qb_indices = df_filtered.index[(df_filtered['POS'] == 'QB') & (df_filtered['TEAM'] == team)].tolist()
        wr_indices = df_filtered.index[(df_filtered['POS'] == 'WR') & (df_filtered['TEAM'] == team)].tolist()
        if qb_indices and wr_indices:
            prob += lpSum(A[i] for i in wr_indices) >= lpSum(A[i] for i in qb_indices), f"QB_WR_Stack_{team}"

# Additional constraints for checkboxes 73-75
if checkbox_73 == 1:
    if checkbox_76 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 5) >= 1, "At least 1 < 5% Constraint"
    if checkbox_77 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 5) >= 2, "At least 2 < 5% Constraint"
    if checkbox_78 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 5) >= 3, "At least 3 < 5% Constraint"

if checkbox_74 == 1:
    if checkbox_76 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 10) >= 1, "At least 1 < 10% Constraint"
    if checkbox_77 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 10) >= 2, "At least 2 < 10% Constraint"
    if checkbox_78 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 10) >= 3, "At least 3 < 10% Constraint"

if checkbox_75 == 1:
    if checkbox_76 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 15) >= 1, "At least 1 < 15% Constraint"
    if checkbox_77 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 15) >= 2, "At least 2 < 15% Constraint"
    if checkbox_78 == 1:
        prob += lpSum(A[i] for i in range(num_rows) if df_filtered['%'].iloc[i] < 15) >= 3, "At least 3 < 15% Constraint"

# Constraint: If a checkbox is checked, the corresponding player must be included in the optimization
for player_name in players_to_include:
    indices = df_filtered.index[df_filtered['PLAYER'] == player_name].tolist()
    if indices:
        i = indices[0]
        prob += A[i] == 1, f"Include {player_name} Constraint"
    else:
        print(f"Player {player_name} not found in filtered data.")

# NEW: apply exclude-by-day constraints (but don't override explicit includes)
for player_name in players_to_exclude_by_day:
    if player_name in players_to_include:
        # If the same player was explicitly included, skip exclusion and warn
        print(f"Warning: player {player_name} is both in include-list and in day-exclude list. Keeping include.")
        continue
    indices = df_filtered.index[df_filtered['PLAYER'] == player_name].tolist()
    if indices:
        for i in indices:
            prob += A[i] == 0, f"Exclude {player_name} Constraint"
    else:
        print(f"Player {player_name} from day-exclude list not found in filtered data.")

# Solve the problem
try:
    prob.solve()

    # Output the results
    total_projection = prob.objective.value()
    total_salary = sum(A[i].varValue * df_filtered['SALARY'].iloc[i] for i in range(num_rows))
    selected_players = [(df_filtered['PLAYER'].iloc[i], df_filtered['POS'].iloc[i], df_filtered['SALARY'].iloc[i], df_filtered['PROJ'].iloc[i]) 
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
    print(f"Error: {str(e)}")
