import pandas as pd # For creating dataframe
import matplotlib.pyplot as plt # For data representation/graph plotting
import numpy as np
from scipy.optimize import curve_fit # For curve fitting
import scipy.constants as cnst # For scientific constants
import csv # For writing extracted parameters to csv 
import os

# Plot aesthetics :)
import matplotlib as mpl # For font size
mpl.rcParams.update({'font.size': 22})  # Adjust the font size as needed

###############~~~~~ C O D E ~~~~~###############
def string_nums_convert_to_range(input_range):
    '''Returns range (e.g "1-6,8") as a list of every number (as integers) implicated in that range.'''
    list_full = []
    # Split input first by commas
    range_segments = input_range.split(',')
    for segment in range_segments:
        # Check if the segment is a range (i.e. 1-5)
        if '-' in segment:
            start_range, end_range = segment.split('-')
            list_full.extend(range(int(start_range), int(end_range)+1))
        else:
            list_full.append(int(segment))
    return list_full

def extract_sweep_data(sweep_num):
    '''Function to extract sweep voltage and current data from .csv. Takes in sweep_num to know which datafile to read.'''
    # Initialise header 
    header_names = ['<ChA>', 'Current (nA)', 'Time (s)', 'Voltage (V)',	'MAG1 (nA)', 'MAG2 (nA)', 'X1 (nA)', 'X2 (nA)', 'Y1 (nA)', 'Y2 (nA)', 'Theta1',	'Theta2', 'Leakage Current (nA)', 'Gate Voltage (V)']
    # NOTE: Edit filename as necessary
    datafile_name = f'Data_Processing/Data_vic/Chip#{chip_name}/CSV-data_files/{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-S{sweep_num}.csv'
    try:
        # Try to create the dataframe from the csv file which needs to be created beforehand.
        df = pd.read_csv(datafile_name, header=None, names=header_names)
    except FileNotFoundError:
        print(f'File not found at\n Chip#{chip_name}/CSV-data_files/{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-S{sweep_num}.csv')
        print("Make sure this file path exists!")
    # Take out columns of interest
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df['<ChA>'], df['Current (nA)']

# Curve fitting equation
def Simmons_equation_alpha(V, Phi_B=5, m=1, alph=0.5):
    '''Function for the Simmons equation of tunnelling, accouting for asymmetry with the alpha term.'''
    # Defining equation constants
    q, e, h_bar, d, m_e = cnst.e, cnst.e, cnst.hbar, 1e-9, cnst.m_e
    return q*e/(4*np.pi**2*h_bar*d**2) * (((Phi_B-1/2*V*alph)*(np.exp(-2*d*np.sqrt(2*m*m_e*e)/h_bar * np.sqrt(Phi_B-1/2*V*alph)))) - ((Phi_B+1/2*V*(1-alph))*(np.exp(-2*d*np.sqrt(2*m*m_e*e)/h_bar * np.sqrt(Phi_B+1/2*V*(1-alph))))))
    # junction_area*1e7

def extract_parameters(sweep_voltage, current):
    '''Function to extract parameters from Simmons equation.'''
    return curve_fit(Simmons_equation_alpha, np.array(sweep_voltage), current, bounds = ([0, 0, 0], [np.inf, np.inf, 1]), maxfev=5000)

def make_figures_save(sweep_voltage, current, params, sweep_num):
    '''Function to save figures in a new folder for sanity check.'''
    # Define a folder relative to the current script directory
    folder_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}/Figure_Fits/J{junction_num}")
    # If folder doesn't exist yet, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    ## Create a figure
    plt.figure(figsize=(16,9)) # create a new figure that is full-screen size
    plt.plot(sweep_voltage, Simmons_equation_alpha(sweep_voltage, *params), label=f'Fitted curve: $\\Phi_B$, m*, $\\alpha$ = {round(params[0],2)}eV, {round(params[1],1)}m$_e$, {round(params[2],3)}', color='red') # plot of fit
    plt.plot(sweep_voltage, current, label=f'Raw Data') # plot of raw data
    plt.title(f"{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-J{junction_num}-S{sweep_num}")
    plt.ylabel("Current(nA)")
    plt.xlabel("Sweeping Voltage(V)")
    plt.legend()
    # Define the full file path, including the folder and file name
    file_path = os.path.join(folder_path, f"{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-J{junction_num}-S{sweep_num}.png")
    # Save the figure to the specified folder
    plt.savefig(file_path, dpi=300)
    # Close the figure so a new one can be created next time
    plt.close()

def sweep_size_determiner(sweep_voltage):
    '''Function to determine whether the sweep is purely positive, purely negative, or both.''' 
    # Find the minimum and maximum voltage of the sweep
    voltage_max, voltage_min = max(sweep_voltage), min(sweep_voltage)
    # Determine the type of sweep
    if voltage_min == 0: # (this still includes sweeps that DNF)
        sweep_type = 'positive'
    elif voltage_max == 0: # if voltage max = 0
        sweep_type = 'negative'
    elif voltage_max == -voltage_min:
        if voltage_max < 0.4: # if the sweep is small
            sweep_type = 'small' # small sweep
        else:
            sweep_type = 'both' # both/long/middle sweeps. Refers to the ones that sweep the entire range
    else:
        sweep_type = 'DNF' # DNF = Did Not Finish
    return sweep_type

def write_parameters_to_csv(params, sweep_type):
    '''Writes the extracted parameters to a dedicated .csv file.'''
    # Make the folder for it; tidy things up a bit
    folder_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}/Extracted_Parameters")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Send to csv
    row_for_csv = [sweep_num, *np.round(params, 3), sweep_type]
    # Open file in append mode ('a') and write new rows in each iteration 
    # Write to junction-specific CSV, but also to chip-specific csv
    with open(f'{folder_path}/J{junction_num}-extracted_parameters.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_for_csv)
    
    with open(f'{folder_path}/extracted_parameters.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_for_csv)

def write_excluded_sweeps_to_csv(sweeps_to_exclude_range):
    '''Writes the excluded sweeps for a particular junction to a dedicated .csv file.'''
    # Make the folder for it; tidy things up a bit
    folder_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}/Excluded_Sweeps")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Row entry for csv
    entry_for_csv = [f"J{junction_num}",f"S{sweeps_to_exclude_range}"]
    # Open file in append mode ('a') and write new rows in each iteration 
    with open(f'{folder_path}/J{junction_num}-excluded_sweeps.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(entry_for_csv)
    # Write to non-junction-specific CSV as well
    with open(f'{folder_path}/excluded_sweeps.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(entry_for_csv)
 
def main(sweep_num):
    '''Main function to run all other functions in this file.'''
    sweep_voltage, current = extract_sweep_data(sweep_num)
    # Extract parameters:
    params, _ = extract_parameters(sweep_voltage, current)
    print(f'S{sweep_num}', params) #sanity check
    # Save figures in other folder for easy access and separation
    sweep_type = sweep_size_determiner(sweep_voltage)
    # Select only long sweeps to plot
    if sweep_type == "both":
        make_figures_save(sweep_voltage, current, params, sweep_num)
    # Write the extracted parameters to a csv file
    write_parameters_to_csv(params, sweep_type)

def print_all_junction_options():
    '''Print all junction options to choose from.'''
    file_path = f"{current_dir}/Chip#{chip_name}-{data_prefix}/{data_prefix}-Chip#{chip_name}-junction_sweep_correspondence.csv"
    df = pd.read_csv(file_path, header=None)
    junctions_df = df[0]
    junctions_list = junctions_df.tolist()
    print("Junctions available to choose from are: ")
    for junction in junctions_list:
        print(f"{junction}", end="\t")
    print(end="\n")

def process_junction_data_from_csv(junction_num):
    '''Read the CSV file into a DataFrame (without predefined column names)'''
    global junc_sweep_dict
    # Filepath of the sweep-junction data
    file_path = f"{current_dir}/Chip#{chip_name}-{data_prefix}/{data_prefix}-Chip#{chip_name}-junction_sweep_correspondence.csv"
    # Convert file to a dataframe
    df = pd.read_csv(file_path, header=None)
    # Convert to dictionary
    junc_sweep_dict = pd.Series(df[1].values, index=df[0]).to_dict()
    # Extract sweeps corresponding to correct junction
    sweeps_to_analyze_string = junc_sweep_dict[f"J{junction_num}"]
    # Converts sweep numbers from string to integer, and removes 'S' from sweeps_to_analyze_string 
    sweeps_to_analyze_list = [int(sweep_num) for sweep_num in sweeps_to_analyze_string[1:].split(";")] 
    print(f"Sweeps corresponding to junction {junction_num} are: {sweeps_to_analyze_list}")
    return sweeps_to_analyze_list

def initialise_constants():
    '''Initialise the variables which are constant for the whole duration of the program.'''
    #NOTE: Here is where to change values for different datasets
    global current_dir, chip_name, data_prefix, junction_areas, temperature
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get chip name and data prefix
    chip_name = input("What's the chip name? (e.g. 35-C5 or 102-D4 etc) ")
    data_prefix = input("What's the data prefix (e.g. AA or BB etc) ")
    temperature = input("What's the temperature (e.g. 77 or 300K)? ")

######################################################################################################################################################
## CODE to run through all relevant sweeps (specified by the user)
initialise_constants()

def how_many_junctions():
    '''Function to determine how many junctions should be analysed.'''
    global junc_sweep_dict, junction_nums
    # Filepath of the sweep-junction data
    file_path = f"{current_dir}/Chip#{chip_name}-{data_prefix}/{data_prefix}-Chip#{chip_name}-junction_sweep_correspondence.csv"
    # Convert file to a dataframe
    df = pd.read_csv(file_path, header=None)
    # Convert to dictionary
    junc_sweep_dict = pd.Series(df[1].values, index=df[0]).to_dict()
    junctions = list(junc_sweep_dict.keys())
    numeric_parts = [int(junc[1:]) for junc in junctions]
    print(numeric_parts)
    how_many_junc = int(input("Would you like to only look at one junction [1], some junctions[2], or all junctions[3]? "))
    if how_many_junc == 1:
        print("You've chosen just one junction!")
        junction_num = int(input("Which junction would you like to include? "))
        junction_nums = [junction_num]
    elif how_many_junc == 2:
        junctions_range_string = input("Which junctions would you like to include? Put them in the form 1,5-8,9 .etc\n")#"2,3,5,8,9,10,12,13"#"1,2,3,4,5,6,7,8,9,10,12,14,16,17,19"#"3,5,8,9,10,11,12,13,14"
        junction_nums = string_nums_convert_to_range(junctions_range_string)
    elif how_many_junc == 3:
        junction_nums = numeric_parts # All junctions 
    else:
        print("Pick 1, 2 or 3 please. Restart the program.") 
        quit()
    junction_nums.sort()

# Decide which junctions
how_many_junctions()
# Analyse each junction
for junction_num in junction_nums:
    # Sweep range for each junction
    sweep_range = process_junction_data_from_csv(junction_num)
    # Analyzes dataset for each sweep
    for sweep_num in sweep_range:
        try:
            main(sweep_num)
        except (ValueError, RuntimeError) as e:
            print(f"Skipping sweep_num {sweep_num} due to: {e}")
            continue
    # Write the excluded sweeps to a csv to automatically be taken into account by the next program (parameter_plotting-XXXXX.py)
    figures_folder_path, satisfied_range = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}\\Figure_Fits\\J{junction_num}"), False
    print(f"\nCheck the images I've just put in {figures_folder_path}, and let me know which sweeps to exclude in further analysis!")
    sweeps_to_exclude_range_commas = input("Would you like to exclude any sweeps from this junction? If so, put them in the form 1-5,8,9-11 .etc \tDefault is none.\n")
    sweeps_to_exclude_range = sweeps_to_exclude_range_commas.replace(",",";")
    # Write the excluded sweeps to a csv. If none, write 0
    if sweeps_to_exclude_range != "":
        write_excluded_sweeps_to_csv(sweeps_to_exclude_range)
    else:
        write_excluded_sweeps_to_csv("0")