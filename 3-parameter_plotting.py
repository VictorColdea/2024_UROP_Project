import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

# Aesthetics :)
import matplotlib as mpl # For font size
mpl.rcParams.update({'font.size': 20})  # Adjust the font size as needed
from matplotlib.ticker import MaxNLocator # For integer y-axis ticks

###############~~~~~ C O D E ~~~~###############
def how_many_junctions():
    '''Function to determine whether only the long sweep or all sweeps should be analysed.'''
    # List all files in the folder
    files = os.listdir(folder_path)
    # Extract the numeric parts from filenames
    numeric_parts = [int(f.split('-')[0][1:]) for f in files if f.endswith('.csv') and f.startswith('J')]
    print(f"Available junctions to analyze are: {numeric_parts}")
    # User input
    how_many_junc = int(input("Would you like to only look at one junction [1], some junctions[2], or all junctions[3]? "))
    if how_many_junc == 1:
        print("You've chosen just one junction!")
        junction_num = int(input("Which junction would you like to include? "))
        junction_nums = [junction_num]
    elif how_many_junc == 2:
        junctions_range_string = input("Which junctions would you like to include? Put them in the form 1,5-8,9 .etc\n")
        junction_nums = string_nums_convert_to_range(junctions_range_string)
    elif how_many_junc == 3:
        junction_nums = numeric_parts# all junctions 
    else:
        print("Pick 1, 2 or 3 please. Restart the program.") 
        quit()
    junction_nums.sort()
    return junction_nums

def create_dict_of_sweeps_to_exclude_from_csv():
    '''Create dictionary of the sweeps to exclude from the appropriate csv file'''
    global junc_sweep_exclude_dict
    # Filepath of the sweep-junction data
    file_path = f"{current_dir}/Chip#{chip_name}-{data_prefix}/Excluded_Sweeps/excluded_sweeps.csv"    
    # Convert file to a dataframe
    df = pd.read_csv(file_path, header=None)
    # Convert to dictionary, with junction numbers being the keys and the corresponding sweep numbers being the values
    junc_sweep_exclude_dict = pd.Series(df[1].values, index=df[0]).to_dict()
    ## This way of seeing which sweeps to exclude means the dictionary is not created each time, and so saves time.

def extract_sweeps_to_exclude_from_dict(junction_num):
    '''Selecting junction from dictionary, and extracting the sweeps to exclude from this.'''
    # Extract sweeps corresponding to correct junction
    sweeps_to_exclude_string = junc_sweep_exclude_dict[f"J{junction_num}"]
    # Convert string to range
    sweeps_to_exclude = string_nums_convert_to_range(sweeps_to_exclude_string[1:])
    # Converts sweep numbers from string to integer, and removes 'S' from sweeps_to_analyze_string 
    print(f"Sweeps to exclude corresponding to this junction are: {sweeps_to_exclude}")
    return sweeps_to_exclude

def extract_param_data(csv_file, junction_num):
    '''Function to extract parameter data from .csv'''
    # Initialise header 
    header_names = ['SweepNum','Phi_B','m*','alpha','SweepType']
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, header=None, names=header_names)
    df_unique = df.drop_duplicates()
    # Sweeps to exclude:
    try:
        sweeps_to_exclude = extract_sweeps_to_exclude_from_dict(junction_num)
        df_filtered = df_unique[~df_unique['SweepNum'].isin(sweeps_to_exclude)] # Without sweeps to exclude
        # df_excluded = df_unique[df_unique['SweepNum'].isin(sweeps_to_exclude)]  # With sweeps to exclude (here for debugging purposes)
    except KeyError as e:
        print(f"{e}-excluded_sweeps.csv not found.")
        df_filtered = df_unique
        pass
    # Print the parameter being analysed
    print(f'Analysing param \"{parameters[i]}\", at junction {junction_num}...')
    # Separate sweeps in accordance with SweepType
    both_df = df_filtered[df_filtered['SweepType'] == 'both']
    # Convert the DataFrames to arrays (list of lists)
    both_array = np.array(both_df.values.tolist())
    return both_array.T # Swap rows and columns

def string_nums_convert_to_range(input_range):
    '''Returns range string(e.g "1-6,8") as a list of every number implicated in that range (e.g. [1,2,3,4,5,6,8]).'''
    list_full = []
    # Split input first by commas
    input_range_commas = input_range.replace(";",",")
    range_segments = input_range_commas.split(',')
    for segment in range_segments:
        # Check if the segment is a range (i.e. 1-5)
        if '-' in segment:
            start_range, end_range = segment.split('-')
            # print(f"start_range: {start_range}, end_range: {end_range}")
            list_full.extend(range(int(start_range), int(end_range)+1))
        else:
            list_full.append(int(segment))
    return list_full

def make_figures_save_multiple_junc(junction_nums):
    '''Function to save figures in a new folder for sanity check.'''
    # Define a folder relative to the current script directory
    folder_figures_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}/Parameter_Plots")
    # If folder doesn't exist yet, create it
    if not os.path.exists(folder_figures_path):
        os.makedirs(folder_figures_path)
    ## Create a figure
    # Initialise bins
    num_bins = 21
    preset_range = [(0,2.5),(0,25),(0.4,0.6)]
    xlimits = preset_range
    bin_width = (preset_range[i][1]-preset_range[i][0])/(num_bins-1)
    print(f"Bin width = {bin_width}")

    # Histogram plots
    sum_hists = np.zeros(num_bins-1)
    plt.figure(figsize=(16,9)) # Create a new figure that is big enough to see titles
    for junction_num in junction_nums:
        datafile_name = f'{folder_path}/J{junction_num}-extracted_parameters.csv'
        try: # Try to extract the data
            both_array = extract_param_data(datafile_name, junction_num) 
        except ValueError as e: # If there's an error, print it and continue to the next junction
            print(f"ValueError: {e}")
            continue
        # Define the number of bins
        bins = np.linspace(*preset_range[i], num_bins)
        try: # Try to extract the data
            int_both_array = both_array[i+1].astype(float)
        except IndexError as e: # If there's an error, print it and continue to the next junction
            print(f"IndexError: {e}")
            print(f"Size of both_array: {both_array.shape}")
            continue
        # Calculate the histogram data for each dataset
        hist_data, _ = np.histogram(int_both_array, bins=bins, weights=None)
        # Stack the histograms
        plt.bar(bins[:-1], hist_data, width=np.diff(bins), bottom=sum_hists, color=colors_dict[junction_num], label=f'J{junction_num}')
        sum_hists = sum_hists + hist_data
    # Print histogram data for debugging
    print(f"sum_hists after junction {junction_num} is: {sum_hists}")
    x_points = np.arange(preset_range[i][0], preset_range[i][1], bin_width)
    popt = fit_gaussian(sum_hists, x_points, xlimits[i])
    # Aesthetics
    plt.xlim(None)#list(xlimits[i]))
    plt.ylim([None,max(sum_hists)+1])
    plt.ylabel("Frequency")
    plt.xlabel(parameter)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) # Make y-axis have integer ticks
    plt.legend()#loc='lower right')
    plt.title(f"Variation of {parameter}: $\mu$ = {round(popt[0],2)}, $\sigma$ = {round(np.sqrt(popt[1]),2)}")
    plt.suptitle(f"Chip#{chip_name}: Dataset {data_prefix} @ {temperature}K") # Over-arching title for the whole figure
    # Define the full file path, including the folder and file name
    plot_file_path = os.path.join(folder_figures_path, f"{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K -J{[j_num for j_num in junction_nums]}-parameter{i+1}.png")
    # Save the figure to the specified folder
    plt.savefig(plot_file_path, dpi=300)
    # Close the figure so a new one can be created next time
    plt.close()
    print("Next parameter")

def plot_phiB_vs_m():
    '''Plot Phi_B vs m*'''
    # Extract phiB and m* columns from dataframe.
    csv_file = f'Chip#{chip_name}-{data_prefix}/Extracted_Parameters/extracted_parameters.csv'
    # Initialise header 
    header_names = ['SweepNum','Phi_B','m*','alpha','SweepType']
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, header=None, names=header_names)
    df_unique = df.drop_duplicates()
    # Separate out sweeps_to_exclude
    sweeps_to_exclude = list(junc_sweep_exclude_dict.values())
    df_filtered = df_unique[~df_unique['SweepNum'].isin(sweeps_to_exclude)] # Without sweeps to exclude
    # Separate sweeps in accordance with SweepType
    both_df = df_filtered[df_filtered['SweepType'] == 'both']
    # Take out relevant columns
    sweep_nums, phiB_data, effm_data = np.array(both_df['SweepNum'].tolist()), np.array(both_df['Phi_B'].tolist()), np.array(both_df['m*'].tolist())
    # # Filter data to remove outliers
    # Define a condition to ignore the points with Φ_B > 10
    condition = (phiB_data <= 10)
    # Filter the data
    effm_filtered = effm_data[condition]
    phiB_filtered = phiB_data[condition]
    sweep_nums_filtered = sweep_nums[condition]
    # Plot
    mpl.rcParams.update({'font.size': 24})
    plt.figure(figsize=(16,9)) # Create a new figure that is full-screen size
    plt.scatter(1/effm_filtered, phiB_filtered, s=100, color='red') # Plot data
    plt.title("$\\Phi_B$ vs 1/m*")
    plt.suptitle(f"Chip#{chip_name}: Dataset {data_prefix} @ {temperature}K")
    plt.ylabel("$\\Phi_B$(eV)")
    plt.xlabel("1/m*")
    ####  SAVE FIG
    # Define a folder relative to the current script directory
    folder_figures_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}/Parameter_Plots")
    # If folder doesn't exist yet, create it
    if not os.path.exists(folder_figures_path):
        os.makedirs(folder_figures_path)
    # Define the full file path, including the folder and file name
    plot_file_path = os.path.join(folder_figures_path, f"{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-PhiB-vs-m_eff.png")
    # Save the figure to the specified folder
    plt.savefig(plot_file_path, dpi=300)
    plt.close()
    return sweep_nums_filtered, phiB_filtered, effm_filtered

def average_phiB_m_proportionality_coeff(sweep_nums, phiB, m_eff):
    '''Average the proportionality coefficient between Phi_B and m* per junction'''
    # Create an array of junction numbers corresponding to the sweeps
    junction_sweep_nums = np.array([sweep_junc_dict[sweep_num] for sweep_num in sweep_nums])
    # Create an array to store the average Phi_B m* for each junction
    array_phiB_m_eff = []
    # Run through each junction number
    for junction_num in junction_nums:
        # Find the indices of the sweeps corresponding to the junction number
        indices_to_average = np.where(junction_sweep_nums == junction_num)[0]
        if len(indices_to_average) == 0: # If there are no sweeps for this junction
            # print(f"Junction {junction_num} has no data to average.") # Debugging purposes
            continue
        relevant_phiB = phiB[indices_to_average] # Extract the relevant data
        relevant_m_eff = m_eff[indices_to_average]
        relevant_phiB_m_eff = relevant_phiB * relevant_m_eff
        average_phiB_m_eff = np.mean(relevant_phiB_m_eff) # Average the data (spread is ignored)
        # print(f"Average Phi_B*m* for junction {junction_num} is: {average_phiB_m_eff}")
        array_phiB_m_eff.append(average_phiB_m_eff) # Append value to the array, to be plotted
    plot_phiB_m_proportionality_coeff(array_phiB_m_eff)

def plot_phiB_m_proportionality_coeff(average_phiB_m_eff):
    '''Plot the average (per junction) proportionality coefficient between Phi_B and m* in histograms'''
    # Plot
    mpl.rcParams.update({'font.size': 24})
    plt.figure(figsize=(16,9)) # Create a new figure that is full-screen size
    plt.hist(average_phiB_m_eff, bins=25, color='red', edgecolor='black', alpha=.7) # Plot data
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("$\\Phi_B$m* proportionality coefficient")
    plt.suptitle(f"Chip#{chip_name}: Dataset {data_prefix} @ {temperature}K")
    plt.ylabel("Frequency")
    plt.xlabel("$\\Phi_B$m* (eV)")
    ####  SAVE FIG
    # Define a folder relative to the current script directory
    folder_figures_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}/Parameter_Plots")
    # If folder doesn't exist yet, create it
    if not os.path.exists(folder_figures_path):
        os.makedirs(folder_figures_path)
    # Define the full file path, including the folder and file name
    plot_file_path = os.path.join(folder_figures_path, f"{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature} K-PhiB-m_eff-proportionality.png")
    # Save the figure to the specified folder
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

def junc_sweep_dict_creation():
    '''Create dictionary of junction-sweep correspondence.'''
    # Filepath of the sweep-junction data
    file_path = f"{current_dir}/Chip#{chip_name}-{data_prefix}/{data_prefix}-Chip#{chip_name}-junction_sweep_correspondence.csv"
    # Convert file to a dataframe
    df = pd.read_csv(file_path, header=None)
    # Convert to dictionary
    junc_sweep_dict = pd.Series(df[1].values, index=df[0]).to_dict()
    # Convert the values from a string to a list of integers
    junc_sweep_dict_int = {int(key[1:]): string_nums_convert_to_range(value[1:]) for key, value in junc_sweep_dict.items()}
    # Create a new dictionary where values are now keys
    swapped_dict = {value: key for key, values in junc_sweep_dict_int.items() for value in values}
    # Sort the dictionary by key
    sorted_swapped_dict = {key: swapped_dict[key] for key in sorted(swapped_dict)}
    return sorted_swapped_dict

def gaussian_equation(x, mean, var=0.4, A=5):
    '''Equation for a Gaussian'''
    return A * np.exp(-1/2*(x-mean)**2/(var))

def fit_gaussian(y_data, x_data, x_limits):
    '''Plot a fitted Gaussian on the graph'''
    # Initial guess for the parameters: mean, std deviation, amplitude
    initial_guess = [[1.5, 0.08, 2], [8, 2, 5], [0.5, 0.001, 2]]
    # Fit the data to a Gaussian
    popt, _ = curve_fit(gaussian_equation, x_data, y_data, bounds = ([0, 0, 0], [50, np.inf, np.inf]), p0=initial_guess[i], maxfev=10000)
    print(f"Extracted Gaussian parameters. Mean, Variance, Amplitude: {popt}") # Print the parameters
    # Plot the original data and the Gaussian fit
    # plt.scatter(x_data, y_data, label='Original data')  # Plot original data
    plt.plot(np.linspace(*x_limits,500), gaussian_equation(np.linspace(*x_limits,500), *popt), 'k-', linewidth=2)  # Plot Gaussian fit on top. label=f"$\mu$ = {round(popt[0],2)}",
    # plt.show()
    return popt

def initialise_constants():
    '''Initialise all values which are constant for the duration of the program'''
    global current_dir, chip_name, data_prefix, folder_path, junction_nums, temperature
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get chip name and data prefix
    chip_name = input("What's the chip name? (e.g. 35-C5) ")
    data_prefix = input("What's the data prefix? (e.g. AA or BB etc) ")
    temperature = input("What's the temperature(in K)? (e.g. 77 or 300) ")
    folder_path = f'Chip#{chip_name}-{data_prefix}/Extracted_Parameters'
    create_dict_of_sweeps_to_exclude_from_csv()
    junction_nums = how_many_junctions()

#### CODE
# Dictionary of colors for plotting junction numbers
colors_dict = {1: "#f6192B", 2: "#1c742b", 3: "#aaffc3", 4: "#4363d8", 5: "#f58231",  # Red, Dark Green, Light Mint, Blue, Orange
               6: "#911eb4", 7: "#42d4f4", 8: "#32CD32", 9: "#bfef45", 10: "#800000",  # Purple, Light Blue, Green, Lime, Maroon
               11: "#469990", 12: "#caaade", 13: "#9A6324", 14: "#ff6347", 15: "#daa8b9",  # Teal, Light Purple, Brown, Tomato, Pink
               16: "#ffe119", 17: "#808000", 18: "#dfbf9f", 19: "#8a2be2", 20: "#a9a9a9",}  # Yellow, Olive, Peach, Blue Violet, Dark Gray

initialise_constants()
# Plot Phi_B vs m*
sweepnums, phiB, meff = plot_phiB_vs_m()
# Plot the proportionality coefficient between Phi_B and m*
sweep_junc_dict = junc_sweep_dict_creation()
average_phiB_m_proportionality_coeff(sweepnums, phiB, meff)
# Go through each parameter and plot histograms
parameters = ['$\\Phi_B$(eV)','m*','$\\alpha$']
for i, parameter in enumerate(parameters):
    make_figures_save_multiple_junc(junction_nums)

plots_folder_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}\\Parameter_Plots")
plots_file_name = f"{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K -J{[j_num for j_num in junction_nums]}-parameterX.png"
print(f"\nParameter plots have been created and can be found in {plots_folder_path}, \nunder the names of {plots_file_name}")
