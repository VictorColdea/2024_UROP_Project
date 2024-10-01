import pandas as pd
from collections import defaultdict
import os

def extract_junction_sweep():
    '''Function to extract data describing which sweeps are associated with which junction, from .csv.'''
    # Initialise header 
    header_names = ['Sweep number', 'Row offset', 'sweep length', 'x column', 'SD type', '<VarA>',
                    'V (V)', 'Current(nA)', 'Common contact no.', 'Contact no.', 'Sweep rate', 'date-2000',
                    'time', 'Compliance (nA)', 'x Scaling F', 'lock-in compliance (nA)', 'SEN1 (nA)',
                    'FRQ (Hz)', 'TC1 (s) ', 'Osc Amp (V)', 'Leakage Current (nA)', 'SEN2 (nA)', 'TC2 (s) ',
                    'Probe Height (cm)','Dipping Direction (in+)', 'Ref. theta1', 'Ref. theta2',
                    'Resistance (Ohm)', 'Conductance (S)', 'G/G0', 'temperature (K)', 'diode voltage (V)',
                    'gate contact no.', 'Gate Voltage (V)', 'gate increment direction', 'Relay #']
    # NOTE: Edit filename and filepath as necessary
    filename = f"Data_Processing\\Data_vic\\Chip#{chip_name}\\{data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-status_data.csv"
    try:
        df = pd.read_csv(filename, header=None, names=header_names)
    except FileNotFoundError as e:
        print(f"Oops, error found: {e}\nCheck that you put the right chip name ({chip_name}), and that your files are arranged appropriately!\n")
        print(f"Put the status_data.csv file in the folder: Chip#{chip_name} with name format of {data_prefix}-Chip#{chip_name}-HSC11BIPY-{temperature}K-status_data.csv")
        quit() # Restart if file not found
    # Take out columns of interest
    return df['Contact no.'], df['Sweep number']

def association_junc_sweep(contact_nums, sweep_nums):
    '''Creates dictionary of contact numbers and corresponding sweep numbers'''
    # Create a dictionary to group values
    grouped_dict = defaultdict(list)
    # Add key,value pair to dictionary
    for key, value in zip(contact_nums, sweep_nums):
        grouped_dict[key].append(value)
    return grouped_dict

def write_dict_to_csv(dict):
    '''Writes the junction-sweep correspondence to a dedicated .csv file.'''
    global current_dir
    # Prepare the data to be written
    rows = []
    for key, values in dict.items():
    # Format the row as 'key:values' where values are comma-separated
        row = f"J{key},S{';'.join(map(str, values))}\n"
        rows.append(row)
    # Make the folder for it; tidy things up a bit
    folder_path = os.path.join(current_dir, f"Chip#{chip_name}-{data_prefix}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Write to CSV in write mode
    with open(f'{folder_path}\\{data_prefix}-Chip#{chip_name}-junction_sweep_correspondence.csv', mode='w', newline='') as file:
        file.writelines(rows) #writes it all in one go (hence why row has \n in it)

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get chip name and data prefix
chip_name = input("What's the chip name? (e.g. 35-C5) ")
data_prefix = input("What's the data prefix (e.g. AA or BB etc) ")
temperature = int(input("What's the temperature (e.g. 77 or 300K)? "))
# Extract contact and sweep numbers
contact_nums, sweep_nums = extract_junction_sweep()
dictionary_junc_sweeps = association_junc_sweep(contact_nums, sweep_nums)
# Print the result in the desired format
sorted_dict = {key: dictionary_junc_sweeps[key] for key in sorted(dictionary_junc_sweeps)} # to sort the keys
write_dict_to_csv(sorted_dict)