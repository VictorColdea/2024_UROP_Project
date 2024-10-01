# 2024_UROP_Project
The code I made for data analysis during my summer project at the Cavendish Laboratory in 2024.
The input to the code was data collected from the experimental apparatus, and the output was parameter plots of the fitted parameters to the quantum tunnelling Simmons model.

The individual programs' objectives are as follows:

- 1-junction_sweep_association.py:
This program is used to determine which junction the sweeps (data-collection cycles) correspond to. It takes in the status data (data about each cycle) from a csv and creates a dictionary, which it then writes to a new junction-sweep csv for the next program to use.

- 2-curve_fit-simmons.py:
This program is used to extract the Simmons model parameters, and send this information to an external csv as well as plot fits for the user to distinguish which fits are useful (i.e. not to exclude). It uses csv files containing sweep data to fit the model for each sweep, and then the junction-sweep csv to sort these parameters according to the junction they correspond to.

- 3-parameter_plotting.py:
This program is used to plot the extracted parameters: both in relation to one another and as individual histograms for each parameter.
