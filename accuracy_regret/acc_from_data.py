import numpy as np
import os
import re
import pickle
import matplotlib.pyplot as plt

# Specify the directory containing the .pickle files
directory = 'results\data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter to include only .pickle files
#pickle_files = [f for f in all_files if f.endswith('.pickle')]

alpha_values = np.arange(-8, 8, 0.05)
print(alpha_values[-1])
start, end = -8, 7.95
step = 0.05
start_ind = np.argwhere(alpha_values >=start-1e-3)[0][0]
end_ind = np.argwhere(alpha_values >= end-1e-3)[0][0]
print(start_ind)
print(end_ind)
print(alpha_values[start_ind], alpha_values[end_ind])

acumulated_spo = 0
acumulated_pfyl = 0
acumulated_cave = 0
acumulated_regret_spo = 0
acumulated_regret_pfy = 0
acumulated_regret_cave = 0
start_plot = start
end_plot = end
alpha_plot_points_spo = []
alpha_plot_points_pfy = []
alpha_plot_points_cave = []
alpha_plot_points_spo_miss = []
alpha_plot_points_pfy_miss = []
alpha_plot_points_cave_miss = []

# Maximum number of files to process
max_files_to_process = 1000

# List of all files in the directory
all_files = os.listdir(directory)

# Filter and sort the pickle files
pickle_files = [f for f in all_files if f.endswith('.pickle')]

# Function to extract the number from the filename
def extract_number(filename):
    match = re.search(r'knapsack_(\d+)_data\.pickle', filename)
    return int(match.group(1)) if match else float('inf')

# Sort the files based on the extracted number
pickle_files.sort(key=extract_number)

# Limit the number of files to process
pickle_files = pickle_files[:max_files_to_process]

# Process the files in sorted order
for pickle_file in pickle_files:
    file_path = os.path.join(directory, pickle_file)
    
    # Open and load the pickle file
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
        # Process the data as needed
    
    # Perform your calculations using data_dict
    # Example calculations (replace these with your actual calculations)
    seed = data_dict["seed"]
    arrows = data_dict["arrows"]
    spop_regret = data_dict["spop_regret"]
    spop_gradients = data_dict["spop_gradients"]
    pfy_regret = data_dict["pfy_regret"]
    pfy_gradients = data_dict["pfy_gradients"]
    cave_regret = data_dict["cave_regret"]
    cave_gradients = data_dict["cave_gradients"]
    z = data_dict["z"]
    intervals = data_dict["intervals"]
    horzizontal_plots = data_dict["horizontal_plots"]
    
    # Example calculation: print the seed value
    print(f"Iteration: {seed}")
    accuracy_spo = 0
    accuracy_pfyl = 0
    accuracy_cave = 0
    total_points = 0

    regret_spo = 1e9
    pfy_min_regret = 1e9
    cave_min_regret = 1e9

    alpha_plot_points_spo.append([[round(i, 2), 0] for i in np.arange(start_plot, end_plot + step, step)])
    alpha_plot_points_pfy.append([[round(i, 2), 0] for i in np.arange(start_plot, end_plot + step, step)])
    alpha_plot_points_cave.append([[round(i, 2), 0] for i in np.arange(start_plot, end_plot + step, step)])
    alpha_plot_points_spo_miss.append([[round(i, 2), 0] for i in np.arange(start_plot, end_plot + step, step)])
    alpha_plot_points_pfy_miss.append([[round(i, 2), 0] for i in np.arange(start_plot, end_plot + step, step)])
    alpha_plot_points_cave_miss.append([[round(i, 2), 0] for i in np.arange(start_plot, end_plot + step, step)])

    # Add your own calculations here
    for i in range(start_ind, end_ind+1):
        a = alpha_values[i]
        #print(a)
        for (start, end, color) in arrows:
            if min(start[0], end[0]) <= a < max(start[0], end[0]):
                #print(np.ceil((a-start_plot)/step))
                if (spop_gradients[i] < 0 and color[2] == 1.0) or (spop_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_spo += 1
                    alpha_plot_points_spo[int(seed)][int(np.ceil((a-start_plot)/step))][1] += 1
                else:
                    alpha_plot_points_spo_miss[int(seed)][int(np.ceil((a-start_plot)/step))][1] += 1
                if (pfy_gradients[i] < 0 and color[2] == 1.0) or (pfy_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_pfyl += 1
                    alpha_plot_points_pfy[int(seed)][int(np.ceil((a-start_plot)/step))][1] += 1
                else:
                    alpha_plot_points_pfy_miss[int(seed)][int(np.ceil((a-start_plot)/step))][1] += 1
                if (cave_gradients[i] < 0 and color[2] == 1.0) or (cave_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_cave += 1
                    alpha_plot_points_cave[int(seed)][int(np.ceil((a-start_plot)/step))][1] += 1
                else:
                    alpha_plot_points_cave_miss[int(seed)][int(np.ceil((a-start_plot)/step))][1] += 1
                total_points += 1
                break
            else:
                continue
        
        
        
        


        # Ensure that we don't exceed bounds when looking ahead by one
        if i + 1 < len(alpha_values):
            # Check whether we cross zero around the current alpha
            # The gradient can cross zero by either decreasing or increasing
            if (spop_gradients[i] >= 0 and spop_gradients[i+1] <= 0) \
            or (spop_gradients[i] <= 0 and spop_gradients[i+1] >= 0):

                # Find the interval that corresponds to the current alpha
                # I feel like there is a more efficient way to do this, implement if know
                for interval_idx, interval in enumerate(intervals):
                    interval_lower = interval[0]
                    interval_upper = interval[1]
                    
                    if a >= interval_lower and a <= interval_upper:
                        value_at_zero_grad = horzizontal_plots[interval_idx][0]
                        regret_spo = (z - value_at_zero_grad)/z
                        #print(f"Achieved value of {value_at_zero_grad} with regret {regret}")
                        break

            if (pfy_gradients[i] >= 0 and pfy_gradients[i+1] <= 0) \
            or (pfy_gradients[i] <= 0 and pfy_gradients[i+1] >= 0):

                # Find the interval that corresponds to the current alpha
                # I feel like there is a more efficient way to do this, implement if know
                for interval_idx, interval in enumerate(intervals):
                    interval_lower = interval[0]
                    interval_upper = interval[1]
                    
                    if a >= interval_lower and a <= interval_upper:
                        value_at_zero_grad = horzizontal_plots[interval_idx][0]
                        regret_pfy = z - value_at_zero_grad
                        if regret_pfy < pfy_min_regret:
                            pfy_min_regret = regret_pfy
                        #print(f"Achieved value of {value_at_zero_grad} with regret {regret}")
                        break

            if (cave_gradients[i] >= 0 and cave_gradients[i+1] <= 0) \
            or (cave_gradients[i] <= 0 and cave_gradients[i+1] >= 0):

                # Find the interval that corresponds to the current alpha
                # I feel like there is a more efficient way to do this, implement if know
                for interval_idx, interval in enumerate(intervals):
                    interval_lower = interval[0]
                    interval_upper = interval[1]
                    
                    if a >= interval_lower and a <= interval_upper:
                        value_at_zero_grad = horzizontal_plots[interval_idx][0]
                        regret_cave = z - value_at_zero_grad
                        if regret_cave < cave_min_regret:
                            cave_min_regret = regret_cave
                        #print(f"Achieved value of {value_at_zero_grad} with regret {regret}")
                        break

    accuracy_spo = accuracy_spo / total_points
    accuracy_pfyl = accuracy_pfyl / total_points
    accuracy_cave = accuracy_cave / total_points

    #if the gradient didn't cross, assume it keeps pushing towards most left or right segment
    if(regret_spo == 1e9):
        if(np.mean(spop_gradients) > 0):
            regret_spo = (z-horzizontal_plots[0][0])/z
        else:
            regret_spo = (z-horzizontal_plots[-1][0])/z

    if(pfy_min_regret == 1e9):
        if(np.mean(pfy_gradients) > 0):
            pfy_min_regret = (z-horzizontal_plots[0][0])
        else:
            pfy_min_regret = (z-horzizontal_plots[-1][0])

    if(cave_min_regret == 1e9):
        if(np.mean(cave_gradients) > 0):
            cave_min_regret = (z-horzizontal_plots[0][0])
        else:
            cave_min_regret = (z-horzizontal_plots[-1][0])

    regret_pfy = pfy_min_regret/z
    regret_cave = cave_min_regret/z

    acumulated_spo += accuracy_spo
    acumulated_pfyl += accuracy_pfyl
    acumulated_cave += accuracy_cave
    acumulated_regret_spo += regret_spo
    acumulated_regret_pfy += regret_pfy
    acumulated_regret_cave += regret_cave

runs = 1000
print("The Accuracy for SPO+ is "+str((acumulated_spo/runs)*100)+"%")
print("The Accuracy for PFYL is "+str((acumulated_pfyl/runs)*100)+"%")
print("The Accuracy for CaVE is "+str((acumulated_cave/runs)*100)+"%")
print("The regret for SPO+ is "+str((acumulated_regret_spo/runs)*100)+"%")
print("The regret for PFYL is "+str((acumulated_regret_pfy/runs)*100)+"%")
print("The regret for CaVE is "+str((acumulated_regret_cave/runs)*100)+"%")
print("Processing complete.")


def extract_xy(array):
    x_values = [sub_array[:, 0].tolist() for sub_array in array]
    y_values = [sub_array[:, 1].tolist() for sub_array in array]
    return x_values, y_values

# Convert lists back to numpy arrays to use slicing
array1 = np.array(alpha_plot_points_spo)
array2 = np.array(alpha_plot_points_spo_miss)

# Extract x and y values from array1 and array2
x1_values = array1[:, :, 0].tolist()
y1_values = array1[:, :, 1].tolist()
x2_values = array2[:, :, 0].tolist()
y2_values = array2[:, :, 1].tolist()

# Plotting the bar plot
plt.figure(figsize=(14, 7))

# Plotting array
x1 = [item for sublist in x1_values for item in sublist]
y1 = [sum(column) for column in zip(*y1_values)]
x2 = [item for sublist in x2_values for item in sublist]
y2 = [sum(column) for column in zip(*y2_values)]

plt.bar(x1_values[0], y1, width=0.05, color='blue', alpha=0.7)

plt.bar(x2_values[0], y2, width=0.05, color='red', alpha=0.7)

# Adding labels and title
plt.xlabel('Alpha')
plt.ylabel('Data points')
plt.title('Accuracy of SPO+')
plt.legend([f'Correct Gradient Sense {round((acumulated_spo/runs)*100,2)}%', f'Wrong Gradient Sense {round((1-(acumulated_spo/runs))*100,2)}%'])

# Saving the plot
plt.savefig(f'spo_acc_plot_{-start_plot}.pdf')

# Convert lists back to numpy arrays to use slicing
array1 = np.array(alpha_plot_points_pfy)
array2 = np.array(alpha_plot_points_pfy_miss)

# Extract x and y values from array1 and array2
x1_values = array1[:, :, 0].tolist()
y1_values = array1[:, :, 1].tolist()
x2_values = array2[:, :, 0].tolist()
y2_values = array2[:, :, 1].tolist()

# Plotting the bar plot
plt.figure(figsize=(14, 7))

# Plotting array
x1 = [item for sublist in x1_values for item in sublist]
y1 = [sum(column) for column in zip(*y1_values)]
x2 = [item for sublist in x2_values for item in sublist]
y2 = [sum(column) for column in zip(*y2_values)]

plt.bar(x1_values[0], y1, width=0.05, color='blue', alpha=0.7)

plt.bar(x2_values[0], y2, width=0.05, color='red', alpha=0.7)

# Adding labels and title
plt.xlabel('Alpha')
plt.ylabel('Data points')
plt.title('Accuracy of PFYL')
plt.legend([f'Correct Gradient Sense {round((acumulated_spo/runs)*100,2)}%', f'Wrong Gradient Sense {round((1-(acumulated_spo/runs))*100,2)}%'])

# Saving the plot
plt.savefig(f'pfyl_acc_plot_{-start_plot}.pdf')

# Convert lists back to numpy arrays to use slicing
array1 = np.array(alpha_plot_points_cave)
array2 = np.array(alpha_plot_points_cave_miss)

# Extract x and y values from array1 and array2
x1_values = array1[:, :, 0].tolist()
y1_values = array1[:, :, 1].tolist()
x2_values = array2[:, :, 0].tolist()
y2_values = array2[:, :, 1].tolist()

# Plotting the bar plot
plt.figure(figsize=(14, 7))

# Plotting array
x1 = [item for sublist in x1_values for item in sublist]
y1 = [sum(column) for column in zip(*y1_values)]
x2 = [item for sublist in x2_values for item in sublist]
y2 = [sum(column) for column in zip(*y2_values)]

plt.bar(x1_values[0], y1, width=0.05, color='blue', alpha=0.8)

plt.bar(x2_values[0], y2, width=0.05, color='red', alpha=0.6)

# Adding labels and title
plt.xlabel('Alpha')
plt.ylabel('Data points')
plt.title('Accuracy of CaVE')
plt.legend([f'Correct Gradient Sense {round((acumulated_spo/runs)*100,2)}%', f'Wrong Gradient Sense {round((1-(acumulated_spo/runs))*100,2)}%'])

# Saving the plot
plt.savefig(f'cave_acc_plot_{-start_plot}.pdf')