import numpy as np
import os
import pickle

# Specify the directory containing the .pickle files
directory = 'results\data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter to include only .pickle files
pickle_files = [f for f in all_files if f.endswith('.pickle')]

alpha_values = np.arange(-8, 8, 0.05)
print(alpha_values[-1])
start, end = -8, 7.95
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

# Loop through each .pickle file
for pickle_file in pickle_files:
    file_path = os.path.join(directory, pickle_file)
    
    # Open and load the pickle file
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    
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

    # Add your own calculations here
    for i in range(start_ind, end_ind+1):
        a = alpha_values[i]
        for (start, end, color) in arrows:

            if min(start[0], end[0]) <= a < max(start[0], end[0]):
                if (spop_gradients[i] < 0 and color[2] == 1.0) or (spop_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_spo += 1
                if (pfy_gradients[i] < 0 and color[2] == 1.0) or (pfy_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_pfyl += 1  
                if (cave_gradients[i] < 0 and color[2] == 1.0) or (cave_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_cave += 1
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
