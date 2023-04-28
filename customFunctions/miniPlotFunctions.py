import os
import pandas as pd
import re
import matplotlib.pyplot as plt


def plot_191125():
    upper_path = "/home/burghoff/Daten/2019_Laeufe/191125_morphMnist/structures/2e-08"

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    add_step = 0
    curr_run_first_layer = []
    curr_run_sec_layer = []
    steps = []
    for run in [1,2,3]:
        for step in range(0, 8501, 100):
            cur_path = os.path.join(upper_path, str(run),  "learned_structure", "alive_" + str(step))
            data = pd.read_csv(cur_path)
            tmp_1 = data.axes[0].values[0]
            tmp_2 = data.axes[0].values[1]
            curr_run_first_layer.append(int(re.findall(r'\d+', tmp_1)[-1]))
            curr_run_sec_layer.append(int(re.findall(r'\d+', tmp_2)[-1]))
            steps.append(step + add_step)
            if run == 1:
                marker = "s"
            elif run == 2:
                marker = "^"
            else:
                marker = "o"
        add_step += 8600
    ax1.scatter(steps, curr_run_first_layer, label="first_layer", color="blue", marker = marker)
    ax1.scatter(steps, curr_run_sec_layer, label="second_layer", color="lightgreen", marker = marker)
    ax1.set_xlabel("steps")
    ax1.set_ylabel("layer_size")
    ax1.legend()

    fig1.savefig(os.path.join(upper_path, "VerlaufArchitektur.png"))
    fig1.show()


