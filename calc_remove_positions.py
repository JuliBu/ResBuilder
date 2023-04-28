import customFunctions.changeArchitecture as arch_ch
import os
import math
import matplotlib.pyplot as plt

path_to_dir = "/home/burghoff/Daten/230330_calcPositions/Nets"
#path_to_dir = "/home/burghoff/Daten/230331_testNetFunc"
experiment_types = ["ENstart", "Res18start"]
experi_colors = ["red", "blue"]
addi_x = [0, 0.5]
datasets = ["Animals10", "CIFAR10", "CIFAR100", "EMNIST", "FashionMNIST", "MNIST"]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

max_index = 50
for idx_experi, experiment_type in enumerate(experiment_types):
    for idx_data, dataset in enumerate(datasets):
        dir_of_files = os.path.join(path_to_dir, experiment_type, dataset)
        nets_to_compare = []
        for i in range(max_index):
            file_add = os.path.join(dir_of_files, str(i)+"_03_added_net.json")
            if os.path.isfile(file_add):
                file_del = os.path.join(dir_of_files, str(i)+"_02_deleted_net.json")
                if os.path.isfile(file_del):
                    nets_to_compare.append([file_add, file_del])
                file_del = os.path.join(dir_of_files, str(i+1)+"_04_shrinked_net.json")
                if os.path.isfile(file_del):
                    nets_to_compare.append([file_add, file_del])
        dataset_pos = []
        for net_pair in nets_to_compare:
            file_add = net_pair[0]
            file_del = net_pair[1]
            added_net = arch_ch.gen_net_from_json_path(file_add,old_format=False)
            diffs_add, diffs_del = added_net.get_differences_to_another_net(file_del, old_net_format=False)
            positions = []
            for diff_layer in diffs_del:
                abs_pos_in_net =  float(added_net.get_position_of_layer_in_net(diff_layer))
                num_layers_in_net = float(len(added_net.layer_list))
                rel_pos_in_net = abs_pos_in_net/num_layers_in_net
                positions.append(rel_pos_in_net)
                #print(experiment_type, dataset, str(i))
                #print(positions)
            for pos in positions:
                dataset_pos.append(pos)
        print(experiment_type, dataset)
        if len(dataset_pos)>0:
            mean_dataset = sum(dataset_pos)/len(dataset_pos)
            #ax1.scatter(idx_data + addi_x[idx_experi], dataset_pos)#, color=experi_colors[idx_experi])
            ax1.boxplot(dataset_pos, positions=[(idx_data + addi_x[idx_experi])])#, color=experi_colors[idx_experi])
        dataset_pos.sort()
        if len(dataset_pos)>0:
            print(mean_dataset)
        else:
            print("No layers deleted")
plt.yticks([0,1], ["Input", "Output"])
#plt.xticks(range(len(datasets)), datasets)
for i in [0.75, 1.75, 2.75, 3.75, 4.75]:
    ax1.axvline(i, color="red", linewidth=0.4)
plt.xticks([0.25, 1.25, 2.25, 3.25, 4.25, 5.25], datasets)
plt.savefig("/home/burghoff/Daten/230330_calcPositions/output.pdf")
