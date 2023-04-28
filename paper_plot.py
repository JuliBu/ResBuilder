import os
import pandas as pd
import matplotlib.pyplot as plt

import config
import customFunctions.changeArchitecture

upper_path = "/home/burghoff/Daten/230221HigherReg_MiddleFLOPsResNet18_v03/"

def plot_layer_and_morphing_steps(upper_path, layer_insertion_steps=config.pipeline_parameters.number_layer_steps, plot_positions=True, old_net_format=False, only_dataset="", manual_benchmarks=False, plot_layer_number_on_blue_line=True,
hide_train_accs=False):
    layer_per_morph_step = layer_insertion_steps
    if only_dataset == "":
        datasets = config.environment_parameters.all_possible_datasets
    else:
        datasets = [only_dataset]

    
    for dataset in datasets:
        output_acc_path = os.path.join(upper_path, "test_Overview_Acc_newFormat" + str(dataset) + ".pdf")
        #main_dir = dir_help.Directory(upper_path)
        if os.path.exists(os.path.join(upper_path, str(dataset))):
            df = pd.read_csv(os.path.join(upper_path, str(dataset), "test_accuracy.csv"))
            df_train = pd.read_csv(os.path.join(upper_path, str(dataset), "train_accuracy.csv"))
            df_no_reg = pd.read_csv(os.path.join(upper_path, str(dataset), "no_reg_test_accuracy.csv"))
            #df_no_reg_train = pd.read_csv(os.path.join(upper_path, str(dataset), str(versuch), "no_reg_train_accuracy.csv"))
            df_no_reg_train = pd.read_csv(os.path.join(upper_path, str(dataset), "no_reg_train_accuracy.csv"))
            test_accs = []
            train_accs = []
            test_accs_no_reg = []
            train_accs_no_reg = []
            for tmp_acc in df.Current_run:
                test_accs.append(tmp_acc)
            for tmp_acc in df_train.Current_run:
                train_accs.append(tmp_acc)
            #for tmp_acc in df_no_reg.Current_run:
            #    test_accs_no_reg.append(tmp_acc)
            #for tmp_acc in df_no_reg_train.Current_run:
            #    train_accs_no_reg.append(tmp_acc)
            fig1, axs = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [1, 2]}, sharex=True)
            ax2 = axs[0]
            ax1 = axs[1]

            #fig1 = plt.figure(figsize=(8,6))
            #ax1 = fig1.add_subplot(2, 1, 2, figsize=(3,6))
            #ax2 = fig1.add_subplot(2, 1, 1, sharex=ax1, figsize=(5,6))


            no_reg_test_y = df_no_reg.Current_run.array
            no_reg_train_y = df_no_reg_train.Current_run.array
            tmp_x_axe = df_no_reg.Current_run.axes
            shifted_x_axe = []
            shifted_x_axe_with_weights_import = []
            y_axe_train = []
            y_axe_train_weight_import = []
            y_axe_test = []
            y_axe_test_weight_import = []
            no_reg_weight_import_files = False
            for idx, element in enumerate(tmp_x_axe[0]):
                if element < 20000:
                    shifted_x_axe.append(element-10000)
                    y_axe_train.append(no_reg_train_y[idx])
                    y_axe_test.append(no_reg_test_y[idx])
                else:
                    no_reg_weight_import_files = True
                    shifted_x_axe_with_weights_import.append(element-20000)
                    y_axe_train_weight_import.append(no_reg_train_y[idx])
                    y_axe_test_weight_import.append(no_reg_test_y[idx])

            ax1.scatter(range(1,len(df.Current_run)+1), test_accs, label="With Reg")
            if not hide_train_accs:
                ax1.scatter(range(1, len(df.Current_run) + 1), train_accs, label="Train acc")
            #ax1.scatter(shifted_x_axe, no_reg_test_y, label="No_Reg_Test_acc")
            #ax1.scatter(shifted_x_axe, no_reg_train_y, label="No_Reg_Train_acc")

            if config.plot_options.plot_no_reg_points:
                ax1.scatter(shifted_x_axe, y_axe_test, label="No Reg RI")
                if not hide_train_accs:
                    ax1.scatter(shifted_x_axe, y_axe_train, label="Variant B")
                if no_reg_weight_import_files:
                    ax1.scatter(shifted_x_axe_with_weights_import, y_axe_test_weight_import, label="No Reg WI")
                    if not hide_train_accs:
                        ax1.scatter(shifted_x_axe_with_weights_import, y_axe_train_weight_import, label="No reg train acc weight import")

            number_of_layers = []
            number_of_channels_in_first_layer = []
            # calculate second y axis
            for pipeline_step in range(1, len(df.Current_run)):
                if pipeline_step == 1:
                    path_to_net = os.path.join(upper_path, str(dataset), 'Netze', 'startnet.json')
                elif pipeline_step % (config.pipeline_parameters.number_layer_steps+1) == 1:
                    path_to_net = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step-1) + '_02_deleted_net.json')
                else:
                    path_to_net = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step) + '_02_deleted_net.json')
                tmp_net = customFunctions.changeArchitecture.gen_net_from_json_path(path_to_net,old_format=old_net_format)
                first_layer = tmp_net.get_first_layer_of_list()
                if isinstance(first_layer, customFunctions.changeArchitecture.Conv2D_Layer):
                    number_of_channels_in_first_layer.append(first_layer.channel_number)
                else:
                    raise ValueError("First Layer in Net is not convolutional layer!")
                number_of_layers.append(tmp_net.count_Layer_before_dense())
            #ax2 = ax1.twinx()

            # Plot insertion and deletetion positions
            #ax3 = ax2.twinx()
            if plot_positions:
                x_values = []
                y_values = []
                for pipeline_step in range(1, len(df.Current_run-1)):

                    #skip first step
                    if not pipeline_step == 1:
                        if pipeline_step == 2:
                            path_to_net1 = os.path.join(upper_path, str(dataset), 'Netze', 'startnet.json')
                            path_to_net2 = path_to_net2 = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step) + '_03_added_net.json')
                        elif ((pipeline_step-1) % (config.pipeline_parameters.number_layer_steps+1)) == 0:
                            path_to_net1 = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step-1) + '_04_shrinked_net.json')
                            path_to_net2 = path_to_net2 = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step) + '_02_deleted_net.json')
                        else:
                            path_to_net1 = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step-1) + '_02_deleted_net.json')
                            path_to_net2 = os.path.join(upper_path, str(dataset), 'Netze', str(pipeline_step) + '_03_added_net.json')
                        # get insertion position
                        net1 = customFunctions.changeArchitecture.gen_net_from_json_path(path_to_net1,old_format=old_net_format)
                        net2 = customFunctions.changeArchitecture.gen_net_from_json_path(path_to_net2,old_format=old_net_format)
                        ins, _ = net1.get_differences_to_another_net(path_to_net2, old_net_format=old_net_format)
                        #ins, _ = customFunctions.get_differences_between_two_nets(path_to_net1, path_to_net2)
                        if not ins == []:
                            for entry in ins:
                                x_values.append(pipeline_step-0.33)
                                y_values.append(net2.get_position_of_layer_in_net(entry))
                #ax2.scatter(x_values, y_values, marker = "+")

                # plot deletion positions
                x_values = []
                y_values = []
                for pipeline_step in range(1, len(df.Current_run - 1)):

                    # skip first step
                    if not pipeline_step == 1:
                        last_step_was_morph = ((pipeline_step - 1) % (config.pipeline_parameters.number_layer_steps + 1)) == 0
                        # get deletion position
                        if last_step_was_morph:
                            path_to_net1 = os.path.join(upper_path, str(dataset), 'Netze',
                                                        str(pipeline_step-1) + '_04_shrinked_net.json')
                        else:
                            path_to_net1 = os.path.join(upper_path, str(dataset), 'Netze',
                                                    str(pipeline_step) + '_03_added_net.json')

                        path_to_net2 = os.path.join(upper_path, str(dataset), 'Netze',
                                                    str(pipeline_step) + '_02_deleted_net.json')
                        net1 = customFunctions.changeArchitecture.gen_net_from_json_path(path_to_net1, old_format=old_net_format)
                        net2 = customFunctions.changeArchitecture.gen_net_from_json_path(path_to_net2, old_format=old_net_format)
                        _, deleted = net1.get_differences_to_another_net(path_to_net2, old_net_format=old_net_format)
                        if not deleted == []:
                            for entry in deleted:
                                x_values.append(pipeline_step+0.33)
                                y_values.append(net1.get_position_of_layer_in_net(entry))
                #ax2.scatter(x_values, y_values, marker="x")

            # Plot vertical blue lines with channel number and current number of layers if multiple steps in LayerLasso
            if config.pipeline_parameters.number_layer_steps > 0:
                for tmp_x in range(1,len(df.Current_run)+1):
                    if tmp_x % (layer_per_morph_step+1) == 0:
                        ax1.axvline(x=tmp_x + 0.5, color="red", linewidth=0.4)
                        ax2.axvline(x=tmp_x + 0.5, color="red", linewidth=0.4)
                        if plot_layer_number_on_blue_line:
                            ax2.text(tmp_x-0.2, 0.45, number_of_channels_in_first_layer[tmp_x-2], rotation=90, transform=ax2.get_xaxis_text1_transform(0)[0])

                ax2.set_ylabel('Number of layers')  # we already handled the x-label with ax1
                ax2.yaxis.get_major_locator().set_params(integer=True)
                ax2.plot(range(1, len(df.Current_run)), number_of_layers, linestyle="dashed", label="Depth of the network", color="purple")

            if manual_benchmarks:
                #x_benchmarks = [4.5, 8.5, 12.5]
                #y_benchmarks = [0.8579, 0.875, 0.8759]
                #ax1.scatter(x_benchmarks, y_benchmarks, label="onlyMorphNet_benchmark")
                ax1.axhline(y=0.9297, linestyle='dotted', label="Max MorphNet Benchmark", color="orange", lw=0.7)

            #ax1.scatter()
            ax2.legend(loc="lower right")
            ax1.legend(loc="lower right")#, bbox_to_anchor=(0.05, 0))
            ax1.xaxis.get_major_locator().set_params(integer=True)
            ax1.set_xlabel('Training Pipeline step')
            ax1.set_ylabel('Test Accuracy')
            #fig1.savefig(os.path.join(upper_path, "Overview_Acc_" + str(dataset) + ".png"))
            fig1.savefig(output_acc_path)

            df_loss = pd.read_csv(os.path.join(upper_path, str(dataset), "test_loss.csv"))
            df_train_loss = pd.read_csv(os.path.join(upper_path, str(dataset), "train_loss.csv"))
            test_loss = []
            train_loss = []
            for tmp_loss in df_loss.Current_run:
                test_loss.append(tmp_loss)
            for tmp_loss in df_train_loss.Current_run:
                train_loss.append(tmp_loss)
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            for tmp_x in range(1,len(df.Current_run)+1):
                if tmp_x % (layer_per_morph_step+1) == 0:
                    ax1.axvline(x=tmp_x + 0.5)
            ax1.scatter(range(1, len(df.Current_run) + 1), test_loss, label="Test_loss")
            ax1.scatter(range(1, len(df.Current_run) + 1), train_loss, label="Train_loss")
            ax1.legend()
            ax1.set_xlabel('Training Pipeline step')
            ax1.set_ylabel('Loss')
            fig1.savefig(os.path.join(upper_path, "Loss_" + str(dataset) + ".png"))

plot_layer_and_morphing_steps(upper_path=upper_path,only_dataset="CE_FraudDet", manual_benchmarks=False, plot_positions=False, hide_train_accs=True)