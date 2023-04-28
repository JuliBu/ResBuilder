import csv
import json
import math
import os
import random
import time
from copy import deepcopy

from typing import List, Tuple
from importlib_resources import path
import pandas as pd
from shutil import copyfile

import config
from config import pipeline_parameters, regularization_parameters, environment_parameters

if config.plot_options.plot_arch_HarisIqbal:
    from git_repo_plotNN.plotNN_pycore.tikzeng import *


class Layer:
    def __init__(self, name: str, insertion_step: int, layer_type: str):
        self.name = name
        self.insertion_step = insertion_step
        self.layer_type = layer_type
        self.will_be_kicked = False
        self.kicked_reason = "n.a."
        self.insertion_probability = 0


class Conv2D_Layer(Layer):
    def __init__(self, name: str, insertion_step: int, channel_number: int, filter_size=3, filter_size_x=3, filter_size_y=3, insertion_index=1, stride=1):
        super().__init__(name=name, insertion_step=insertion_step, layer_type="Conv2D")
        self.channel_number = channel_number
        self.filter_size = filter_size
        self.filter_size_x = filter_size
        self.filter_size_y = filter_size
        self.insertion_index = insertion_index
        self.stride=stride

    def log_added_conv_layer(self, logfile_path):
        with open(logfile_path, 'a') as logflie:
            logflie.write("Conv Layer " + self.name + " with " + str(self.channel_number) + " channels and filter size " + str(self.filter_size) + " was added to the net.")


class Dense_Layer(Layer):
    def __init__(self, name: str, insertion_step: int, neuron_number: int):
        super().__init__(name=name, insertion_step=insertion_step, layer_type="Dense")
        self.neuron_number = neuron_number


class Pool_Layer(Layer):
    def __init__(self, name: str, insertion_step: int, pooling_type: str, pool_size=2, pool_stride=2):
        """
        :param pooling_type: Pooling layer_type can be something like "average" or "max". ToDo: At the moment is only "max" implemented
        """
        super().__init__(name=name, insertion_step=insertion_step, layer_type="Pool")
        self.pooling_type = pooling_type
        self.pool_size = pool_size
        self.pool_stride = pool_stride


class Skip_Connection:
    """
    Class where each object contains the information about a single skip connection
    """
    def __init__(self, name: str, start_layer: Layer, target_layer: Layer, insertion_step: int):
        self.name = name
        self.start_layer = start_layer
        self.target_layer = target_layer
        self.insertion_step = insertion_step
        self.skip_value = None
        self.will_be_kicked = False

    def log_added_skip_connect(self, path_to_log):
        """
        Writes information of the skip layer to the logfile
        """
        with open(path_to_log, 'a') as log_file:
            log_file.write("Skip " + self.name +" from " + str(self.start_layer.name) + " to " + str(self.target_layer) + " was added.")

    def print_skip(self):
        """
        Prints explaining information about the skip connection
        """
        print("Die Skip-Connection", self.name, "verläuft von", self.start_layer.name, "nach", self.target_layer.name, "und wurde in Schritt", str(self.insertion_step), "eingefügt.")


class Layer_Block:
    def __init__(self, block_layer_list: List[Layer]):
        self.block_layer_list = block_layer_list

    def number_of_layers_in_block(self) -> int:
        return len(self.block_layer_list)

    def get_list_of_all_layer_names(self):
        full_list = []
        for layer in self.block_layer_list:
            full_list.append(layer.name)
        return full_list


class Net:
    def __init__(self, layer_list: List[Layer], skip_connect_list: List[Skip_Connection], source_path = ""):
        self.layer_list = layer_list
        self.skip_connect_list = skip_connect_list
        self.source_path = source_path

    def write_to_json_file(self, path_to_file):
        new_json_net = self.toJSON()
        with open(path_to_file, 'w') as json_file:
            json_file.write(new_json_net)

    def write_old_format_json(self, path_to_file):
        list_of_all = []
        dont_write_last_comma = True
        for layer in self.layer_list:
            if not layer.will_be_kicked:
                if isinstance(layer, Conv2D_Layer):
                    str_line = "\"" + layer.name + "\": " + str(layer.channel_number) + ",\n"
                    list_of_all.append(str_line)
                elif isinstance(layer, Dense_Layer):
                    str_line = "\"" + layer.name + "\": " + str(layer.neuron_number) + ",\n"
                    list_of_all.append(str_line)
                elif isinstance(layer, Pool_Layer):
                    str_line = "\"" + layer.name + "\": \"" + str(layer.pooling_type) + "\",\n"
                    list_of_all.append(str_line)
        for skip in self.skip_connect_list:
            if not skip.will_be_kicked:
                output_string = skip.start_layer.name + "&" + skip.target_layer.name
                str_line = "\"" + skip.name + "\": \"" + str(output_string) + "\",\n"
                list_of_all.append(str_line)
        with open(path_to_file, "w") as write_file:
            write_file.write("{\n")
            for entry in list_of_all:
                if dont_write_last_comma and entry == list_of_all[-1]:
                    write_file.write("  " + entry.split(",")[0] + "\n")
                else:
                    write_file.write("  " + entry)
            write_file.write("}")

    # Functions for Changing architecture
    def count_Layer_before_dense(self):
        counter = 0
        for layer in self.layer_list:
            if layer.layer_type == "Dense":
                break
            counter += 1
        return counter

    def get_random_layer_before_dense(self):
        layers_before_dense = []
        for layer in self.layer_list:
            if isinstance(layer, Dense_Layer):
                break
            else:
                layers_before_dense.append(layer)
        return random.choice(layers_before_dense)

    def get_first_layer_of_list(self) -> Layer:
        return self.layer_list[0]

    def get_random_conflictless_insert_layer(self) -> Layer:
        possible_layers = []
        possible_layers.append(self.get_first_layer_of_list())
        for layer in self.layer_list:
            if isinstance(layer, Pool_Layer):
                possible_layers.append(layer)
        for skip in self.skip_connect_list:
            possible_layers.append(skip.target_layer)
        # Remove Duplicates and select a random layer
        return random.choice(list(set(possible_layers)))

    def delete_list_of_layers(self, deleted_insertion_steps):
        # Check for layers which are kicked because an other layer in the same block was kicked
        for layer in self.layer_list:
            if layer.insertion_step in deleted_insertion_steps and not layer.will_be_kicked:
                layer.will_be_kicked = True
                layer.kicked_reason = "another layer of this block was kicked"

        # Check for layers which are kicked because they are incepted in a kicked layer(block)
        # And also kick those skip_connects
        for skip in self.skip_connect_list:
            layer_found = False
            for layer in self.layer_list:
                if skip.start_layer.name in layer.name and layer.will_be_kicked:
                    skip.will_be_kicked = True
                    layer_found = True
                if layer_found and not layer.will_be_kicked:
                    layer.will_be_kicked = True
                    layer.kicked_reason = "lays inside of another block which is deleted"
                if skip.target_layer.name in layer.name:
                    break

    def get_random_conflictless_insert_layer_blockwise_probability(self) -> Layer:
        blocks = []
        tmp_block = Layer_Block([])
        blocks.append(tmp_block)
        for layer in self.layer_list:
            if isinstance(layer, Pool_Layer):
                tmp_block = Layer_Block([])
                blocks.append(tmp_block)
                tmp_block.block_layer_list.append(layer)
            elif isinstance(layer, Conv2D_Layer):
                tmp_block.block_layer_list.append(layer)

        possible_layers = []
        possible_layers.append(self.get_first_layer_of_list())
        for layer in self.layer_list:
            if isinstance(layer, Pool_Layer):
                possible_layers.append(layer)
        for skip in self.skip_connect_list:
            possible_layers.append(skip.target_layer)
        # Remove Duplicates and select a random layer
        possible_layers = list(set(possible_layers))

        for block in blocks:
            for b_layer in block.block_layer_list:
                if not b_layer in possible_layers:
                    block.block_layer_list.remove(b_layer)

        random_block = random.choice(blocks)
        random_layer = random.choice(random_block.block_layer_list)

        return random_layer

    def get_last_conv_layer_before_inserted_layer(self, layer_after_which_is_inserted: Layer) -> Conv2D_Layer:
        last_conv_layer = self.get_first_layer_of_list()
        for layer in self.layer_list:
            if isinstance(layer, Conv2D_Layer):
                last_conv_layer = layer
            if layer == layer_after_which_is_inserted:
                return last_conv_layer
        raise ValueError("Layer " + layer_after_which_is_inserted.name + " nicht gefunden!")

    def get_layer_after_which_is_inserted(self) -> Layer:
        """
        This function looks in config.py where to add a new architecture of layers to the net.
        """
        if pipeline_parameters.position_of_adding == "random":
            return self.get_random_layer_before_dense()
        elif pipeline_parameters.position_of_adding == "after_first":
            return self.get_first_layer_of_list()
        elif pipeline_parameters.position_of_adding == "after_ending_skip":
            return self.get_random_conflictless_insert_layer()
        elif pipeline_parameters.position_of_adding == "equal_blockwise":
            return self.get_random_conflictless_insert_layer_blockwise_probability()

    def add_structure(self, Default_directory: str, curr_run: int, logfile_path:str):
        # This is the case for blockwise insertion. Other methods can be added below.
        if pipeline_parameters.type_of_adding_layers == "one_block":
            # Look for the layer after which the new block should be inserted
            layer_after_which_is_inserted = self.get_layer_after_which_is_inserted()
            index_to_insert = self.layer_list.index(layer_after_which_is_inserted) + 1

            # The new layer will get the same channel_size as the layer before to reduce dimension conflicts.
            # It could be solved in other way if the Zero_Padding of channels is active
            channel_number = self.get_last_conv_layer_before_inserted_layer(
                layer_after_which_is_inserted).channel_number

            # To generate different names of the Conv_Layers which are inserted at the same time, we set up a counter.
            number_of_layer_in_block = 0
            layer_name = gen_conv_layer_name(curr_run, number_of_layer_in_block)
            # Generate and a new layer to net
            new_layer = Conv2D_Layer(layer_name, curr_run, channel_number)
            self.layer_list.insert(index_to_insert, new_layer)
            # log the insertion
            new_layer.log_added_conv_layer(logfile_path)

            # Check how many layers per block should be added
            if pipeline_parameters.number_of_layers_added == 1:
                # Generate new skip connection
                skip_name = gen_skip_connect_name(curr_run)
                new_skip_connect = Skip_Connection(skip_name, new_layer, new_layer, curr_run)
                self.skip_connect_list.append(new_skip_connect)
                new_skip_connect.log_added_skip_connect(logfile_path)

            elif pipeline_parameters.number_of_layers_added > 1:
                # If there are more than one layer per block, we add them in a loop
                first_layer_in_block = new_layer
                last_layer_in_block = new_layer
                for step_of_layer_in_block in range(pipeline_parameters.number_of_layers_added - 1):
                    index_to_insert += 1
                    number_of_layer_in_block += 1
                    layer_name = gen_conv_layer_name(curr_run, number_of_layer_in_block)
                    new_layer = Conv2D_Layer(layer_name, curr_run, channel_number)
                    self.layer_list.insert(index_to_insert, new_layer)
                    new_layer.log_added_conv_layer(logfile_path)
                    last_layer_in_block = new_layer
                skip_name = gen_skip_connect_name(curr_run)
                new_skip_connect = Skip_Connection(skip_name, first_layer_in_block, last_layer_in_block, curr_run)
                self.skip_connect_list.append(new_skip_connect)
                new_skip_connect.log_added_skip_connect(logfile_path)
            else:
                raise ValueError("number_of_layers_added must be a positive integer!")
        else:
            raise ValueError(
                "type_of_adding_layers = " + str(pipeline_parameters.type_of_adding_layers) + " is no valid value!")

    def morphnet_shrinkage(self, path_to_alive_file: str):
        alive_net = alive_json_to_net(path_to_alive_file)
        morph_intensity_shrinkage = config.regularization_parameters.morph_intensity_shrinkage
        # Shrinkage routine
        # Look for renamed layer in alive file because in alive files their names end with "/convolution" instead of "/Conv2D"
        for layer in self.layer_list:
            if isinstance(layer, Conv2D_Layer):
                for alive_layer in alive_net.layer_list:
                    if isinstance(alive_layer, Conv2D_Layer):
                        if alive_layer.name.split("/")[0] in layer.name:
                            layer.channel_number = math.floor(morph_intensity_shrinkage * alive_layer.channel_number + (
                                        1 - morph_intensity_shrinkage) * layer.channel_number)
                    else:
                        raise ValueError("Layer in alive file is not type Conv2D!")

    def morphnet_expanding(self, path_to_flopcosts: str):
        target_flops = regularization_parameters.target_flops
        morph_intensity_expanding = config.regularization_parameters.morph_intensity_expanding

        def calc_omega_for_expanding(flopcost_path: str, flop_target: float) -> float:
            with open(flopcost_path, 'r') as flop_file:
                reader = csv.reader(flop_file, delimiter=",")
                for row in reader:
                    if row:
                        actual_flops = row[1]
            return math.sqrt(float(flop_target) / float(actual_flops))

        omega = calc_omega_for_expanding(path_to_flopcosts, float(target_flops))
        for layer in self.layer_list:
            if isinstance(layer, Conv2D_Layer):
                old_wide = layer.channel_number
                suggested_wide = math.floor(old_wide * omega)
                layer.channel_number = math.floor(
                    morph_intensity_expanding * suggested_wide + (1 - morph_intensity_expanding) * old_wide)
                if config.pipeline_parameters.delete_layers_with_0_channels and layer.channel_number == 0:
                    layer.will_be_kicked = True
                    layer.kicked_reason = "Channel number is 0"

    def morphnet_routine(self,
                                   path_to_alive_file: str,
                                   path_to_morphed_net: str,
                                   path_to_flopcosts: str,
                                   expanding=True) -> str:
        """
        Routine that handles MorphNets Shrinkage and Expanding function
        """
        # Shrinkage routine
        self.morphnet_shrinkage(path_to_alive_file)
        # Expanding routine
        if expanding:
            self.morphnet_expanding(path_to_flopcosts)

        if config.pipeline_parameters.delete_layers_with_0_channels:
            # Delete layers with channel size set to zero.
            self.mark_layers_with_0_channels_for_del()

        self.mark_skips_and_layers_in_same_block()
        self.del_marked_layers_and_skips()

        self.write_to_json_file(path_to_morphed_net)
        return path_to_morphed_net

    def get_differences_to_another_net(self, path_to_other_net: str, old_net_format=False) -> Tuple[List[Layer], List[Layer]]:
        #other_net = json_to_net(path_to_other_net)
        other_net = gen_net_from_json_path(path_to_other_net, old_format=old_net_format)
        inserted_layers = []
        deleted_layers = []
        # Search for deleted layers
        for this_layer in self.layer_list:
            layer_found = False
            for other_layer in other_net.layer_list:
                if this_layer.name in other_layer.name:
                    layer_found = True
                    break
            if not layer_found:
                deleted_layers.append(this_layer)
        # Search for added layers
        for other_layer in other_net.layer_list:
            layer_found = False
            for this_layer in self.layer_list:
                if other_layer.name in this_layer.name:
                    layer_found = True
                    break
            if not layer_found:
                inserted_layers.append(other_layer)
        return inserted_layers, deleted_layers

    def get_position_of_layer_in_net(self, searched_layer: Layer) -> int:
        for index, layer in enumerate(self.layer_list):
            if searched_layer.name in layer.name:
                return index+1
        raise ValueError("Layer " + searched_layer.name + " not found!")

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def mark_layers_with_0_channels_for_del(self):#
        for layer in self.layer_list:
            if isinstance(layer, Conv2D_Layer) and layer.channel_number == 0:
                layer.will_be_kicked = True
                layer.kicked_reason = "Channel size set to 0"

    def mark_skips_and_layers_in_same_block(self):
        """
        Sets will_be_deleted True for skip connects if a layer inside this skip connection is set to zero channels.
        Also sets other layers inside these skip_connection to will_be_deleted=True.
        """
        # Search for skip connections which contain at least one layer which should be kicked
        for skip_conection in self.skip_connect_list:
            skip_started = False
            for layer in self.layer_list:
                if skip_conection.start_layer == layer:
                    skip_started = True
                if skip_started and layer.will_be_kicked:
                    skip_conection.will_be_kicked = True
                    break
                if skip_conection.target_layer == layer:
                    break
        # Set all layers inside the skip_connection which should be deleted also to will_be_deleted
        for skip_conection in self.skip_connect_list:
            if skip_conection.will_be_kicked:
                skip_started = False
                for layer in self.layer_list:
                    if skip_conection.start_layer == layer:
                        skip_started = True
                    if skip_started:
                        if not layer.will_be_kicked:
                            layer.will_be_kicked = True
                            layer.kicked_reason = "Another layer of this block was set to be kicked"
                    if skip_conection.target_layer == layer:
                        break

    def mark_layers_under_threshold(self, path_to_summed_weights: str, path_to_del_net: str,
                                      deleting_threshold: float):
        old_layer_list = self.layer_list
        weights = read_summed_weights(path_to_summed_weights)
        # Check for layers which are under the threshold
        for layer in old_layer_list:
            for weight in weights:
                if weight.layer_name in layer.name:
                    if weight.weight < deleting_threshold:
                        layer.will_be_kicked = True
                        layer.kicked_reason = "under threshold"
                    break
        self.write_to_json_file(path_to_del_net)

    def del_marked_layers_and_skips(self):
        #time.sleep(pipeline_parameters.waiting_time_before_delete)
        # for skip_connect in self.skip_connect_list:
        #     if isinstance(skip_connect, Skip_Connection) and skip_connect.will_be_kicked:
        #         self.skip_connect_list.remove(skip_connect)
        self.skip_connect_list = [x for x in self.skip_connect_list if not x.will_be_kicked]
        # for layer in self.layer_list:
        #     if isinstance(layer, Conv2D_Layer) and layer.will_be_kicked:
        #         self.layer_list.remove(layer)
        self.layer_list = [x for x in self.layer_list if not x.will_be_kicked]
        
    
    def plot_net_as_tex(self, output_path, plot_resNet=True, highlight_newest_layer=True):
        if highlight_newest_layer and self.source_path == "":
            raise ValueError("Tried to use plot_net_as_tex function but source path for net was not specified which is necessary for this function!")
        
        def gen_neighbour_name(layer_name: str) -> str:
            return "(" + layer_name + "-east)"
        
        def get_rid_of_neighbour_name(neighbour_name) -> str:
            tmp_layer_name = neighbour_name[1:]
            return tmp_layer_name[:-6]
        
        def scale_height(orig_width):
            return math.log2(orig_width)*3
        
        def scale_width(orig_width):
            return orig_width / 32 / config.plot_options.width_shrinkage_scale

        def scale_dense_depth(nr_neurons):
            return math.log2(nr_neurons)*3
        
        def append_layer_bock_to_arch(arch, layer_block: Layer_Block, left_neighbour: str, newest_layer: Layer) -> str:
            if layer_block.number_of_layers_in_block() > 2:
                #raise ValueError("PlotNN_function not implemented for layerblocks with more than 2 layers!")
                print("123test456")
            else:
                new_layer_inserted = False
                block_layer_names_list = layer_block.get_list_of_all_layer_names()
                if newest_layer.name in block_layer_names_list:
                    new_layer_inserted = True
                first_layer_in_block = layer_block.block_layer_list[0]
                if not isinstance(first_layer_in_block, Conv2D_Layer):
                    raise ValueError("Cannot print layertype " + first_layer_in_block.layer_type + " within block!")
                else:
                    if layer_block.number_of_layers_in_block() == 1:
                        arch.append(to_Conv(first_layer_in_block.name, first_layer_in_block.channel_number, current_image_size[0], offset="(0.5,0,0)", to=left_neighbour, height=scale_height(current_image_size[1]), depth=scale_height(current_image_size[0]), width=scale_width(first_layer_in_block.channel_number), new_inserted=new_layer_inserted))
                    elif layer_block.number_of_layers_in_block() == 2:
                        second_layer_in_block = layer_block.block_layer_list[1]
                        if not isinstance(second_layer_in_block, Conv2D_Layer):
                            raise ValueError("Cannot print layertype " + first_layer_in_block.layer_type + " within block!")
                        n_filter1 = first_layer_in_block.channel_number
                        n_filter2 = second_layer_in_block.channel_number
                        sum_symbol_name = "sum_" + first_layer_in_block.name

                        arch.append(to_ConvConvRelu(name=first_layer_in_block.name, s_filer=current_image_size[0], n_filer=(n_filter1,n_filter2), offset="(0.5,0,0)", to=left_neighbour, width=(scale_width(n_filter1),scale_width(n_filter2)), height=scale_height(current_image_size[1]), depth=scale_height(current_image_size[0]), new_inserted=new_layer_inserted)) # , caption="\"" + first_layer_in_block.name.replace("_", "") + "\""
                        arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), first_layer_in_block.name))
                        #arch.append(to_path(left_neighbour, first_layer_in_block.name, "out_"+ get_rid_of_neighbour_name(left_neighbour)))
                        arch.append(to_Sum(sum_symbol_name, offset="(0.8,0,0)", to="(" + first_layer_in_block.name + "-east)", radius=1.5, opacity=0.6))
                        entries_in_arch_list.append(sum_symbol_name)
                        arch.append(to_connection(first_layer_in_block.name, sum_symbol_name))
                        #arch.append(to_path(gen_neighbour_name(first_layer_in_block.name), sum_symbol_name, "in_"+ sum_symbol_name))
                        #entries_in_arch_list.append(first_layer_in_block.name)
                        return gen_neighbour_name(sum_symbol_name)

        if highlight_newest_layer:
            newest_layer = get_newly_inserted_layer(self.source_path)
        else:
            newest_layer = None    
        entries_in_arch_list = []
        left_neighbour = "(0,0,0)"
        arch = []
        #arch.append(to_head( os.path.join(environment_parameters.project_folder, "git_plot_NN") ))
        arch.append(to_head(config.plot_options.path_to_layer_folder ))
        arch.append(to_cor())
        #arch.append(to_begin(filename=output_path.split("/")[-1]))
        arch.append(to_begin(filename=""))
        any_active_block = False
        active_layer_block = Layer_Block([])
        nr_of_previous_poolings = 0
        orig_image_size = config.plot_options.orig_image_size
        # Here is show image - outcomment for ResNet plot
        #arch.append(to_input("figures/MelanomBeispielBild.png", width=7, height=6.5))

        for layer in self.layer_list:
            divisor = 2 ** nr_of_previous_poolings
            current_image_size = (math.ceil(orig_image_size[0] / divisor), math.ceil(orig_image_size[1] / divisor))
            if isinstance(layer, Conv2D_Layer):
                for skip_con in self.skip_connect_list:
                    if layer == skip_con.start_layer:
                        any_active_block = True
                        break

                if any_active_block:
                    active_layer_block.block_layer_list.append(layer)
                else:
                    arch.append(to_Conv(layer.name, current_image_size[0], layer.channel_number, offset="(0.5,0,0)", to=left_neighbour, height=scale_height(current_image_size[1]), depth=scale_height(current_image_size[0]), width=scale_width(layer.channel_number))) #  , caption="\"" + layer.name.replace("_", "")+ "\""
                    # Aenderung! offset war 0,0,0
                    if plot_resNet:
                        if not left_neighbour=="(0,0,0)":
                            arch.append(to_connection(left_neighbour, layer.name))
                        else:
                            entries_in_arch_list.append(layer.name)
                    else:
                        arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), layer.name))
                    left_neighbour = gen_neighbour_name(layer.name)
                    

                for skip_con in self.skip_connect_list:
                    if layer == skip_con.target_layer:
                        left_neighbour = append_layer_bock_to_arch(arch, active_layer_block, left_neighbour, newest_layer)
                        active_layer_block = Layer_Block([])
                        any_active_block = False
                        break

            if isinstance(layer,Pool_Layer):
                nr_of_previous_poolings += 1
                divisor = 2 ** nr_of_previous_poolings
                current_image_size = (math.ceil(orig_image_size[0] / divisor), math.ceil(orig_image_size[1] / divisor))
                #Works under the constraint that Pooling is not the first layer
                #arch.append(to_Pool(name=layer.name, offset="(1,0,0)", to="(" + previous_layer.name + "-east)", width=1, height=scale_height(current_image_size[1]), depth=scale_height(current_image_size[0]), opacity=0.5))
                arch.append(to_Pool(name=layer.name, offset="(0.5,0,0)", to="(" + get_rid_of_neighbour_name(left_neighbour) + "-east)", width=1, height=scale_height(current_image_size[1]), depth=scale_height(current_image_size[0]), opacity=0.5))
                #if not plot_resNet:
                arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), layer.name))
                
                left_neighbour = gen_neighbour_name(layer.name)

            if isinstance(layer, Dense_Layer):
                cur_depth = scale_dense_depth(layer.neuron_number)
                arch.append(to_SoftMax(layer.name, layer.neuron_number, "(1,0,0)", left_neighbour, depth=cur_depth))
                arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), layer.name))
                left_neighbour = gen_neighbour_name(layer.name)

            previous_layer = layer
            
        for skip_con in self.skip_connect_list:
            #search last layer before start layer
            before_start_layer_name = self.layer_list[0].name
            for single_layer in entries_in_arch_list:
                if skip_con.start_layer.name in single_layer:
                    break
                else:
                    before_start_layer_name = single_layer
            arch.append(to_curve(before_start_layer_name, "sum_" + skip_con.start_layer.name))
            #arch.append(to_skip( of=before_start_layer_name, to="sum_" + skip_con.start_layer.name, pos=5.5))
        
        #arch.append(to_SoftMax("dense1", 4096 ,"(2,0,0)", left_neighbour, depth=35))
        #arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), "dense1"))
        #arch.append(to_SoftMax("dense2", 2048 ,"(1,0,0)", "(dense1-east)", depth=25))
        #arch.append(to_connection("dense1", "dense2"))
        arch.append(to_SoftMax("softmax", 10 ,"(1,0,0)",left_neighbour, depth=4))
        arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), "softmax"))
        left_neighbour = gen_neighbour_name("softmax")

        arch.append(to_SoftMax("softout", 1 ,"(1,0,0)",left_neighbour, caption="Output", depth=3))
        arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), "softout"))


        #arch.append(to_SoftMax("soft1", 10 ,"(2,0,0)", left_neighbour, caption="SOFT"))
        #arch.append(to_connection(get_rid_of_neighbour_name(left_neighbour), "soft1"))
        arch.append(to_end())
        to_generate(arch, output_path)


def get_newly_inserted_layer(path_to_net) -> Layer:
    if not "added" in path_to_net:
        #raise ValueError("get_newly_inserted_layer called for a net that has no newly inserted layer!")
        return Layer("DummyLayer_JustAPlaceholder!", 0, "None")
    else:
        cur_net = gen_net_from_json_path(path_to_net)
        max_insertion_index = -1
        for layer in cur_net.layer_list:
            if "New_" in layer.name and isinstance(layer, Conv2D_Layer) and layer.insertion_step > max_insertion_index:
                max_insertion_index = layer.insertion_step
        for layer in cur_net.layer_list:
            if "New_" in layer.name and isinstance(layer, Conv2D_Layer) and layer.insertion_step == max_insertion_index:
                return layer
    raise ValueError("Layer with max_insertion_index = " + str(max_insertion_index) + " not found!")

def gen_net_from_json_path(path_to_json_net, old_format=False) -> Net:
    """
    Generates a net from a path to a json file.
    """
    if old_format:
        return alive_json_to_net(path_to_json_net)
    new_layer_list = []
    new_skip_connect_list = []
    with open(path_to_json_net, 'r') as json_file:
        json_net = json.load(json_file)
        for json_layer in json_net["layer_list"]:
            json_layer_type = json_layer["layer_type"]
            if json_layer_type == "Conv2D":
                new_layer_list.append(Conv2D_Layer(name=json_layer["name"],
                                                   insertion_step=json_layer["insertion_step"],
                                                   channel_number=json_layer["channel_number"],
                                                   filter_size=json_layer["filter_size"],
                                                   filter_size_x=json_layer["filter_size_x"],
                                                   filter_size_y=json_layer["filter_size_y"],
                                                   insertion_index=json_layer["insertion_index"],
                                                   stride=json_layer["stride"]))
            elif json_layer_type == "Dense":
                new_layer_list.append(Dense_Layer(name=json_layer["name"],
                                                  insertion_step=json_layer["insertion_step"],
                                                  neuron_number=json_layer["neuron_number"]))
            elif json_layer_type == "Pool":
                new_layer_list.append(Pool_Layer(name=json_layer["name"],
                                                 insertion_step=json_layer["insertion_step"],
                                                 pooling_type=json_layer["pooling_type"]))
            else:
                raise ValueError("Layertype " + json_layer_type + " not defined in gen_net_from_json_str function!\n")

        for json_skip_connect in json_net["skip_connect_list"]:
            start_layer_set = False
            target_layer_set = False
            for layer in new_layer_list:
                if layer.name == json_skip_connect["start_layer"]["name"]:
                    start_layer = layer
                    start_layer_set = True
                if layer.name == json_skip_connect["target_layer"]["name"]:
                    target_layer = layer
                    target_layer_set = True
            if not target_layer_set or not start_layer_set:
                raise ValueError("Start or target layer of Skip_Connection " + json_skip_connect["name"] + " not found!")

            new_skip_connect_list.append(Skip_Connection(name=json_skip_connect["name"],
                                                         start_layer=start_layer,
                                                         target_layer=target_layer,
                                                         insertion_step=json_skip_connect["insertion_step"]))
    return Net(new_layer_list, new_skip_connect_list, source_path=path_to_json_net)

def get_insertion_step_from_name(name: str):
    if "New_Conv2D" in name:
        return name[11:15]
    elif "Init_Conv2D" in name:
        #return name[12:16]
        return "0000"
    elif "Init_Dense" in name:
        #return name[11:15]
        return "0000"
    elif "Init_Pool" in name:
        return "0000"
    elif "New_SkipConnect" in name:
        return name[16:20]
    elif "Init_SkipConnect" in name:
        return "0000"

def split_skip_connect_value(name: str):
    start_layer = name.partition("&")[0]
    target_layer = name.partition("&")[2]
    return start_layer, target_layer

def get_layer_from_name(name: str, layer_list: List[Layer]) -> Layer:
    for layer in layer_list:
        if name in layer.name:
            return layer
    raise ValueError("No Layer with name " + name + "found!")

def alive_json_to_net(path_to_file) -> Net:
    """
    Imports a json file as a net. Depricated! Only used for importing alive file
    :param path_to_file: Path to the net which should be read.
    :return: List of all layers as first argument and all skip_connections as second argument.
    """
    with open(path_to_file) as f:
        data = json.load(f)
    layers = []
    skip_connections = []
    def get_name_before_slash(name: str) -> str:
        return name.split("/")[0]

    for layer_name in data:
        if "Init_Conv2D" in layer_name or "New_Conv2D" in layer_name:
            layers.append(Conv2D_Layer(get_name_before_slash(layer_name), get_insertion_step_from_name(layer_name), data[layer_name]))
        elif "Dense" in layer_name:
            layers.append(Dense_Layer(get_name_before_slash(layer_name), get_insertion_step_from_name(layer_name), data[layer_name]))
        elif "Init_Pool" in layer_name:
            layers.append(Pool_Layer(get_name_before_slash(layer_name), get_insertion_step_from_name(layer_name), data[layer_name]))
        elif "SkipConnect" in layer_name:
            start_layer_name, target_layer_name = split_skip_connect_value(data[layer_name])
            start_layer = get_layer_from_name(start_layer_name, layers)
            target_layer = get_layer_from_name(target_layer_name, layers)
            skip_connections.append(Skip_Connection(get_name_before_slash(layer_name), start_layer, target_layer, get_insertion_step_from_name(layer_name)))
    return Net(layers, skip_connections, source_path=path_to_file)


class Layer_Weight:
    def __init__(self, layer_name: str, weight: float):
        self.layer_name = layer_name
        self.weight = weight

    def __str__(self):
        return "Das Layer " + self.layer_name + " hat ein Gesamtgewicht von " + str(self.weight) + "."


def read_summed_weights(path_to_summed_weights) -> List[Layer_Weight]:
    with open(path_to_summed_weights) as weight_file:
        weight_reader = csv.reader(weight_file, delimiter=',')
        weights = []
        for line in weight_reader:
            weights.append(Layer_Weight(line[0], float(line[1])))
    return weights


def last_adding_gave_accuracy_boost(Default_directory, curr_run: int) -> bool:
    if curr_run % (config.pipeline_parameters.number_layer_steps + 1) == 1:
        return not pipeline_parameters.delete_after_shrinkage_step
    else:
        df = pd.read_csv(os.path.join(Default_directory, "test_accuracy.csv"))
        second_to_last_acc = df.tail(2).Current_run.values[0]
        last_acc = df.tail(2).Current_run.values[1]
        if last_acc > second_to_last_acc + pipeline_parameters.layer_lasso_momentum_threshold:
            return True
        else:
            return False


def delete_layers_under_threshold(net: Net, path_to_summed_weights: str, path_to_del_net: str, deleting_threshold: float):
    old_layer_list = net.layer_list
    old_skip_connects = net.skip_connect_list
    weights = read_summed_weights(path_to_summed_weights)

    deleted_insertion_steps = []
    # Check for layers which are under the threshold
    for layer in old_layer_list:
        for weight in weights:
            if weight.layer_name in layer.name:
                if weight.weight < deleting_threshold:
                    layer.will_be_kicked = True
                    layer.kicked_reason = "under threshold"
                    deleted_insertion_steps.append(layer.insertion_step)
                break

    net.delete_list_of_layers(deleted_insertion_steps)
    # ToDo
    net.write_to_json_file(path_to_del_net)


def layer_delete_routine(Default_directory, curr_run: int, path_to_net, deleting_threshold: float, delete_before_training=False):
    """
    This function is used to delete layers which have the sum of weights under a threshold
    :param Default_directory: The path where the "Netze"-folder is located
    :param curr_run: The number of the current run
    :param path_to_net: The path to the net from which layers should be deleted
    :param deleting_threshold: The threshold under which layers will be kicked
    :param delete_before_training: Is this function called before the training with the curr_run is done?
    :return: The path to the net with the deleted Layer(s)
    """
    time.sleep(pipeline_parameters.waiting_time_before_delete)
    path_to_del_net = os.path.join(Default_directory, 'Netze', str(curr_run) + '_02_deleted_net.json')
    if pipeline_parameters.use_layer_lasso_momentum and last_adding_gave_accuracy_boost(Default_directory, curr_run):
        copyfile(path_to_net, path_to_del_net)
    else:
        import_run = curr_run
        if delete_before_training:
            import_run -= 1
        path_to_summed_layer_weights = os.path.join(Default_directory, 'Netze', 'layer_weights_summed_' + str(import_run) + ".csv")
        net = gen_net_from_json_path(path_to_net)
        net.mark_layers_under_threshold(path_to_summed_layer_weights, path_to_del_net, deleting_threshold)
        net.mark_skips_and_layers_in_same_block()
        net.del_marked_layers_and_skips()

        #delete_layers_under_threshold(net, path_to_summed_layer_weights, path_to_del_net, deleting_threshold)
    return path_to_del_net


def gen_conv_layer_name(insertion_step: int, number_of_layer_in_block: int) -> str:
    #return "New_Conv2D_" + "%04d" % (insertion_step) + "_" + "%04d" % (number_of_layer_in_block + 1000) + "/Conv2D"
    return "New_Conv2D_" + "%04d" % (insertion_step) + "_" + "%04d" % (number_of_layer_in_block + 1000)

def gen_init_conv_layer_name(number_of_layer_in_block: int) -> str:
    #return "New_Conv2D_" + "%04d" % (insertion_step) + "_" + "%04d" % (number_of_layer_in_block + 1000) + "/Conv2D"
    return "Init_Conv2D_" + "%04d" % (number_of_layer_in_block)


def gen_skip_connect_name(insertion_step: int) -> str:
    return "New_SkipConnect_" + "%04d" % (insertion_step) + "_1000_Start/Skip"

def gen_init_skip_connect_name(insertion_step: int, initial_position: int) -> str:
    return "New_SkipConnect_" + "%04d" % (insertion_step) + "_" + "%04d" % (initial_position) +"_Start/Skip"


def two_nets_are_equal(net1: Net, net2: Net) -> bool:
    # They can only be equal if the lengths are the same
    if len(net1.layer_list) == len(net2.layer_list):
        # Go over all layers in both lists
        for index, layer1 in enumerate(net1.layer_list):
            layer2 = net2.layer_list[index]
            if not type(layer1) == type(layer2):
                return False
            else:
                if isinstance(layer1, Conv2D_Layer):
                    chan_nr_1 = layer1.channel_number
                    chan_nr_2 = layer2.channel_number
                    # If one channel number is not equal to the other one the nets are not equal
                    if not chan_nr_1 == chan_nr_2:
                        return False
                elif isinstance(layer1, Pool_Layer):
                    pooltype_1 = layer1.pooling_type
                    pooltype_2 = layer2.pooling_type
                    if not pooltype_1 == pooltype_2:
                        return False
                elif isinstance(layer1, Dense_Layer):
                    neuron_nr_1 = layer1.neuron_number
                    neuron_nr_2 = layer2.neuron_number
                    if not neuron_nr_1 == neuron_nr_2:
                        return False
                else:
                    raise ValueError("Layertype " + str(type(layer1)) + "not known!")
        return True
    else:
        return False


def get_net_position_in_list(single_net: Net, net_list: List[Net]) -> int:
    """
    Returns the position of the single_net in the net_list
    """
    for index, l_net in enumerate(net_list):
        if two_nets_are_equal(single_net, l_net):
            return index
    return -1


def layer_adding_routine(Default_directory, curr_run, path_to_net, logfile_path, used_nets: List[Net]) -> str:
    """
    Depricated!
    Adds the structure which is specified in config.py to the net which is in path_to_net
    :param Default_directory: The directory where things like the output net is saved.
    :param curr_run: The current run for example; for example to generate names of layers and skip_connections.
    :param path_to_net: The path to the net which should be specified.
    :param logfile_path: Where wo log the events?
    :return: The path where the new (added) net is saved.
    """
    curr_net = gen_net_from_json_path(path_to_net)
    for i in range(1,config.pipeline_parameters.attemps_to_get_new_architecture):
        possible_curr_net = deepcopy(curr_net)
        possible_curr_net.add_structure(Default_directory, curr_run, logfile_path)
        # Test if possible_curr_net has been trained before
        if get_net_position_in_list(possible_curr_net, used_nets) == -1:
            break
    else:
        raise ValueError("attemps_to_get_new_architecture must be a positive integer!")
    path_to_added_net = os.path.join(Default_directory, 'Netze', str(curr_run) + '_03_added_net.json')
    possible_curr_net.write_to_json_file(path_to_added_net)
    return path_to_added_net


def morphnet_call_function(Default_directory, curr_run, path_to_net, with_dummy_training = False):
    expand_layers = True
    # Do MorphStep Shrinking/Expanding Step
    path_to_alive_file = os.path.join(Default_directory, 'Netze', 'learned_structure', 'alive_' + str(curr_run))
    path_to_shrinked_net = os.path.join(Default_directory, 'Netze', str(curr_run) + '_04_shrinked_net.json')
    # ToDo: Die flopcost-files so aufraeumen, dass dummy-trainings nicht in den anderen flopcost-files erfasst werden
    if with_dummy_training:
        path_to_flopcosts = os.path.join(Default_directory, 'Netze', 'learned_structure', 'morph_flopcosts.csv')
    else:
        path_to_flopcosts = os.path.join(Default_directory, 'Netze', 'learned_structure', 'flopcosts.csv')
    curr_net = gen_net_from_json_path(path_to_net)
    # curr_net.morphnet_shrinkage_routine(path_to_alive_file, path_to_shrinked_net, path_to_flopcosts,
    #                                     expanding=expand_layers)
    curr_net.morphnet_routine(path_to_alive_file, path_to_shrinked_net, path_to_flopcosts, expanding=expand_layers)
    return path_to_shrinked_net

