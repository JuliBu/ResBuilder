from customFunctions.changeArchitecture import Net, Conv2D_Layer, Pool_Layer, Skip_Connection, Dense_Layer,\
    gen_init_conv_layer_name, gen_init_skip_connect_name

def gen_res_block_2_equal_layers(square_filter_size: int, channel_size: int):
    """Generates two convolutional layers with a skip connection and returns """
    print()


def gen_ResNet18():
    blocks_and_channels = [[64, 64], [128, 128], [256, 256], [512, 512]]
    number_of_blocks = [2,2,2,2]
    filter_sizes = [[3,3],[3,3],[3,3],[3,3]]
    return gen_any_ResNet(blocks_and_channels, number_of_blocks, filter_sizes)

def gen_ResNet34():
    blocks_and_channels = [[64, 64], [128, 128], [256, 256], [512, 512]]
    number_of_blocks = [3,4,6,3]
    filter_sizes = [[3,3],[3,3],[3,3],[3,3]]
    return gen_any_ResNet(blocks_and_channels, number_of_blocks, filter_sizes)

def gen_ResNet50():
    blocks_and_channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    number_of_blocks = [3,4,6,3]
    filter_sizes = [[1,3,1],[1,3,1],[1,3,1],[1,3,1]]
    return gen_any_ResNet(blocks_and_channels, number_of_blocks, filter_sizes)

def gen_ResNet101():
    blocks_and_channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    number_of_blocks = [3,4,23,3]
    filter_sizes = [[1,3,1],[1,3,1],[1,3,1],[1,3,1]]
    return gen_any_ResNet(blocks_and_channels, number_of_blocks, filter_sizes)

def gen_ResNet152():
    blocks_and_channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    number_of_blocks = [3,8,36,3]
    filter_sizes = [[1,3,1],[1,3,1],[1,3,1],[1,3,1]]
    return gen_any_ResNet(blocks_and_channels, number_of_blocks, filter_sizes)


def gen_any_ResNet(blocks_and_channels, number_of_blocks, filter_sizes):
    layer_list = []
    skip_list = []
    layer_list.append(Conv2D_Layer("Init_Conv2D_0001", 0, 64, 7, 7, 7, 1, 2))
    layer_list.append(Pool_Layer("Init_Pool_0001", 0, "max", 3, 2))
    #for index_stage, layer_stage in enumerate(number_of_blocks, start=1):
    for index_stage in range(len(number_of_blocks)):
        layer_counter_in_stage = 1
        for i in range(number_of_blocks[index_stage]):
            start_layer_set, target_layer_set = False, False
            for index_layer_in_block, layer_channel_number in enumerate(blocks_and_channels[index_stage]):
                layer_name = gen_init_conv_layer_name((index_stage+1)*1000+layer_counter_in_stage)
                filter_size = filter_sizes[index_stage][index_layer_in_block]
                tmp_layer = Conv2D_Layer(layer_name, 0, layer_channel_number, filter_size, filter_size,filter_size, 1)
                layer_list.append(tmp_layer)
                if index_layer_in_block == 0:
                    if layer_counter_in_stage == 1 and index_stage > 0:
                        tmp_layer.stride = 2
                    start_layer = tmp_layer
                    start_layer_set = True
                if index_layer_in_block == len(blocks_and_channels[index_stage])-1:
                    target_layer = tmp_layer
                    target_layer_set = True
                layer_counter_in_stage += 1

            if start_layer_set and target_layer_set:
                skip_name = gen_init_skip_connect_name(0, len(skip_list))
                skip_list.append(Skip_Connection(skip_name, start_layer, target_layer, 0))
            else:
                raise ValueError("Start_layer_set: " + str(start_layer_set) +
                                 " and target_layer_set: " + str(target_layer_set) +
                                 " but both has to be true!")
    layer_list.append(Pool_Layer("Pool_layer_before_fc", 1, "average"))
    layer_list.append(Dense_Layer("Dense_layer_before_softmax",1,1000))
    return Net(layer_list, skip_list)

def gen_NeraCare_VGG_alike_net():
    def gen_one_conv_layer(block_id, inner_id, nr_channels):
        return Conv2D_Layer("Conv2D_" + str(block_id) + "_" + str(inner_id), 0, nr_channels)
    def gen_one_maxPool_layer(block_id, inner_id=-1):
        if inner_id == -1:
            name = "MaxPool_"+ str(block_id)
        else:
            name = "MaxPool_"+ str(block_id) + "_" + str(inner_id)
        return Pool_Layer(name, 0, "max")
    layer_list = []
    layer_list.append(gen_one_conv_layer(1,1,64))
    layer_list.append(gen_one_maxPool_layer(1,1))
    layer_list.append(gen_one_conv_layer(1,2,64))
    layer_list.append(gen_one_maxPool_layer(1,2))

    layer_list.append(gen_one_conv_layer(2,1,128))
    layer_list.append(gen_one_conv_layer(2,2,128))
    layer_list.append(gen_one_maxPool_layer(2))

    layer_list.append(gen_one_conv_layer(3,1,256))
    layer_list.append(gen_one_conv_layer(3,2,256))
    layer_list.append(gen_one_maxPool_layer(3))

    layer_list.append(gen_one_conv_layer(4,1,512))
    layer_list.append(gen_one_conv_layer(4,2,512))

    layer_list.append(Dense_Layer("Dense_1", 0, 4096))
    layer_list.append(Dense_Layer("Dense_2", 0, 2048))
    layer_list.append(Dense_Layer("Dense_out", 0, 1))

    return Net(layer_list, [])


#this_net = gen_ResNet152()

this_net = gen_NeraCare_VGG_alike_net()
this_net.write_to_json_file("/home/burghoff/Daten/220917_plotVGGarchi/VGG16_alike.json")
print()