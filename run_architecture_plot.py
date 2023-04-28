import os
import customFunctions.changeArchitecture as cc
from shutil import copyfile
from config import environment_parameters
from pdf2image import convert_from_path

#path_to_tex_file = "/home/burghoff/Daten/220824_debug_plotNN/output.tex"
#os.system("pdflatex " + path_to_tex_file)

#dir_to_many_nets = "/home/burghoff/Daten/220825_plotNets/Animals10/nets"
#dir_to_many_nets = "/home/burghoff/Daten/220418_newPlotFunctions/210901_v01_regs_10_10_8_morphLR025/CIFAR10/Netze"
#dir_to_many_nets = "/home/burghoff/Daten/230120_plottingArchi/220808_shortRun_CIFARResNet18_lowReg"
dir_to_many_nets = "/home/burghoff/Daten/230214_emptyNetPlot/nets"
dir_to_many_nets = "/home/burghoff/Daten/230425_CIFAR10_EN_nets/"
layers_folder = os.path.join(dir_to_many_nets, "layers")
path_of_project_folder_layer = os.path.join(environment_parameters.project_folder, "git_repo_plotNN", "layers")
#path_of_project_folder = "/home/burghoff/PyCharmProjects/PyCharm_AutoMLwithMorphNet/git_plot_NN/layers"
#files_to_copy = ["layers/Ball.sty", "layers/Box.sty", "layers/init.tex", "layers/RightBandedBox.sty"]
files_to_copy = ["Ball.sty", "Box.sty", "init.tex", "RightBandedBox.sty"]


if not os.path.exists(layers_folder):
        os.makedirs(layers_folder)
        for file in files_to_copy:
            copyfile(os.path.join(path_of_project_folder_layer, file), os.path.join(layers_folder, file))


for file in os.listdir(dir_to_many_nets):
    if file.endswith(".json"):
        path_to_json_file = os.path.join(dir_to_many_nets, file)
        #net = cc.alive_json_to_net(path_to_json_file)
        net = cc.gen_net_from_json_path(path_to_json_file)
        output_filename = file[:-4]
        output_filename = output_filename + "tex"
        path_to_output_file = os.path.join(dir_to_many_nets, output_filename)
        net.plot_net_as_tex(path_to_output_file)
        os.system("pdflatex -output-directory=" + dir_to_many_nets + " " + path_to_output_file)

        images = convert_from_path(path_to_output_file[:-4]+".pdf")
        for i in range(len(images)):
            # Save pages as images in the pdf
            images[i].save(path_to_output_file[:-4] + "_" + str(i) +'.jpg', 'JPEG')