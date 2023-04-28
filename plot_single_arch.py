import os
import customFunctions.changeArchitecture as cc

conv1 = cc.Conv2D_Layer("Conv1", 0, 6, 5, 5, 5)
pool1 = cc.Pool_Layer("Pool1", 0, "max", 2, 2)
conv2 = cc.Conv2D_Layer("Conv2", 0, 16, 5, 5, 5)

layerlist = [conv1, pool1, conv2]
net = cc.Net(layerlist, [])
net.plot_net_as_tex("/home/burghoff/Daten/230308_PHS_paper/out.tex",highlight_newest_layer=False)