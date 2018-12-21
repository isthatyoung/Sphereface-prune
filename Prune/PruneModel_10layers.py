import sys

import caffe
import re
import google.protobuf.text_format as txtf
from caffe.proto import caffe_pb2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    caffe.set_device(2)
    caffe.set_mode_gpu()
    before_prune_deploy = ''
    before_prune_model=''
    before_prune_caffemodel = ''
    before_prune_solver_proto = ''
    before_prune_additional_solver_proto = ''
    before_prune_solver_proto2 = ''
    prune_solver_proto=''
    output_model = ''
    prune_layers=['conv1_1','conv2_2','conv3_2','conv3_4','conv4_1']## because of residual block structure, not prune all the layer
    been_saved_layers=[]
    compress_rate=0.75
    save_indexes = {}#save residual channel after pruning
    prune_indexes = {}

    for pruning_layer_name in prune_layers:
        if (len(been_saved_layers) == 0):
            print("########## Now pruning conv1_1 ##########")
            before_prune_solver = caffe.SGDSolver(before_prune_solver_proto)
            before_prune_solver.net.copy_from(before_prune_caffemodel)
        else:
            # before_prune_solver=caffe.SGDSolver(before_prune_solver_proto2)
            print("########## Now pruning " + pruning_layer_name + " ##########")
            before_prune_solver.net.copy_from(output_model)

        save_layers = before_prune_solver.net.params.keys()[:]
        ##TODO Copy training parameterts of certain layers to another empty matrix
        weight_temp2 = before_prune_solver.net.params[pruning_layer_name][0].data
        weight_temp = np.zeros(before_prune_solver.net.params[pruning_layer_name][0].shape, np.float32)
        bias_temp2 = before_prune_solver.net.params[pruning_layer_name][1].data
        bias_temp = np.zeros(before_prune_solver.net.params[pruning_layer_name][1].shape, np.float32)
        np.copyto(weight_temp, weight_temp2)
        np.copyto(bias_temp, bias_temp2)
        ##TODO Select pruned channel by greed algorithm and set the values of pruned channel in the matrix as zero by index
        weight_temp, bias_temp, prune_indexes[pruning_layer_name] = prune_dense(before_prune_solver.net,
                                                                                pruning_layer_name, weight_temp,
                                                                                bias_temp, compress_rate, save_indexes)
        np.copyto(before_prune_solver.net.params[pruning_layer_name][0].data, weight_temp)
        np.copyto(before_prune_solver.net.params[pruning_layer_name][1].data, bias_temp)
        ChangedeployPrototxt(before_prune_deploy, pruning_layer_name, save_indexes)
        ChangedeployPrototxt(before_prune_model, pruning_layer_name, save_indexes)
        prune_net = caffe.Net(before_prune_model,
                              caffe.TEST)  # generate empty caffemodel as same structure with deploy prototxt
        ##TODO copy all the parameters from original .caffemodel to empty ./caffemodel
        for saving_layer_name in save_layers:
            print("Now saving " + saving_layer_name + " parameters")

            if ('relu' in saving_layer_name):
                ## TODO copy relu layers parameters
                if (saving_layer_name == save_layers[save_layers.index(pruning_layer_name) + 1]):
                    weight = before_prune_solver.net.params[saving_layer_name][0]
                    weight = weight.data[save_indexes[pruning_layer_name]]
                    np.copyto(prune_net.params[saving_layer_name][0].data, weight)
                else:
                    weight = before_prune_solver.net.params[saving_layer_name][0]
                    weight = weight.data
                    np.copyto(prune_net.params[saving_layer_name][0].data, weight)
            else:
                SaveModelParameter(saving_layer_name, pruning_layer_name, before_prune_solver.net, prune_net,
                                   save_indexes, save_layers, been_saved_layers)
        prune_net.save(output_model)
        
        before_prune_solver = caffe.SGDSolver(before_prune_solver_proto)
        before_prune_solver.net.copy_from(output_model)
        ##Finetuning after pruning every layer
        for i in range(1768):
            before_prune_solver.step(1)
        before_prune_solver.net.save(output_model)

        ##Additional one epoch finetuning for conv4_1
        # if(pruning_layer_name=='conv4_1'):
        #     before_prune_solver = caffe.SGDSolver(before_prune_additional_solver_proto)
        #     before_prune_solver.net.copy_from(output_model)
        #     for i in range(1768):
        #         before_prune_solver.step(1)
        #
        # before_prune_solver.net.save(output_model)
        been_saved_layers.append(pruning_layer_name)

    ##Last round finetuning
    retrain_solver = caffe.SGDSolver(prune_solver_proto)
    retrain_solver.net.copy_from(output_model)
    retrain_pruned(retrain_solver, output_model)


def ChangedeployPrototxt(prototxtfile, layer_name, save_indexes):
    net = caffe_pb2.NetParameter()
    fn = prototxtfile
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)
        net.name = 'SpherefaceNet-10'
        layerNames = [l.name for l in net.layer]
        idx = layerNames.index(layer_name)
        l = net.layer[idx]
        num_output = len(save_indexes[layer_name])
        l.convolution_param.num_output = num_output
        outFn = prototxtfile
    with open(outFn, 'w')as f:
        net = str(net)
        print(net)
        pattern = re.compile(r'std: [-+]?[0-9]*\.?[0-9]+')
        net_final = pattern.sub('std: 0.01', net)
        print(net_final)
        f.write(net_final)


def ChangeLayer_Lr(prototxtfile, layer_name):
    net = caffe_pb2.NetParameter()
    fn = prototxtfile
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)
        net.name = 'SpherefaceNet-10'
        layerNames = [l.name for l in net.layer]
        idx = layerNames.index(layer_name)
        l = net.layer[idx]
        l.param[0].lr_mult = 10
        l.param[1].lr_mult = 20
        outFn = prototxtfile
        with open(outFn, 'w')as f:
            f.write(str(net))


def RestoreLayer_Lr(prototxtfile, layer_name):
    net = caffe_pb2.NetParameter()
    fn = prototxtfile
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)
        net.name = 'SpherefaceNet-10'
        layerNames = [l.name for l in net.layer]
        idx = layerNames.index(layer_name)
        l = net.layer[idx]
        l.param[0].lr_mult = 1
        l.param[1].lr_mult = 2
        outFn = prototxtfile
        with open(outFn, 'w')as f:
            f.write(str(net))


def SaveModelParameter(layer_name, pruning_layer_name, Net, prune_net, save_indexes, save_layers, been_saved_layers):
    if (layer_name not in been_saved_layers):
        if (layer_name == pruning_layer_name):
            weight, bias = Net.params[layer_name]
            weight, bias = weight.data, bias.data
            np.copyto(prune_net.params[layer_name][0].data, weight[save_indexes[layer_name], :, :, :])
            np.copyto(prune_net.params[layer_name][1].data, bias[save_indexes[layer_name]])

        ##This layer's input must be the same with the output of last convolution layer
        elif (save_layers[save_layers.index(layer_name) - 2] == pruning_layer_name):
            if (layer_name == 'fc5'):
                weight, bias = Net.params[layer_name]
                print("Innerproduct layer will be reshape to 512X512X7X6")
                weight = weight.data.reshape(512, 512, 7, 6)
                weight, bias = weight[:, save_indexes[pruning_layer_name], :, :], bias.data
                weight = weight.reshape(512, len(save_indexes[pruning_layer_name]) * 7 * 6)
                np.copyto(prune_net.params[layer_name][0].data, weight)
                np.copyto(prune_net.params[layer_name][1].data, bias)

            else:
                weight, bias = Net.params[layer_name]
                weight, bias = weight.data[:, save_indexes[pruning_layer_name], :, :], bias.data
                np.copyto(prune_net.params[layer_name][0].data, weight)
                np.copyto(prune_net.params[layer_name][1].data, bias)
        else:
            if (layer_name == 'fc6'):
                weight = Net.params[layer_name][0]
                weight = weight.data
                np.copyto(prune_net.params[layer_name][0].data, weight)
            else:
                weight, bias = Net.params[layer_name]
                weight, bias = weight.data, bias.data
                np.copyto(prune_net.params[layer_name][0].data, weight)
                np.copyto(prune_net.params[layer_name][1].data, bias)

    else:
        weight, bias = Net.params[layer_name]
        weight, bias = weight.data, bias.data
        np.copyto(prune_net.params[layer_name][0].data, weight)
        np.copyto(prune_net.params[layer_name][1].data, bias)


def retrain_pruned(solver, prune_caffemodel):
    for i in range(9):
        solver.step(1768)
    solver.net.save(prune_caffemodel)
    print("Pruned model retrain finish")


### A greedy algorithm ###
def greedy(channel_list, remove_channel_number):
    # todo: plus each two number in the list, and choose the min group as candidate group
    # backup_channel_list=np.array(channel_list)
    channel_list = list(channel_list)
    backup_channel_list = []
    for channel_sum in channel_list:
        backup_channel_list.append(channel_sum)
    num = []
    save_channel = []
    pruned_channel = []

    for i in range(remove_channel_number / 2):  # todo:each round will choose two alternative number
        channel_list, N = extract(channel_list)
        for x in N:
            num.append(x)
    for P_channel in num:
        # prune_index=np.argwhere(backup_channel_list == P_channel)
        prune_index = backup_channel_list.index(P_channel)
        pruned_channel.append(prune_index)
    pruned_channel.sort()
    for S_channel in channel_list:
        # save_index=np.argwhere(backup_channel_list == S_channel)
        save_index = backup_channel_list.index(S_channel)
        save_channel.append(save_index)
    save_channel.sort()
    return save_channel, pruned_channel


def extract(rand):
    # todo: plus each two number in the rand list, and choose the min group as candidate group
    L = np.zeros((len(rand) * (len(rand) - 1), 3))
    i = 0
    for x in rand:
        for y in rand:
            if y != x:
                L[i][0] = x
                L[i][2] = y
                L[i][1] = abs(x + y)
                i += 1
    df = pd.DataFrame(L)
    df.columns = range(len(df.columns))
    min_num = df[1].min()
    l_num = list(df[df[1] == min_num].values[0])
    l_num.remove(min_num)

    for x in l_num:
        rand.remove(x)
    return rand, l_num


def prune_dense(Net, layer_name, weight_matrix, bias_matrix, rate, save_indexes):
    # Net.forward()
    Net.forward()
    Net.backward()
    feature_maps = Net.blobs[layer_name].data
    features_sum_on_channel = np.sum(feature_maps, axis=(0, 2, 3))
    print("feature map sum :")
    print(features_sum_on_channel)
    num_output = feature_maps.shape[1]
    print("Before:" + " Total " + str(num_output) + " filters")
    T_count = int(num_output * (1 - rate))
    save_indexes[layer_name], prune_indexes = greedy(features_sum_on_channel, T_count)
    weight_matrix[prune_indexes, :, :, :] = 0
    bias_matrix[prune_indexes] = 0
    print("After:" + "Total " + str(len(save_indexes[layer_name])) + " filters")
    print("Pruned channel is :")
    print(prune_indexes)
    return weight_matrix, bias_matrix, prune_indexes


if __name__ == '__main__':
    main()



