#!usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author:chaowei
@file: net_test.py
@time: 2019/09/16
"""
from __future__ import print_function
from caffe2.python import core, workspace

net = core.Net("my_first_net")
print("current network proto:\n\n{}".format(net.Proto()))

X = net.GaussianFill([], ["X"], mean=0.0, std=1.0, shape=[2, 3], run_once=0)
print("new network proto:\n\n{}".format(net.Proto()))

W = net.GaussianFill([], ["W"], mean=0.0, std=1.0, shape=[5, 3], run_once=0)
print("new network proto:\n\n{}".format(net.Proto()))  # Now, there are two operators in the network.
b = net.ConstantFill([], ["b"], shape=[5, ], value=1.0, run_once=0)
print("new network proto:\n\n{}".format(net.Proto()))

Y = net.FC([X, W, b], ["Y"])
print("current network proto:\n\n{}".format(net.Proto()))
print("current network proto.name:\n\n{}".format(net.Proto().name))  # "my_first_net"


print("1 current blobs in the workspace: {}".format(workspace.Blobs()))  # it is empty.
# from caffe2.python import net_drawer
# from IPython import display
# graph = net_drawer.GetPydotGraph(net, rankdir="LR")
# display.Image(graph.create_png(), width=800)


def run_net_method1():
    workspace.ResetWorkspace()  # reset workspace
    print("2 current blobs in the workspace: {}".format(workspace.Blobs()))  # it is empty.
    print("3 current blobs in the workspace: {}".format(workspace.Blobs()))  # it is empty.
    workspace.RunNetOnce(net)  # Initialize the network, run the network, and then destroy the network.
    print("4 current blobs in the workspace: {}".format(workspace.Blobs()))  #
    for name in workspace.Blobs():
        print("{}:\n{}".format(name, workspace.FetchBlob(name)))

    pass


def run_net_method2():
    workspace.ResetWorkspace()
    print("2 current blobs in the workspace: {}".format(workspace.Blobs()))  # it is empty.
    workspace.CreateNet(net)  # Initialize the network
    workspace.RunNet(net.Proto().name)  # net.Proto().name: "my_first_net"  # run the network
    print(" blobs in the workspace after execution: {}".format(workspace.Blobs()))  # [u'W', u'X', u'Y', u'b']
    for name in workspace.Blobs():
        print("{}:\n{}".format(name, workspace.FetchBlob(name)))

    pass


if __name__ == '__main__':

    # run_net_method1()
    run_net_method2()

    pass