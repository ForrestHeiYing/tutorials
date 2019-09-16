#!usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author:chaowei
@file: main.py
@time: 2019/07/18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from matplotlib import pyplot
import numpy as np
import time

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python import model_helper


def workspace_test():

    print("current blobs in the workspace : {}".format(workspace.Blobs))
    print("Workspace has blob 'X' ?: {}".format(workspace.HasBlob("X")))

    X = np.random.randn(2, 3).astype(np.float32)
    print("Generated X from numpy: \n{}".format(X))
    workspace.FeedBlob("X", X)

    print("current blobs in the workspace:{}".format(workspace.Blobs()))
    print("Workspace has blob 'X' ?{}".format(workspace.HasBlob("X")))
    print("Fethched X:\n{}".format(workspace.FetchBlob("X")))

    print("current workspace: {}".format(workspace.CurrentWorkspace()))
    print("current blobs in the workspace: {}".format(workspace.Blobs))

    workspace.SwitchWorkspace("gutentag", True)  # switch the workspace. The second
    print("After Switch Workspace ................")
    print("current workspace:{}".format(workspace.CurrentWorkspace()))
    print("current blobs in the workspace:{}".format(workspace.Blobs()))


def operators_test():
    # create operator.
    op = core.CreateOperator(
        "Relu",  # The type of operator that we want to run.
        ["X"],  # A list of input blobs by their names
        ["Y"],  # A list of output blobs ...
    )

    print("Type of the created op is: {}".format(type(op)))
    print("content: \n")
    print(str(op))

    workspace.FeedBlob("X", np.random.randn(2, 3).astype(np.float32))
    print("current blobs in the workspace:{}\n".format(workspace.Blobs()))
    workspace.RunOperatorOnce(op)
    print("current blobs in the workspace:{}\n".format(workspace.Blobs()))
    print("X:\n{}\n".format(workspace.FetchBlob("X")))
    print("Y:\n{}\n".format(workspace.FetchBlob("Y")))

    print("Expected:\n{}\n".format(np.maximum(workspace.FetchBlob("X"), 1)))

    op1 = core.CreateOperator(
        "GaussianFill",
        [],
        ["W"],
        shape=[100, 100],
        mean=1.0,
        std=1.0,
    )
    print("content of op1:\n")
    print(str(op1))

    workspace.RunOperatorOnce(op1)
    temp = workspace.FetchBlob("W")
    print("temp=", temp)
    # pyplot.hist(temp.flatten(), bins=50)
    # pyplot.title("ddd of Z")


def model_helper_test():
    data = np.random.rand(16, 100).astype(np.float32)  # create the input data
    label = (np.random.rand(16)*10).astype(np.int32)  # create the label
    workspace.FeedBlob("data", data)
    workspace.FeedBlob('label', label)

    m = model_helper.ModelHelper(name="my_first_net")  # create model

    weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
    bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

    fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
    pred = m.net.Sigmoid(fc_1, "pred")
    softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

    print("m.net=", m.net.Proto())

    print("m.param_init_net.Proto=", m.param_init_net.Proto())

    workspace.RunNetOnce(m.param_init_net)


    pass


if __name__ == '__main__':

    workspace_test()
    # operators_test()
    # model_helper_test()
    pass