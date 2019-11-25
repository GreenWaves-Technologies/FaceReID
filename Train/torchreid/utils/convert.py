import os
import numpy as np

import torch
import onnx
from onnx_tf.backend import prepare

import tensorflow as tf
from tensorflow.python.platform import gfile


from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes


def validate_tf_model(inputs, valid_output, node_names = None, filename = "tf_net.pb"):
    sess = tf.Session()
    f = gfile.FastGFile(filename, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    tf.global_variables_initializer().run(session=sess)

    LOGDIR = 'tensorboard'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    x_tensor = sess.graph.get_tensor_by_name(node_names[0])
    output = sess.graph.get_tensor_by_name(node_names[1])
    feed_dict = {x_tensor: inputs}
    tf_output = sess.run([output], feed_dict =feed_dict)
    np.testing.assert_almost_equal(valid_output.data.cpu().numpy(), tf_output[0], decimal=3)


def optimize_tf_model_for_inference(input_names, output_names, input_filename="tf_net.pb",
                                    output_filename="tf_models/optimized_tf_net.pb"):
    input_graph_def = tf.GraphDef()
    input_f = gfile.FastGFile(input_filename, 'rb')
    input_graph_def.ParseFromString(input_f.read())
    node_map = {}
    for node in input_graph_def.node:
        if node.name not in node_map:
            node_map[node.name] = node
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names,
        output_names,
        dtypes.float32.as_datatype_enum)
    output_f = gfile.GFile(output_filename, "w")
    output_f.write(output_graph_def.SerializeToString())


def convert_to_onnx(model, args):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    num_channels = 3
    if args.grayscale:
        num_channels = 1
    dummy_input = torch.randn(1, num_channels, args.height, args.width)
    onnx_filename = args.load_weights[:args.load_weights.find('.')] + '.onnx'
    model.eval()
    torch_out = model(dummy_input)
    torch.onnx.export(model, dummy_input, onnx_filename,
                      input_names=['input'], output_names=['output'], export_params=True)
    # torch.onnx.export(model, dummy_input, )#,
    #                   #input_names=['input:0'], output_names=['output:0'], verbose=True)
    onnx_model = onnx.load(onnx_filename)
    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    tf_rep = prepare(onnx_model)  # Import the ONNX model to Tensorflow
    # Get corresponding names for input and output nodes from TensorFlow graph
    input_node_name = tf_rep.tensor_dict[tf_rep.inputs[0]].name
    output_node_name = tf_rep.tensor_dict[tf_rep.outputs[0]].name
    print(input_node_name, output_node_name)
    tf_out = tf_rep.run(dummy_input.data.numpy())
    np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), tf_out[0], decimal=3)
    tf_filename = "tf_net.pb"
    print("Saving TensorFlow model to ", tf_filename)
    tf_rep.export_graph(tf_filename)
    print("Validating saved model")
    validate_tf_model(dummy_input.detach().numpy(), torch_out, [input_node_name, output_node_name],
                      tf_filename)
    tf_filename = "new_tf_net.pb"
    optimized_tf_filename = "optimized_tf_net.pb"
    inputs = [input_node_name.split(":")[0]]
    outputs = [output_node_name.split(":")[0]]
    print("Optimizing TensorFlow model")
    optimize_tf_model_for_inference(inputs, outputs, input_filename=tf_filename,
                                    output_filename=optimized_tf_filename)
    print("Optimized model saved at ", optimized_tf_filename)
    print("Validating optimized model")
    validate_tf_model(dummy_input.detach().numpy(), torch_out, [input_node_name, output_node_name],
                      optimized_tf_filename)

    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(tf_filename, inputs, outputs)
    converter.post_training_quantize = True
    # converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_quant_model)
