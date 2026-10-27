import onnx
from onnx import helper as h, TensorProto as T

node = h.make_node(
    "ConvInteger", ["x", "w", "xz", "wz"], ["y"],
    pads=[1, 1, 1, 1],
    strides=[1, 1],
    dilations=[1, 1],
)

graph = h.make_graph(
    [node], "repro",
    [h.make_tensor_value_info("x", T.UINT8, [1, 3, 2, 2])],
    [h.make_tensor_value_info("y", T.INT32, [1, 1, 2, 2])],
    initializer=[
        h.make_tensor("w", T.UINT8, [1, 3, 3, 3], [138] * 27),
        h.make_tensor("xz", T.UINT8, [], [10]),
        h.make_tensor("wz", T.UINT8, [], [137]),
    ],
)

model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, "repro.onnx")
