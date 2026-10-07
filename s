import onnxruntime as ort

path = r"C:\Users\tolog\Desktop\gig"

for name in [
    "v3_rnnt_encoder.int8.onnx",
    "v3_rnnt_decoder.int8.onnx",
    "v3_rnnt_joint.int8.onnx",
]:
    print("\n" + "=" * 70)
    print(name)

    model = ort.InferenceSession(
        path + "\\" + name,
        providers=["CPUExecutionProvider"]
    )

    print("\nINPUTS:")
    for x in model.get_inputs():
        print(
            "name =", x.name,
            "| shape =", x.shape,
            "| type =", x.type
        )

    print("\nOUTPUTS:")
    for x in model.get_outputs():
        print(
            "name =", x.name,
            "| shape =", x.shape,
            "| type =", x.type
        )
