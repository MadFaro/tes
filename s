======================================================================
v3_rnnt_encoder.int8.onnx

INPUTS:
name = audio_signal | shape = ['batch_size', 64, 'seq_len'] | type = tensor(float)
name = length | shape = ['batch_size'] | type = tensor(int64)

OUTPUTS:
name = encoded | shape = ['batch_size', 768, 'Transposeencoded_dim_2'] | type = tensor(float)
name = encoded_len | shape = ['batch_size'] | type = tensor(int32)

======================================================================
v3_rnnt_decoder.int8.onnx

INPUTS:
name = x | shape = [1, 1] | type = tensor(int64)
name = h.1 | shape = [1, 1, 320] | type = tensor(float)
name = c.1 | shape = [1, 1, 320] | type = tensor(float)

OUTPUTS:
name = dec | shape = [1, 1, 320] | type = tensor(float)
name = h | shape = [1, 1, 320] | type = tensor(float)
name = c | shape = [1, 1, 320] | type = tensor(float)

======================================================================
v3_rnnt_joint.int8.onnx

INPUTS:
name = enc | shape = [1, 768, 1] | type = tensor(float)
name = dec | shape = [1, 320, 1] | type = tensor(float)

OUTPUTS:
name = joint | shape = [1, 1, 1, 34] | type = tensor(float)
