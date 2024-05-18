from tensorflow.lite.python.util import convert_bytes_to_c_source


with open('quantized_model.tflite', 'rb') as f:
    tflite_model = f.read()
source_text, header_text = convert_bytes_to_c_source(tflite_model,  "quantized_model.tflite")
with  open('model_c_files/model.h',  'w')  as  file:
    file.write(header_text)
with  open('model_c_files/model.cc',  'w')  as  file:
    file.write(source_text)