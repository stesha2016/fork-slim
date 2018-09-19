#coding:utf-8
from flask import request, Flask
import os
import predict
import cv2
app = Flask(__name__)

@app.route("/", methods=['POST'])
def get_frame():
  upload_file = request.files['file']
  old_file_name = upload_file.filename
  file_path = os.path.join('./uploaded', old_file_name)

  if upload_file:
    upload_file.save(file_path)
    result = predict.server_predict(file_path)
    return result
  else:
    return 'failed'

if __name__ == "__main__":
  app.run("0.0.0.0", port=5000)