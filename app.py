import cv2
import numpy as np
import base64
from flask import Flask, json, request, jsonify, Response,make_response
from flask_cors import CORS
from ocr import *
from detection import *

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True
index_class={"cpt1":5,"cpt2":6,"cpt3":7}

@app.route('/api/detection', methods=['GET','POST'])
def detect():
    try:
        imageb = request.get_json()['image']
    except:
         return jsonify({'message':'no image sent!'})
    try:
        with open("getImage.png", "wb") as fh:
            fh.write(base64.b64decode(str(imageb)))
        
    except:
        return jsonify({'message':'error image format!'})
    try:
        image=cv2.imread('getImage.png')
        detected_image,classes = detection(image)
        # cv2.imshow("det",detected_image)
        # cv2.waitKey(0)
        index,scores=ocr(detected_image)
        # print(index)
        for class_det in classes:
            # print(index_class[class_det])
            # print(classes)
            if not class_det:
                return jsonify({'message':'no image detected'})
            if len(index)==index_class[class_det]:
                cv2.imwrite('postImage.png',detected_image)
                with open('postImage.png', "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                return jsonify({'index':index,'scores':scores})
            else:
                return jsonify({'message':'ocr nonfunctional'})
        
    except:
        return jsonify({'message':'ocr or detection error'})



if __name__ == "__main__":
    app.run()


