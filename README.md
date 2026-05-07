**Mammogram Tumor Identifier**

Try it here: fletcher-s.github.io/mammogram-tumor-identifier

What it does

This tool takes a mammogram image and highlights areas where the model thinks a tumor or mass may be present. It overlays the prediction on top of the image and also gives a confidence score along with the percentage of the image marked as abnormal.

Instead of only classifying an image as “tumor” or “no tumor,” the model performs image segmentation, meaning it tries to predict the exact region where suspicious tissue appears.

**How it works**
1. Training the model

The model was trained in Python using PyTorch on a dataset of mammogram images paired with segmentation masks. These masks showed the location of masses, allowing the network to learn which image features corresponded to abnormal regions.

2. Converting the model

After training, the model was exported to ONNX format so it could run outside of Python. ONNX makes it possible to use the model directly in a web browser without requiring users to install anything.

3. Running in the browser

The website uses ONNX Runtime Web to load and run the model locally in the browser. When an image is uploaded, it is resized and normalized the same way as during training, passed through the model, and the predicted mask is drawn over the original mammogram.

Everything runs locally on the device.

**Accuracy and limitations**

The model is able to roughly identify the location of masses in many cases, but the predictions are still fairly limited. Some of the main reasons are:

Small dataset — The model was trained on a relatively small number of mammogram images. More data would likely improve performance significantly.
Limited training — The model was not trained for very long due to hardware and time constraints. Longer training and better hyperparameter tuning could improve results.
Simple architecture — More advanced medical imaging models, such as a larger U-Net with a stronger encoder backbone, would probably capture finer detail better. Since this project was mainly built to learn how segmentation models work, I kept the architecture simpler so training and testing stayed manageable.

This project should not be used for medical diagnosis.

Project structure
mammogram-tumor-identifier.ipynb   # Training notebook
convert_to_onnx.py                 # Converts the trained model to ONNX
model.onnx                         # Exported model used in the browser
index.html                         # Browser interface
