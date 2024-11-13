import cv2
import typing
import numpy as np
import onnx
import onnxruntime
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, (128, 32))

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        session=onnxruntime.InferenceSession("D:/Projects/Handwritten text recognition/model.onnx")
        session.get_inputs()[0].shape
        session.get_outputs()[0].type
        input_name=session.get_inputs()[0].name
        output_name=session.get_outputs()[0].name
        preds = self.model.run(None, {input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("D:/Projects/Handwritten text recognition/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv(r"D:\Projects\Handwritten text recognition\tests.csv").values.tolist()

    accum_cer = []
    for image_path, label in df:
        image = cv2.imread(image_path)

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

        # resize by 4x
        #image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}")