from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(execution_path, "static/densenet121-a639ec97.pth"))
prediction.loadModel()
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "static/horse.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , ":" , eachProbability)