# Multi-label-emotions

Dataset Preprocessing – Steps:


In this project, we used the GoEmotions dataset. It consists of 29 emotions, each assigned to sentences. First, we cleaned the dataset by handling null values and removing duplicates. Then, class weighting was performed . After that, we applied tokenization to break down the text into tokens, and we padded the sentences to a fixed length. The dataset was split into training-testing split.

Model Selection and Rationale:


We chose the ALBERT model for this task, as it is a lighter version of BERT, based on the Transformer architecture, which is highly effective for text processing. ALBERT was fine-tuned on the GoEmotions dataset. Its smaller size allows for faster inference and lower resource consumption. For multi-label classification, we used a sigmoid activation function to predict independent probabilities for each label. The binary cross-entropy loss function was employed to optimize the model, which is suitable for multi-label tasks.

Challenges Faced and Solutions:


A major challenge was data imbalance, as some emotions were more common than others. We addressed this by adjusting class weights, improving the model's ability to predict rare emotions. Another challenge was handling multi-label classification, where a sentence could have multiple emotions. We solved this by using the sigmoid activation and binary cross-entropy loss. Overfitting was also an issue, which we tackled with dropout layers and early stopping. Finally, the limitation of computational resources was managed by using Google Colab with GPU support for efficient training but it it is till predicting some of emotions wrong . We should optimize its performance in near future with better resources iA .

You can check it out on :
https://huggingface.co/spaces/usaid123/multi-label-emotions-detector
