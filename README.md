\# Handwritten Digit Recognition



A convolutional neural network (CNN) built with PyTorch that recognizes handwritten digits (0-9). It is trained on the MNIST dataset and reaches about 99% test accuracy. You can also use it to predict digits from your own images.



\## Features

\- Trains a CNN on MNIST with light data augmentation

\- Predicts a single digit from your own image

\- Finds and predicts several digits in one image

\- Shows predictions on random test samples



\## Requirements

\- Python 3.9 or newer

\- Libraries listed in `requirements.txt`



\## Setup

```

git clone https://github.com/Ankita-m23/digit-recognition.git

cd digit-recognition

pip install -r requirements.txt

```



\## Usage



Train the model first (this creates `digit\_cnn.pt`):

```

python digit\_recognition.py train

```



Then use any of these:

```

python digit\_recognition.py samples

python digit\_recognition.py predict my\_digit.png

python digit\_recognition.py multi my\_digits.png

```



\## Tips for your own images

\- Draw with a thick pen or brush, one clear digit per image

\- Dark ink on light paper or white on black both work

\- For `multi`, leave small gaps between digits



\## Project files

\- `digit\_recognition.py`: model, training and prediction code

\- `requirements.txt`: Python dependencies

\- `my\_digit.png`: example image





