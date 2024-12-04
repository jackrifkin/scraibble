# ScrAIbble

## Document Overview
This repo includes a Scrabble-playing heuristic-based model with an accompanying Scrabble gym environment and GADDAG dictionary implementation.

The GADDAG class definitions are contained in the `gaddag.py` file, and include logic for traversing a Gaddag instance and verifying word validity

The `scrabble_gym.py` file contains a gym implementation of the game, Scrabble, and can be interacted with using the `reset()`, `step()` and `render()` methods.

A gradient descent training program is defined in the `descent_training.py` file. The program initializes the gym environment initialization, generates moves, evaluates heuristics, interacts with the gym environment, and plots the results. 

The `util.py` file contains methods used by both the gym and gradient descent training to evaluate shared metrics, and is where our logic for generating moves lies. 

## Virtual Environment Setup (optional)
(Only do the first step once, after you initially clone the repo)
1. Create a virtual environment using:
``python3 -m venv venv``
2. Activate your venv with:
``source venv/bin/activate``, or on Windows, ``venv\Scripts\activate``
3. Install the required packages:
   
   a. pip install numpy gym matplotlib scipy
   
   b. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

   Replace the above index URL with whichever version of Pytorch corresponds to your drivers, you can check at [this website](https://pytorch.org/get-started/locally/)

## Running the Program
This program requires that you have the SOWPODS.txt file in your project locally. Since the file is too large to include on GitHub, please download it and place it in your repo from [this link](https://web.mit.edu/jesstess/www/sowpods.txt)

Alternatively, run `curl https://web.mit.edu/jesstess/www/sowpods.txt -o SOWPODS.txt` in your terminal.

To run the gradient descent program, run `py3 .\descent_training.py` and input your preferred gradient descent parameters when prompted.
