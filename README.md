# ScrAIbble
This repo includes a Scrabble-playing heuristic-based model with an accompanying Scrabble gym environment and GADDAG dictionary implementation.

The GADDAG class definitions are contained in the `gaddag.py` file, and include logic for traversing a Gaddag instance and verifying word validity

The `scrabble_gym.py` file contains a gym implementation of the game, Scrabble, and can be interacted with using the `reset()`, `step()` and `render()` methods.

A gradient descent training program is defined in the `descent_training.py` file. The program initializes the gym environment initialization, generates moves, evaluates heuristics, interacts with the gym environment, and plots the results. 

## Virtual Environment Setup (optional)
(Only do the first step once, after you initially clone the repo)
1. Create a virtual environment using:
``python3 -m venv venv``
2. Activate your venv with:
``source venv/bin/activate``, or on Windows, ``venv\Scripts\activate``
3. Install the required packages:
   
   a. pip install numpy
   
   b. pip install gym
   
   c. pip install matplotlib
   
   d. pip install scipy
   
   e. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

   Replace the above index URL with whichever version of Pytorch corresponds to your drivers, you can check at [this website](https://pytorch.org/get-started/locally/)

## Running the Program

To run the gradient descent program, &lt;_Command to run program_&gt;
