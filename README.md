# STFC Hartree Centre: AI work experience

Welcome to STFC Hartree Centre! We hope you enjoy your week of work experience with the AI Group. You'll find a [Schedule](Schedule.pdf) for the week here too to give you an idea of what to expect. This repository also contains all the materials you'll need.


## 🎯 Learning Objectives

By the end of this week, you will be able to:

* Understand the core concepts behind neural networks (neurons, layers, activation functions, backpropagation).
* Preprocess and prepare data for a machine learning model using Pandas.
* Build a simple neural network from scratch and using Python libraries.
* Train and evaluate your model's performance.
* Gain practical experience using Git for version control in a project setting.

## 💡 What You'll Need

* A basic understanding of **Python**. You should be comfortable with its fundamental syntax, data structures (like lists and dictionaries), and how to write functions. We (and Google) are always here to help with anything you might get stuck with. It's also a great idea to talk to each other for help - that's how a real research environment works!

* Display screen equipment you're happy with - we will help you get set up on your laptops.

## 🚀 Getting Started: Initial Setup

### Step 1: Create a GitHub Account
If you don't already have one, you'll need to create a free GitHub account. This is an essential tool for software developers.

1.  Go to [https://github.com/join](https://github.com/join).
2.  Follow the instructions to sign up.

### Step 2: Fork This Repository
"Forking" creates a personal copy of this entire project in your own GitHub account.

1.  Make sure you are logged into your GitHub account.
2.  Click the **"Fork"** button in the top-right corner of this page.
3.  GitHub will create a copy of the repository under your username. You'll be automatically taken there.

### Step 3: Clone Your Forked Repository
Now, you need to get the files onto your computer. This is called "cloning".

1.  On your forked repository page on GitHub, click the green **"<> Code"** button.
2.  Make sure the "HTTPS" tab is selected, and copy the URL.
3.  Open a terminal or command prompt on your computer and run the following command (paste the URL you just copied):
    ```bash
    git clone [URL_YOU_COPIED_FROM_YOUR_FORK]
    ```
4.  This will create a folder on your computer with all the project files.

**Remember to commit your progress:** Use Git to commit your changes regularly. This is great practice and helps you track your work.

     ```bash
    git add .
    git commit -m "Completed Day 1 - data preprocessing"
    ```

Here's a quick video to explain git in a little more detail - [https://www.youtube.com/watch?v=e9lnsKot_SQ](https://www.youtube.com/watch?v=e9lnsKot_SQ).

## 💻 Setting Up Your Development Environment

Next, let's install the tools you will need for coding. We will use Visual Studio Code as our editor and set it up for Python and Jupyter Notebooks.

### Part 1: Install Visual Studio Code (VS Code)
VS Code is a modern, powerful, and free code editor.

1.  Download VS Code from the official website: [https://code.visualstudio.com/](https://code.visualstudio.com/).
2.  Run the installer and follow the on-screen instructions. Accepting the default settings is fine.

### Part 2: Install Python
You need Python installed on your computer to run the code.

1.  Download the latest version of Python from [https://www.python.org/downloads/](https://www.python.org/downloads/).
2.  Run the installer.
3.  **Important:** On the first screen of the installer, make sure to check the box that says **"Add Python to PATH"** or **"Add python.exe to PATH"**. This will make it much easier to run Python from the command line.
4.  Continue with the default installation.

### Part 3: Set Up VS Code for Python & Jupyter
Now we configure VS Code to work with Python files and Jupyter Notebooks.

1.  **Open VS Code.**
2.  **Open your cloned project folder:** In VS Code, go to `File > Open Folder...` and select the folder you created when you cloned the repository.
3.  **Install Extensions:** Click on the Extensions icon in the Activity Bar on the side of the window (it looks like four squares). Search for and install the following two extensions from Microsoft:
    * `Python` (ms-python.python)
    * `Jupyter` (ms-toolsai.jupyter)
4.  **Install Jupyter package:** Open a new terminal directly within VS Code (`Terminal > New Terminal`). In the terminal window that appears at the bottom, run the following command to install the necessary package for running notebooks:
    ```bash
    pip install notebook
    ```

5.  **Select the Python Kernel:** The first time you open a Jupyter Notebook (`.ipynb` file), you need to tell VS Code which Python installation to use. This is called "selecting a kernel".
    * Open a notebook file from the project (e.g., in the `Day1_DataPrep` folder).
    * Look in the top-right corner of the screen. If you see a button that says **"Select Kernel"**, click it.
    * A dropdown list will appear. Choose the Python environment you installed in Part 2. It should be listed as something like `Python 3.x.x`.
    * Once selected, you are ready to run the code cells in the notebook by clicking the 'play' button next to each cell. VS Code will remember your choice for this project.

6.  You are now ready to code! You can create and run `.py` files and `.ipynb` (Jupyter Notebook) files.

---




## 🗓️ Workflow


* **Task 1: Pandas**

Navigate to the `Task 1: pandas` folder. This notebook will get you familiar with the pandas library, and get the python brain whirring. 

* **Task 2: Neural Network**

Navigate to the `Task 2: Neural network` folder. The [pdf file](Task 2: Neural network/work_exp_proj_ai/work_exp_proj_ai.pdf) file will guide you through the main segment of your work experience. 

Good luck!
