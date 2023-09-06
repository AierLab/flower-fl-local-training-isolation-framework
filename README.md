# 🌸 Flower-FL-Local-Training-Isolation-Framework 🌸

Welcome to the AIerLab's groundbreaking Federated Learning repository! We are a group of geeks committed to making AI safe and friendly for everyone. Our setup integrates complex training methodologies seamlessly, paving the way for the next generation of machine learning solutions without impinging on the native Flower framework logic.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)
![Flower Framework](https://img.shields.io/badge/framework-Flower-brightgreen.svg)
[![Join our Discord AIerLab](https://img.shields.io/badge/discord-join%20chat-blue.svg)](https://discord.gg/pjhf5TqH)

## 💻 Setting Up Your Development Environment

Before you dive into our repository, ensure your system is equipped with the necessary frameworks to facilitate a seamless experience:

### 1. 🌺 Flower Framework Installation

Set up the Flower framework as the foundational element for federated learning by executing the following command in your terminal:

```shell
python -m pip install flwr 
```

### 2. 🕯️ PyTorch Installation

Next, install PyTorch, the powerful library facilitating machine learning and deep learning implementations. Follow the official guidelines for a smooth installation:

```shell
python -m pip install torch torchvision
```

## 🚀 Running the Simulation

Our innovative approach segregates the model training process from the Flower framework, allowing the seamless integration of complex training methodologies without disruptions. Here's your guide to executing the simulation:

### 1. 🖥️ Server Startup

Navigate to the project directory and initialize the server with this command:

```shell
cd path/to/the/folder
python server_main.py
```

### 2. 📱 Client Main Activation

In separate terminals, initiate the client main instances sequentially, assigning them respective ranks:

```shell
cd path/to/the/folder
python client_main.py --rank 1
```
```shell
cd path/to/the/folder
python client_main.py --rank 2
```

### 3. 🏭 Model Main Activation

Simultaneously, launch the model training threads in distinct terminals, assigning them the respective ranks:

```shell
cd path/to/the/folder
python model_main.py --rank 1
```
```shell
cd path/to/the/folder
python model_main.py --rank 2
```

This establishes a standard simulation setup orchestrating five concurrent terminals: two hosting `client_main.py`, two accommodating `model_main.py`, and one dedicated to `server_main.py`.

> 💡 Explore the wealth of configuration options with the `-h` or `--help` command-line arguments.

## 🌈 Features and Advancements

Dive into our enriched repository that offers:

- 🎓 **Detached Training Process:** Our system separates the training process from the Flower framework, paving the way for intricate training schemes to function harmoniously without interfering with the core logic.

- 📦 **Localized State Dictionary Communication:** We employ local state dictionaries as a mode of internal communication, fostering a ground where sophisticated training techniques can be employed with ease and efficiency.

- 💾 **Checkpoint Management:** Our setup facilitates the efficient management of checkpoints, both server-side and client-side, instilling a resilient learning protocol.

- 📈 **Adaptable Aggregation Functionality:** Customize the aggregation functions to cater to your split learning requirements, facilitating an optimized learning workflow crafted for specific project needs.

We're continually enhancing our repository, so stay tuned for new, dynamic features!

---

Join us in this endeavor as we aim to redefine federated learning protocols, fostering an environment where complexity meets simplicity, paving the path for the next generation of machine learning solutions. Become a part of this revolutionary journey and connect with fellow enthusiasts in our [AIerLab Discord community](https://discord.gg/pjhf5TqH). Thank you for being a part of this transformative journey!
