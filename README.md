# CS475 Team C: Creating FPS AI with Deep Learning
This project contains everything needed to train an AI to shoot targets in the unreal engine. It contains an unreal engine project that provides a level and an interface to train the agent. The unreal engine will communicate with a flask server that hosts the agent.

## Installation
1. Install Unreal Engine 4.26
2. Install the free VaREST plugin from the Unreal marketplace
3. Clone the repository
4. Install the requirments.txt located in the ml_server folder

## Usage
First launch the Flask server, if it is not launched then the UE4 project will not work. Once the Flask server is up, launch the Unreal Engine and load the project. Then hit play to begin training the Agent!