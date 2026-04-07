Final Project Proposal: Autonomous ML Training Agent
Version History
Date: April 6, 2026
Version: 0.1 (First Draft for TA Review)
Related Documents/Links
CS Final Project Guidelines (Project Genesis)
Tinker API Docs (Still trying to find the updated ones...)
Hugging Face Hub API Docs
Karpathy’s autoresearch repo (Our main inspiration)
Overview
The Autonomous ML Training Agent is an end-to-end script/agent that takes a plain-English prompt (like, "build a model that classifies handwritten digits") and a strict student budget (e.g., $50), and handles the rest. It finds or makes the data, picks the right model architecture, runs the training on Tinker's distributed GPUs and, most importantly, kills the run before it burns through our monthly budget.
The Problem / Why We're Doing This

What are the user pain points you are trying to solve or new functionalities you are trying to innovate?
Here are the main headaches we've noticed (and experienced):
Data wrangling sucks. Even if you know the exact PyTorch architecture you want, finding, cleaning, and formatting the data takes up 80% of the project time. Our agent automates this by scraping, pulling from Hugging Face, or using a Teacher LLM to fake it. 
Architecture paralysis. Should we fine-tune? Train from scratch? Use LoRA? Grab an 8B model or push for 235B? It’s overwhelming. The agent looks at the task and our wallet, and makes the call for us.
Cloud costs are terrifying. Leaving a GPU on accidentally is a classic student mistake. "I'll just let it run for an hour" turns into a $400 charge. Our agent acts as a financial bodyguard with a hard kill-switch tied directly to the API.
Hyperparameter tuning is tedious. Sitting there tweaking learning rates is painful. The AutoResearch loop automates the cycle of hypothesis → code edit → test → commit/revert so we don't have to stare at the terminal.
The Opportunity

Identify some opportunities for your product.

What are the opportunities to have an impact in your project about which you are most passionate? What are realistic targets for those opportunities that would be fulfilling if you achieved them by June?  How niche or how big is the Total Addressable Market (TAM)?
Where we think this will actually get used by June:
Labs across campus: Bio or Econ researchers who need ML models for their papers but don't know how to set up the infrastructure.
Quick projects/experimentation: When you have 24 hours and just need a custom classifier by 8 AM Sunday.
Realistic June Goal: Have a working agent to demo that handles 3 types of tasks (basic classification, fine-tuning an LLM, custom pre-training), runs entirely on Tinker without us touching it, and spits out a saved model weights file.
Ambitious Goal: Improving infrastructure and agent capabilities to allow this agent to be used in enterprise settings 

SOM (Serviceable Obtainable Market): Our initial launch targets academic researchers, grad students, and hackathon participants. These users have specific ML needs (e.g., custom classifiers for thesis data), strict deadlines, and limited budgets, but lack the time or expertise to configure cloud ML infrastructure.
SAM (Serviceable Available Market): As the agent's AutoResearch loop stabilizes, the market expands to bootstrapped startups, indie hackers, and small software agencies. This represents roughly 5 to 10 million developers globally who want to train custom AI models but cannot afford to hire a dedicated $150k+/year MLOps engineer.
TAM (Total Addressable Market): The ultimate vision targets the global software engineering workforce. Currently, there are roughly 30 million software developers worldwide, but fewer than 2 million are specialized Data Scientists or ML Engineers. Our agent bridges this talent gap, giving the remaining 26+ million standard developers the autonomous infrastructure needed to train and deploy custom models without needing a specialized team.


User segments
Primary Segments 
Student ML Builders 
Non-ML Academic researchers 
Lightweight prototyping in academic research/enterprise contexts 
Secondary Segments: 
bootstrapped founders/small startups (those who need custom models for a product but cannot afford ML talent/oversight of GPU clusters 
General purpose users or AI experimenters
Value Prop / Differentiators
An AI “co‑pilot” that turns a plain‑English idea and a hard budget cap into a deployed model by automatically searching, curating, and/or generating data, choosing the right training strategy (LoRA, fine‑tune, or pre‑train), running an AutoResearch loop, and enforcing strict cost guardrails so students and researchers ship useful models without touching infra or blowing up their cloud bill.

Functional Requirements
The agent receives a training prompt and budget: 

Feature 0: The Manager: This is the central agent that controls the entire process. 
	
Feature 1: The Data Generator:
There are three possible cases here: 1. user passes in data but its not structured nor in a trainable format. 2. Users doesnt have data just passses in prompt but sufficient data is on hugging face. 3. Data is not on hugging face, model will need to intelligently scrape and create the dataset (hardest option) 
The data generator is the hardest / most technical part of this product. The data generator will first search huggingface for relevant datasets. If the data required is not present, it either synthetically generates a dataset, or activates an intelligent scraper that creates it.  Or a user can pass in a messy dataset and we will clean it up? Agent will decide on the quality of the dataset


Feature 2: The Decision Engine 
The Issue: Deciding whether to fine-tune or pre-train is hard.
The Fix: The agent looks at the prompt and budget
Case A (Fine-Tune): Grabs a base model from HF, calls Tinker to run a LoRA script.
Case B (Pre-Train): Writes a custom model.py and train.py from scratch, then hands it over to the AutoResearch loop. 
Feature 3: AutoResearch Agentic Loop
This is the phase where the agent improves upon the model it has created. It does the hyperparameter tuning + architectural tuning, the way a machine learning engineer would. The agent forms a hypothesis ("More attention heads = better"), tweaks the code, runs a 5-minute test on Tinker, and checks the validation loss. It keeps the changes that work and trashes the ones that don't. This would be built on top of the same structure as Andrej Karpathy’s autoresearch. 
Some visualization feature to see lo
Feature 4: The Cost Manager
Constantly polls the Tinker billing API. If we go beyond the budget, it saves the state_dict and nukes the instance. 
Feature 5: Observability 
Observability into the entire model training process

—--------------------------- After Draft (everything after this does not need to be completed now) 
