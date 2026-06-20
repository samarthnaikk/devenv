# Objective

We want to make a coding agent (similar to codex) but with the following features
- infinite contexts. along with the current session the agent also remembers what has happened before.
- runs on local/other API keys, like groq 
- has custom made tools

# What has been completed

- all the tools
- basic ui
- ai, runtime, memory coding
- a complete chain of actions

# The new logic

- Currently everything is good at its place but its not "integrated" so we introduce the "three step process"
- this has 3 distinct steps. 1. context/memory 2. brain 3. metadata
- these are 3 steps for every prompt. have their own tasks
- before this we do checkpoint creation.

# Technical constraints

- Unlike OpenAI we dont have very high token per minute or very high single contexts


# checkpoint creation

- what is a checkpoint? a checkpoint is a task that can be done in a single shot in our given context. say explaining about backend or such basic things
- what if it requires multiple file reads or different analysis, many file writes, you break them into check points
- the first very task is to break them into how many ever number of checkpoints (it can be 10, 20, 100 or how many ever)
- you create them and store them sequentially. Now start executing them one by one using the "three step process"
- say while doing a checkpoint it feels that this cannot be done in a single shot, there itself break into checkpoints and solve
- checkpoint creation does not need too much context, it just breaking the objective/target into small target and doing each of them.
- the creation of checkpoint itself is not bounded to follow the "three step process"

# checkpoint verification

- a checkpoint is complete only if the objective of that checkpoint was done
- and a small verification tool runs, say if its a function does it work, is it complete, if its a html file is able to render
- like a verification of is the checkpoint objective completed, if not there itself create a new checkpoint whose object is just to fix that issue and achieve the target
- once completed only then move to next objective.

# 1. context/memory

- this part deals with the "what?" . What exactly do you want to build, what data do I need to know, do I have that information or do I need to fetch it. If not available I need it.
- since we have technically infinite memory, what that means is the model has a very huge library and it has a small book say of 6-8 pages. At a time it can only store that much. 
- We have tools (if we dont have, we can create) that get semantic search results etc. So out of those 6-8 pages, 4 pages is alloted for the what. 
- what is the "current" context. Say the user is working on a project, and says do you remember what issue was coming with that project. So now you need to change the context. It needs to change those 4 pages, look up in the huge library using tools gather info. Now your "current" context is completely different. At each convo you write down that and store in library, when you need you would extract.
- Say the users asks to create a new function, and since its new it does not exist, so the context is only that this was the goal, i need the content to this function.
- this whole process is executed with a tiny local model + tools existing

# 2. brain

- this where we get ai into picture. Now this brain knows what it needs to do cause we have context/memory.
- it knows if it should give function code, or take the data and summarize or whatever

# 3. metadata

- the smallest aspect.
- it will at the start note down what the objective was, and where is the target.
- if it was a file write, location of file write, if it was suppose to be text for chat in user or anything. this will help us keep our "pages" of book to be used for actual required things

At every point we break the checkpoint, how much ever big or small to these 3 points and solve it.