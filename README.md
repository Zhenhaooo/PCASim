## 📜 Project Overview

**PCASim** (**P**romptable **C**losed-loop **A**dversarial **Sim**ulation for Urban Traffic Environment) is a framework for generating and evaluating urban-traffic adversarial scenarios in a fully closed loop.

Key contributions

- **Multi-driver scenario corpus** Data-, knowledge-, and adversarial-driven tags are fused from the INTERACTION dataset, yielding a rich three-layer corpus.
- **RAG-enhanced DSL generation** Retrieval + Few-shot + Chain-of-Thought + self-consistency voting produce structured `geometry / spawn / behavior` DSL snippets.
- **DSL → Python compiler** A FAISS index of highway-env code fragments is retrieved into prompts; iterative syntax repair yields runnable scripts.
- **Two-phase PPO RL Validation** Adversary vehicles are trained first, then frozen; the ego agent is trained next and post-smoothed via Bézier convex optimisation.



## 🗂️ Directory Layout



```bash
PCASim
│
├── data/
│   └── raw/                      # Unprocessed datasets (e.g., INTERACTION)
├── corpus/                       # Processed scenario corpus with DSL + metadata
│   ├── brake/                    # Scenarios involving braking behaviors
│   ├── following/                # Scenarios involving car-following
│   └── …                         # Other categories: lane_change, turn_left, etc.
│
├── env/                          # Simulation environment and static resources
│   ├── NGSIM_env/                # Customized highway-env implementation
│   ├── simulation_environment/   # Additional modules (e.g., UI, sensors)
│   ├── index.faiss/              # FAISS index for code fragment retrieval
│   └── map/DR_USA_Intersection_MA.osm  # Example OSM map for simulation
│
├── dsl_generation/               # RAG + CoT + voting pipeline for DSL generation
├── code_compiler/                # DSL-to-Python compiler and syntax validator
├── rl_agent/                     # PPO training scripts and Bézier trajectory optimizer
├── scripts/                      # One-click pipelines and utility tools
├── highway_code_index/ 
│   ├── index.faiss
│   ├── index.pkl
│
├── docs/                         # Documentation, paper PDF, and interface specs
│   └── img/                      # Diagrams and framework illustrations
│
├── .gitignore                    # Ignore rules for caches, temp files, IDE folders
├── README.md                     # Main entry point and project description
└── requirements.txt              # Python dependency list for quick setup

```



## 🔧 Installation

```bash
conda create -n pcasim python=3.8
conda activate pcasim

# requirments
pip install -i requirments.txt

# Local highway-env fork
pip install -e env/

```



## 🚀 Quick Start

1、Build description

```
cd Corpus/Integrate_ADK
python assemble_description.py
```

2、Create Corpus

```
cd ..
python refine_and_snippet_export.py
```

3、Generate DSL

```
cd ../dsl_generation
python dsl_LLM_nostream.py
```

4、Compile Python

​	copy the final DSL to  `dsl_to_env_generator_test.py`, then

```
cd ../code_compiler
python dsl_to_env_generator_test.py
```

5、Train adversaries

```
cd ../rl_agent
python train_adv.py
```

6、Train ego agent

```
python train_ego.py
```

7、Evaluate

```
python test_model.py
python evaluation.py
```



## 📝 Module Details

### 1. Corpus (`corpus/`)

- Three-driver fusion (data, knowledge, adversarial).
- Output: `description.xlsx` plus aligned JSON-DSL.

### 2. DSL Generation (`dsl_generation/`)

- Retrieval-augmented prompt templates, CoT reasoning, semantic alignment filters.
- Self-consistency: cosine-centroid or field-wise cluster voting.

### 3. DSL → Python (`code_compiler/`)

- Prompt injects top-k code fragments from `env/index.faiss`.
- Three-round syntax-check/repair loop until `compile()` passes.

### 4. Reinforcement Learning (`rl_agent/`)

- **Phase 1**: train adversarial vehicles with PPO.
- **Phase 2**: freeze adversaries, train ego.
- Bézier convex optimiser smooths velocity & curvature.



## 📝 Reproduction Checklist

-  Download and unzip INTERACTION into `data/interaction/`.
-  Run **Step 1** to build corpus (Excel + JSON).
-  Build FAISS indexes for corpus and code.
-  Execute **Steps 2-4** to obtain runnable scenario scripts.
-  Finish **Steps 5-6** training.
-  Validate metrics with `eval_scenarios.py`.





## 📄 Citation

