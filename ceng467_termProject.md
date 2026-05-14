# CENG 467 Natural Language Understanding and

# Generation

## Term Project Topics

## ̇Izmir Institute of Technology ( ̇IYTE)

## Prof. Dr. Aytu ̆g ONAN

## Spring 2026

## Student Project Record

- Student Name: Zübeyr Almaho
- Student ID: 300201023
- Selected Topic: Controllable Text Generation via Diffusion Models

## Short Project Focus

This project will study controllable text generation with diffusion-based language models and compare
their behavior against autoregressive baselines. The main focus will be measuring fluency, lexical
diversity, and controllability under different generation settings, with Diffusion-LM as the primary
method and standard Transformer-based generation models as comparison baselines.

## Project Philosophy

The goal of the term project is to bridge the gap between undergraduate-level NLP practice and
graduate-level research in Natural Language Understanding and Generation.
Students are expected not only to train models but also to analyze and evaluate their behavior using
rigorous experimental methodology. The projects emphasize modern Transformer architectures, large
language models (LLMs), evaluation methodologies, and responsible AI practices.
Each project should demonstrate:

- Understanding of modern NLP architectures
- Careful experimental design
- Quantitative evaluation using established NLP metrics
- Analysis of model limitations and potential biases

## Project Requirements

Students will work in groups of up to three members. Each group must:

- Implement a complete NLP pipeline using modern deep learning methods
- Compare at least two baseline approaches
- Implement the proposed architecture or technique
- Conduct systematic experimental evaluation
- Provide fully reproducible code through a GitHub repository
- Submit a 6–8 page academic-style report
- Present a short live demonstration of their system

## Project Selection and Assignment Policy

Each group is expected to select a project topic from the list provided in this document. In addition,
groups may propose an original project topic, provided that it is relevant to the scope of the course
and receives prior approval from the instructor.
Proposed projects must:

- Be clearly within the scope of Natural Language Processing / Natural Language Understanding
    and Generation
- Demonstrate sufficient technical depth for implementation and evaluation


- Include appropriate baseline comparisons
- Be feasible within the semester timeline

To ensure diversity of work across the class, no two groups may work on the same project
topic. Each topic (listed or proposed) will be assigned to only one group.
Project topic assignments will be managed via a shared spreadsheet (Excel/Google Sheets), the link
to which will be announced.
Each group is required to:

- Register group members and selected project topic in the shared sheet
- Ensure that the topic has not already been assigned

Topics will be allocated on a first-come, first-registered basis. The shared sheet will serve as the
official record of project assignments.

Students who fail to register their group or project topic in a timely manner may be randomly assigned
into groups and allocated a project topic at the instructor’s discretion.

Approval Requirement for Proposed Topics:
All student-proposed project topics must receive explicit approval from the instructor before they
can be registered in the project selection sheet.
Groups proposing a custom topic are required to submit a short proposal document (see template
below). Approval will be granted based on:

- Relevance to the course content and learning objectives
- Technical depth and feasibility
- Availability of suitable datasets or evaluation benchmarks
- Clarity of evaluation methodology and baseline comparisons

```
Projects that are not approved may be revised or replaced with a topic from the provided list.
```
## Required Experimental Components

Each project must include the following elements:

- Baseline Comparison Students must implement at least two baseline models or prompting strate-
    gies to provide a meaningful comparison with the proposed approach.
- Evaluation Metrics Appropriate NLP metrics must be used depending on the task (e.g., BLEU,
    ROUGE, METEOR, BERTScore, Exact Match, F1-score, or perplexity).
- Ablation or Sensitivity Study Students should analyze how architectural components, prompts,
    or hyperparameters affect model performance.
- Error Analysis Qualitative analysis of failure cases must be provided, discussing common mis-
    takes, hallucinations, or linguistic errors.
- Ethical Considerations Projects involving generative models should include a brief discussion
    of potential bias, hallucination risks, or misuse scenarios.


## Reproducibility

All projects must be fully reproducible.
Each GitHub repository must include:

- Clear project structure
- Training and evaluation scripts
- Dataset preparation instructions
- Dependency file (requirements.txt or environment.yml)
- README explaining how results can be reproduced

```
If LLM APIs are used, prompts and configuration settings must also be documented.
```
## Academic Report Structure

Reports must follow a standard scientific structure:

1. Introduction
2. Related Work
3. Methodology
4. Experimental Setup
5. Results
6. Error Analysis
7. Discussion
8. Conclusion

```
Proper citation of datasets, libraries, and research papers is mandatory.
```
## Project Proposal Template (for Custom Topics)

Groups proposing an original project topic must submit a one-page proposal including the following
components:

- Project Title
    Clear and concise title of the proposed project.
- Problem Description
    A brief explanation of the problem being addressed and its importance.
- Proposed Approach
    Description of the planned NLP approach, model architecture, prompting strategy, or pipeline.
- Baseline Methods
    At least two baseline models, prompting strategies, or evaluation baselines that will be used for
    comparison.
- Dataset / Benchmark
    Description of the dataset(s), benchmark(s), or corpus/corpora to be used, including source and
    scale if available.


- Evaluation Strategy
    Metrics and evaluation methodology (e.g., BLEU, ROUGE, F1-score, Exact Match, BERTScore,
    perplexity, etc.).
- Expected Challenges
    Potential difficulties, limitations, or risks anticipated in the project.

```
Proposals should be concise, clearly structured, and limited to one page.
```
## Optional Graduate-Level Extension

Students who wish to explore the topic further may optionally include:

- Implementation of an additional advanced architecture
- Reproduction of results from a recent NLP research paper
- Proposal of a small methodological improvement
- Additional cross-dataset evaluation

```
These extensions are optional but may positively influence evaluation.
```
```
Project Name Project Description Architectures /
Methods
```
```
Dataset
```
1. Instruction
Tuning with
Parameter-Efficient
Fine-Tuning
(PEFT)

```
Fine-tuning massive LLMs from scratch is
computationally expensive. Students will
implement QLoRA to instruction-tune an
open-weight language model for a specific
domain, analyzing memory efficiency and
generation quality.
```
```
QLoRA, Instruction
Tuning, Causal LMs
```
```
Alpaca /
PubMedQA
```
2. Alignment via
Direct Preference
Optimization (DPO)

```
Students will explore alignment techniques
for LLMs by implementing Direct Preference
Optimization on a human preference dataset
and analyzing how alignment affects
generation behavior.
```
```
DPO, Transformer
Decoders
```
```
Anthropic
HH-RLHF
```
3. Advanced RAG
with Hard Negative
Mining

```
Students will build a retrieval-augmented
generation pipeline and train a dense
retriever using hard negative mining to
improve question answering robustness.
```
```
DPR, Vector
Databases, LLMs
```
```
HotpotQA /
MS MARCO
```
4. Controllable Text
Generation via
Diffusion Models

```
Students will implement a diffusion-based
text generation approach and compare its
lexical diversity and fluency against
autoregressive generation methods.
```
```
Diffusion-LM,
Non-autoregressive
Generation
```
### E2E NLG /

```
Common-
Gen
```
5. Cross-lingual
Abstractive
Summarization

```
Students will design a Transformer-based
sequence-to-sequence system that
summarizes documents in one language while
generating summaries in another.
```
```
mT5 / mBART,
Cross-Attention
```
```
XSum /
WikiLingua
```
```
Continued on next page
```

```
Project Name Project Description Architectures /
Methods
```
```
Dataset
```
6. In-Context
Learning:
Chain-of-Thought vs
Tree-of-Thoughts

```
Students will experimentally compare
reasoning prompting techniques to analyze
their effectiveness in complex logical
reasoning tasks.
```
```
CoT Prompting,
ToT, LLM APIs
```
### GSM8K /

```
StrategyQA
```
7. Mechanistic
Interpretability in
Transformers

```
Students will analyze attention head
behavior and perform causal tracing
experiments to understand how factual
knowledge emerges in Transformers.
```
```
Causal Tracing,
Attention Analysis
```
```
CounterFact
/ SQuAD
```
8. Tokenization
Impact on
Morphologically
Rich Languages

```
Students will evaluate different subword
tokenization strategies when training small
autoregressive language models on Turkish
text.
```
```
BPE, WordPiece,
Unigram
```
### OSCAR

```
(Turkish)
```
9. Unsupervised
Neural Machine
Translation

```
Students will implement an unsupervised
machine translation system using
back-translation and denoising objectives to
learn translation without parallel data.
```
```
Back-translation,
Denoising
Autoencoders
```
### WMT /

### FLORES-

### 200

10. LLM-as-a-Judge
for Evaluating
Generation

```
Students will design an evaluation pipeline
where a strong LLM assesses the quality of
machine translation or summarization
outputs.
```
```
Prompt Engineering,
LLM Evaluation
```
### WMT

```
Metrics
Shared Task
```
11. Mitigating
Hallucinations via
Epistemic
Uncertainty

```
Students will design a generation pipeline
that estimates uncertainty using
self-consistency and entropy-based methods.
```
```
Self-Consistency,
Entropy Measures
```
```
TruthfulQA
/ HaluEval
```
12. Text-to-SQL
Semantic Parsing

```
Students will fine-tune an instruction-based
language model to convert natural language
queries into executable SQL statements.
```
```
Instruction Tuning,
Schema Encoding
```
```
Spider /
GeoQuery
```
13. Knowledge
Distillation for
Task-Specific NLU

```
Students will distill knowledge from a large
teacher model into a lightweight student
model optimized for efficient inference.
```
```
Knowledge
Distillation, BERT
```
### GLUE

```
Benchmark
```
14. Retrieval-
Augmented Machine
Translation

```
Students will implement kNN-MT to enhance
neural machine translation with
retrieval-based memory at inference time.
```
```
kNN-MT, Faiss IWSLT
```
15. NLP Ethics:
Quantifying and
Mitigating
Generation Bias

```
Students will measure and mitigate bias in
generative models using established bias
metrics and debiasing techniques.
```
```
Bias Metrics,
Debiasing Methods
```
### BOLD /

```
RealToxici-
tyPrompts
```
## Project Evaluation Rubric

Projects will be evaluated according to the following criteria:


```
Criterion Points Description
Problem Formulation and Task Under-
standing
```
```
10 Clear definition of the NLP task, moti-
vation, and explanation of dataset char-
acteristics.
Baseline or Prompting Comparison 15 Implementation of baseline models,
prompting strategies, or evaluation
baselines.
Model / Method Implementation 20 Correct implementation of the pro-
posed NLP architecture or prompting
strategy.
Evaluation Methodology 15 Appropriate use of evaluation metrics
(BLEU, ROUGE, F1, perplexity, etc.)
and proper experimental setup.
Ablation or Prompt Sensitivity Analy-
sis
```
```
10 Analysis of model behavior under dif-
ferent prompts, hyperparameters, or ar-
chitectural choices.
Generation Quality Analysis 10 Qualitative evaluation of generated out-
puts including coherence, factual accu-
racy, and linguistic quality.
Ethical Considerations and Bias Anal-
ysis
```
```
5 Discussion of hallucinations, bias, or
misuse risks for generative models.
Reproducibility and Code Quality 5 Well-structured GitHub repository
with instructions to reproduce results.
Presentation and Demonstration 10 Clarity and quality of the presentation
and demonstration of the system.
Total 100
```
Bonus (+5 points): Students may receive bonus points for implementing an additional advanced
model, reproducing results from a recent research paper, or conducting additional experiments.

## Project Timeline

```
Week Milestone Deliverable
Week 5 Project Assignment Released –
Week 7 Group Formation + Topic Selection + Proposal
Submission
```
### *

```
Week 8–10 Literature Review and Initial Development –
Week 11 Technical Checkpoint (working baselines + ini-
tial results)
```
### *

```
Week 12–14 Method Development, Experiments, and Paper Writing –
Week 15 Final Submission (LNCS paper + system) *
Week 15–16 Project Presentations and Demo –
```
Important: Progress will be monitored via GitHub commits. Lack of consistent development may
negatively affect evaluation.


