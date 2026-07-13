# Product Plan: Agentic Chemometrician Framework

## Product Vision
Transform the existing collection of exploratory chemometrics notebooks (00–08) and early agent experiments (09, 10) into a cohesive, **Agentic Chemometrics Framework**. 

The goal is not to replace the analytical chemist, but to automate the repetitive "first-pass" workload—data loading, preprocessing comparison, model benchmarking, and baseline validation—using an LLM agent that orchestrates robust, deterministic Python tools via the Model Context Protocol (MCP).

## Core Value Proposition
- **Expert Effort Reduction:** Automates routine spectral method development so chemists can focus on complex edge cases and scientific interpretation.
- **Scientific Rigor over Hallucination:** Computations are handled by deterministic Python tools, not the LLM. The agent's job is purely orchestration, planning, and interpretation.
- **Human-in-the-Loop Architecture:** Designed with explicit approval gates. The agent handles the heavy lifting but stops for human approval before finalizing scientific conclusions.

## Target User Experience
1. **Ingestion:** The user provides a spectral dataset (e.g., NIR flooring data). The agent uses an `inspect_dataset` tool to load, parse, and identify data shapes and potential target labels.
2. **Analysis Proposal:** The agent proposes a bounded analysis plan (e.g., trying Savitzky-Golay preprocessing, baseline SVM/XGBoost models, and PCA clustering) and requests the user's approval.
3. **Automated Execution:** Upon approval, the agent executes the tasks, using deterministic tools to train models, generate cross-validation metrics, and select important features.
4. **Validation & Fallback:** If a model fails to converge or exhibits data leakage, the agent diagnoses the failure and suggests a fallback model.
5. **Reviewable Reporting:** The agent generates a comprehensive report containing figures, metrics, validation warnings (e.g., class imbalances, leakage risks), and a human-review checklist.

## Key Product Features
- **Chemometrics MCP Server:** A central hub of chemometrics tools (preprocessing, classification, regression, interpretation, and validation).
- **Agent-Neutral Prompt Library:** A collection of pre-defined workflows, skills, and scientific guardrails that work with any MCP-capable client (Claude, Codex, etc.).
- **Validation Engine:** Built-in checks for replicate leakage, group leakage, split instability, and suspiciously high metrics.
- **Standardized Data Contracts:** Unified `SpectralDataset` and `AnalysisResult` structures to ensure seamless data handoffs between tools.
- **Method Memory (Future-facing):** A store for previously human-approved analyses to bootstrap future dataset explorations.

## High-Level Rollout Strategy

### Phase 1: Repository Consolidation
Transition the repository from a numbered-folder exploration structure into a unified software package. The core logic from notebooks 00–08 will be refactored into a `chemometrics_core` Python library.

### Phase 2: MCP Server & Agent Foundation
Construct the MCP server skeleton and establish the agent-neutral prompt library. Implement the first end-to-end path using the simplest tools (Data Inspection, Preprocessing, and Binary Classification).

### Phase 3: The "Paper-Ready" MVP
Expand the toolset to include XGBoost, SVR wear-layer regression, clustering, feature selection, and the validation engine. Run the agent against the initial NIR flooring dataset from start to finish to generate a human-reviewable report and agent trace artifacts. 

### Phase 4: Generalization
Prove the framework's robustness by ingesting a new spectral modality (e.g., FTIR data) with minimal prompt changes, validating that the pipeline is broadly applicable across chemometric domains.
