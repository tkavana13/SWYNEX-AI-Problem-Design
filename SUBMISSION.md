# Task 1 Submission — AI Problem Design

**Candidate:** T Kavana  
**Task:** AI Problem Design  
**Use Case:** Customer Support Ticket Priority Classification

> This document defines a practical, narrow AI use case, its users, data source, constraints, AI approach, evaluation methodology, and measurable success criteria.

## Executive Summary

The proposed AI system classifies incoming customer support tickets into three priority levels: P1 (Critical), P2 (High), and P3 (Normal). The purpose is to help support agents triage tickets consistently and quickly.

The prototype uses a small synthetic dataset and a TF-IDF + Logistic Regression baseline. The primary success criterion is Macro F1 ≥ 0.80, with an additional safety-focused requirement of P1 recall ≥ 0.90.

The system is a decision-support tool. Critical and low-confidence cases remain subject to human review.

## 1. User

The primary user is a customer support or IT help-desk agent. Support leads are secondary users who can use the predictions to standardize triage.

## 2. Data Source

The prototype uses a small synthetic dataset stored in `data/support_tickets.csv`. Each row contains a ticket ID, subject, description, and ground-truth priority.

For production, historical tickets from an authorized ticketing system could be used, subject to privacy, security, and data-governance requirements.

## 3. AI Problem

**Input:** Ticket subject + description.

**Output:** One of:
- P1 — Critical
- P2 — High
- P3 — Normal

This is a supervised multi-class text-classification problem.

## 4. AI Approach

The baseline pipeline is:

`Ticket text → TF-IDF features → Logistic Regression → Priority + confidence`

TF-IDF is suitable for a small dataset and gives a simple, explainable baseline. If performance is insufficient, a transformer-based classifier can be evaluated as a second iteration.

## 5. Constraints

- Small prototype dataset.
- Synthetic data does not represent every real organization.
- Priority labels depend on company policy.
- Ambiguous tickets require human review.
- Sensitive information must be protected.
- The model must not make autonomous security or business-critical decisions.
- Model performance should be monitored for drift after deployment.

## 6. Evaluation

Use a stratified train/validation/test methodology where possible. For the small prototype, cross-validation can be used to reduce dependence on one split.

Report:
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision/recall/F1
- Confusion matrix

### Success criteria

| Criterion | Target |
|---|---:|
| Macro F1 | ≥ 0.80 |
| P1 recall | ≥ 0.90 |
| High-confidence prediction precision | ≥ 90% for confidence ≥ 0.80 |
| Human override | Always available for ambiguous cases |

P1 recall is emphasized because missing a critical incident is more costly than incorrectly escalating a non-critical ticket.

## 7. Example

**Input:** "Production database is unavailable for all users."

**Expected output:** P1 — Critical.

**Input:** "How do I change my email notification settings?"

**Expected output:** P3 — Normal.

## 8. Risks and Mitigation

- **False negative P1:** prioritize P1 recall and require human review for uncertain cases.
- **Historical label bias:** audit and document labeling rules.
- **Data leakage/privacy:** remove or protect personal information.
- **Model drift:** evaluate periodically on recent tickets.
- **Overconfidence:** calibrate probabilities and define an uncertainty threshold.

## 9. Acceptance Decision

The model should be considered ready for a limited prototype only if it meets the stated metric targets and passes manual review. If it fails, the next step is to improve the data, labeling policy, features, or model rather than deploy it blindly.

## 10. Files

- `README.md` — complete problem design
- `data/support_tickets.csv` — small prototype dataset
- `src/train.py` — baseline implementation
- `requirements.txt` — Python dependencies
