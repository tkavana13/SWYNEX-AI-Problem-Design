# AI Problem Design – Customer Support Ticket Priority Classification

## Task 1: AI Problem Design

### Project Overview

This project defines and prototypes a practical AI problem for **automated customer support ticket priority classification**.

Customer support teams receive a large number of tickets every day. Manually identifying which tickets require immediate attention can be time-consuming and inconsistent.

The proposed AI system analyzes the **subject and description of a support ticket** and classifies it into one of three priority levels:

- **P1 – Critical:** Immediate attention is required.
- **P2 – High:** Important issue requiring timely attention.
- **P3 – Normal:** Routine question or low-impact request.

The system is designed as a **decision-support tool for support agents**, not as a replacement for human decision-making.

---

## Problem Statement

The objective is to develop a machine-learning model that can automatically classify incoming customer support tickets according to their priority.

### Input

The model receives:

- Ticket subject
- Ticket description

### Output

The model predicts one of:

```text
P1 – Critical
P2 – High
P3 – Normal
```

This is a **supervised multi-class text classification problem**.

---

## Target Users

### Primary User

**Customer Support / IT Help-Desk Agent**

The agent can use the predicted priority to quickly identify tickets that may require immediate attention.

### Secondary User

**Support Team Lead**

Team leads can use the system to improve consistency in ticket triage and monitor the overall workload.

---

## Data Source

For this prototype, a small **synthetic customer-support-ticket dataset** has been created.

The dataset is available at:

```text
data/support_tickets.csv
```

Each record contains:

| Column | Description |
|---|---|
| `ticket_id` | Unique ticket identifier |
| `subject` | Short title of the support ticket |
| `description` | Detailed description of the issue |
| `priority` | Target priority label |

### Production Data Source

In a real-world deployment, historical support tickets from an organization's authorized ticketing system could be used.

Before using real customer data, appropriate:

- Privacy controls
- Data protection
- Access controls
- Anonymization
- Organizational authorization

would be required.

---

## AI Approach

The proposed baseline uses a traditional machine-learning pipeline because the prototype dataset is small.

### Pipeline

```text
Support Ticket
      ↓
Subject + Description
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Priority Prediction
      ↓
P1 / P2 / P3 + Confidence
```

### Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression

---

## Why TF-IDF + Logistic Regression?

The dataset is relatively small, so a lightweight and interpretable machine-learning approach is appropriate for the initial prototype.

### Advantages

- Simple to implement
- Fast to train
- Works well for text classification
- Suitable for small datasets
- Easy to evaluate
- Provides a strong baseline before considering more complex models

If the baseline does not meet the required performance, future versions could evaluate transformer-based models.

---

## Project Structure

```text
ai-problem-design-task1/
│
├── README.md
├── SUBMISSION.md
├── requirements.txt
│
├── data/
│   └── support_tickets.csv
│
└── src/
    └── train.py
```

### File Description

| File | Purpose |
|---|---|
| `README.md` | Project documentation |
| `SUBMISSION.md` | Detailed task submission document |
| `requirements.txt` | Python dependencies |
| `data/support_tickets.csv` | Prototype dataset |
| `src/train.py` | Model training and evaluation code |

---

## Dataset Examples

### Example 1 – Critical

**Subject:**

```text
Production website is completely down
```

**Description:**

```text
Our production website is unavailable for all customers and the checkout page cannot be opened.
```

**Expected Priority:**

```text
P1
```

---

### Example 2 – High

**Subject:**

```text
Report generation is failing
```

**Description:**

```text
I cannot generate the monthly sales report and need it for today's meeting.
```

**Expected Priority:**

```text
P2
```

---

### Example 3 – Normal

**Subject:**

```text
Change notification settings
```

**Description:**

```text
How do I change which email notifications I receive?
```

**Expected Priority:**

```text
P3
```

---

## Model Evaluation

The model should be evaluated using a held-out test dataset.

### Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Per-class performance
- Confusion matrix

### Why Macro F1?

Macro F1 gives equal importance to each priority class.

This is important because a model should not appear successful simply by performing well on the most common class.

---

## Success Criteria

The proposed system has the following measurable success criteria:

| Metric | Target |
|---|---:|
| Macro F1 | ≥ 0.80 |
| P1 Recall | ≥ 0.90 |
| High-confidence prediction precision | ≥ 90% |
| Human review | Available for uncertain cases |

### Critical Requirement

**P1 recall is particularly important.**

A false negative for a critical ticket could delay the response to a serious service problem.

Therefore, the system should prioritize identifying critical tickets reliably.

---

## Evaluation Process

The evaluation process is:

1. Load the dataset.
2. Combine ticket subject and description.
3. Split the dataset using stratification.
4. Convert text into TF-IDF features.
5. Train the Logistic Regression model.
6. Predict priorities for unseen tickets.
7. Calculate evaluation metrics.
8. Generate a confusion matrix.
9. Analyze incorrectly classified tickets.
10. Review critical-ticket errors manually.

For a larger production dataset, a separate training, validation, and test set should be maintained.

For a very small dataset, stratified cross-validation can also be considered.

---

## Constraints

The prototype has several limitations:

### 1. Small Dataset

The prototype dataset is intentionally small and may not represent all possible customer-support situations.

### 2. Synthetic Data

The dataset is synthetic and therefore does not represent the language patterns of a particular organization's historical tickets.

### 3. Priority Definitions

Different organizations may define P1, P2, and P3 differently.

### 4. Ambiguous Tickets

Some tickets may contain insufficient information to determine their priority automatically.

### 5. Sensitive Information

Real support tickets may contain personal or confidential information. Such information must be protected before model training.

### 6. Model Drift

The language and types of support requests can change over time, so model performance should be monitored after deployment.

---

## Human Oversight

The AI system should assist support agents rather than make final decisions independently.

The support agent should be able to:

- Review the prediction
- Check the ticket information
- Override the predicted priority
- Escalate critical cases
- Send uncertain cases for manual review

Tickets with low model confidence should be routed to a human instead of being automatically prioritized.

---

## Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Critical ticket classified as normal | Monitor P1 recall and require human review |
| Incorrect historical labels | Audit and document labeling rules |
| Privacy issues | Remove/anonymize sensitive information |
| Model overconfidence | Use confidence thresholds and calibration |
| Model performance decreases | Periodically evaluate on recent tickets |
| Ambiguous ticket | Send to human support agent |

---

## Out of Scope

This prototype will **not**:

- Automatically close customer tickets
- Automatically respond to customers
- Make financial decisions
- Make security decisions
- Replace support employees
- Use customer data without authorization
- Automatically escalate every ticket without human review

---

## Example Model Workflow

```text
                Customer Support Ticket
                         |
                         ↓
              Subject + Description
                         |
                         ↓
                 TF-IDF Vectorizer
                         |
                         ↓
              Logistic Regression
                         |
              ┌──────────┼──────────┐
              ↓          ↓          ↓
             P1         P2         P3
          Critical     High      Normal
              |
              ↓
       Confidence Check
              |
       ┌──────┴───────┐
       ↓              ↓
   High confidence  Low confidence
       ↓              ↓
   Agent review    Human review
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ai-problem-design-task1.git
```

Move into the project directory:

```bash
cd ai-problem-design-task1
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Prototype

Run the training script:

```bash
python src/train.py
```

The program will:

1. Load the dataset.
2. Split the data.
3. Train the TF-IDF + Logistic Regression model.
4. Generate predictions.
5. Display the classification report.
6. Display the confusion matrix.
7. Demonstrate predictions on example tickets.

---

## Future Improvements

Future versions could improve the system by:

- Increasing the size of the training dataset
- Using real anonymized support tickets
- Applying cross-validation
- Hyperparameter tuning
- Calibrating prediction probabilities
- Comparing Logistic Regression with Linear SVM
- Evaluating transformer-based models
- Adding explainability features
- Monitoring model drift
- Creating a web interface for support agents
- Integrating with a ticket-management system

---

## Acceptance Criteria

The prototype can be considered successful if:

1. Macro F1 reaches or exceeds **0.80**.
2. P1 recall reaches or exceeds **0.90**.
3. Performance is reported separately for P1, P2, and P3.
4. The confusion matrix is reviewed.
5. Low-confidence predictions can be sent for human review.
6. The limitations of the dataset are clearly documented.
7. The system remains a decision-support tool with human oversight.

If these criteria are not achieved, the model should not be deployed directly. The dataset, labeling strategy, features, or model should first be improved.

---

## Conclusion

This project demonstrates how a broad customer-support problem can be converted into a **specific, measurable AI problem**.

Instead of attempting to automate the entire customer-support process, the project focuses on one narrow task:

> **Classifying customer support tickets into P1, P2, or P3 priority levels.**

The project defines the target users, data source, AI approach, constraints, evaluation methodology, success criteria, risks, and human-oversight requirements.

The resulting prototype provides a practical foundation that can later be expanded using larger datasets and more advanced natural-language-processing models.

---

## Author

**T Kavana**

Computer Science & Engineering (Data Science)

PES Institute of Technology and Management, Shivamogga

---

## Disclaimer

This is an educational/prototype AI problem-design project. The dataset is synthetic and the model should not be used for real production support operations without proper validation, security review, privacy controls, and human oversight.
