# Email Multi-Label Classification (Design Choice 1)

This project implements **Chained Multi-Output Architecture** for email classification.

The system classifies emails using three dependent variables:

- Type2
- Type3
- Type4

To respect label dependency, the system creates chained targets:

| Target | Meaning |
|------|------|
| t2 | Type2 |
| t23 | Type2 + Type3 |
| t234 | Type2 + Type3 + Type4 |

Example:

Type2 = Suggestion  
Type3 = Refund  
Type4 = Subscription cancellation  

t234 = Suggestion||Refund||Subscription cancellation

Architecture:

Controller → Preprocessing → Feature Extraction → Target Builder → DataBundle → Model Wrapper → Evaluator

Model Used:
RandomForestClassifier

Outputs:

- Accuracy for Type2
- Accuracy for Type2+Type3
- Accuracy for Type2+Type3+Type4
