import matplotlib.pyplot as plt
import numpy as np

def plot_communication_efficiency(rounds, baseline_comm, proposed_comm):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, baseline_comm, label='Baseline', marker='o')
    plt.plot(rounds, proposed_comm, label='Proposed', marker='o')
    plt.title("Comparison of Communication Overhead")
    plt.xlabel("Rounds")
    plt.ylabel("Data Transmitted (MB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/comm_efficiency.png")
    plt.show()

def plot_model_aggregation_performance(rounds, accuracy_baseline, accuracy_proposed, f1_score_baseline, f1_score_proposed):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracy_baseline, label='Accuracy Baseline', marker='o')
    plt.plot(rounds, accuracy_proposed, label='Accuracy Proposed', marker='o')
    plt.plot(rounds, f1_score_baseline, label='F1 Score Baseline', marker='o')
    plt.plot(rounds, f1_score_proposed, label='F1 Score Proposed', marker='o')
    plt.title("Model Aggregation Performance")
    plt.xlabel("Rounds")
    plt.ylabel("Performance Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/aggregation_performance.png")
    plt.show()

def plot_privacy_security_analysis(rounds, precision_baseline, precision_proposed, recall_baseline, recall_proposed):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, precision_baseline, label='Precision Baseline', marker='o')
    plt.plot(rounds, precision_proposed, label='Precision Proposed', marker='o')
    plt.plot(rounds, recall_baseline, label='Recall Baseline', marker='o')
    plt.plot(rounds, recall_proposed, label='Recall Proposed', marker='o')
    plt.title("Privacy and Security Analysis")
    plt.xlabel("Rounds")
    plt.ylabel("Performance Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/privacy_security.png")
    plt.show()

def plot_data_heterogeneity(rounds, data_heterogeneity_baseline, data_heterogeneity_proposed):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, data_heterogeneity_baseline, label='Baseline', marker='o')
    plt.plot(rounds, data_heterogeneity_proposed, label='Proposed', marker='o')
    plt.title("Robustness to Data Heterogeneity")
    plt.xlabel("Rounds")
    plt.ylabel("Performance Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/data_heterogeneity.png")
    plt.show()
