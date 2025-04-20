from matplotlib import pyplot as plt
import pandas as pd

# Define the tasks and schedule in a DataFrame similar to a Gantt chart layout
tasks = [
    "Conceptualization and Scoping",
    "Proposal Writing & Submission to SCM",
    "System Development (Flask, MongoDB, HTML/CSS/JS)",
    "Dashboard Implementation (Streamlit)",
    "Project Report Writing",
    "Project Presentation and Submission",
    "Corrections and Final Submission"
]

# Create a matrix where 1 represents filled (shaded) months
schedule = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Conceptualization and Scoping
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Proposal Writing
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # System Development
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # Dashboard Implementation
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # Project Report Writing
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Project Presentation
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # Corrections and Final Submission
]

# Convert to DataFrame
df = pd.DataFrame(schedule, columns=[str(i) for i in range(1, 13)], index=tasks)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['lightgrey' if val == 0 else 'black' for row in schedule for val in row]
bars = ax.imshow(schedule, cmap='Greys', aspect='auto')

# Set ticks
ax.set_xticks(range(12))
ax.set_xticklabels(range(1, 13))
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(tasks)

# Title and labels
ax.set_xlabel("Months")
ax.set_ylabel("Items of Work/Activities")

# Remove grid lines
ax.grid(False)

plt.tight_layout()
plt.show()

