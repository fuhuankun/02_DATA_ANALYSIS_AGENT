# demo/demo.md

# Demo: Agent Interaction Examples

This file contains examples demonstrating the agent’s capabilities, including both single-turn and multi-turn sessions. Images illustrate each step.

---

## Single-Turn Example

### Example Question 0: What data do you have?

```text
# User input:
What data do you have?
```

![Example 0](demo_0.png)

---

## Multi-Turn Session Example

### Example Question 1: What data do you have?

```text
# User input:
What data do you have?
```

![Example 1](demo_1.png)

---

### Example Question 2: Can you give me the total quantity along with pipeline?

```text
# User input:
Can you give me the total quantity along with pipeline?
```

![Example 2](demo_2.png)

---

### Example Question 3: Do you see any outliers of quantity?

```text
# User input:
Do you see any outliers of quantity?
```

![Example 3](demo_3.png)

---

### Example Question 4 (follow-up): Are you calculating outliers along with pipeline, or using raw data?

```text
# User input:
Are you calculating outliers along with pipeline, or using raw data?
```

![Example 4](demo_4.png)

---

### Example Question 5: Can you provide me the trend of quantity along effective month?

```text
# User input:
Can you provide me the trend of quantity along effective month?
```

![Example 5](demo_5.png)
```text
Here is the graph store under result folder:
```
![Example 5](trend_0.png)
---

### Example Question 6 (follow-up): Can you explain it?

```text
# User input:
Can you explain it?
```

![Example 6](demo_6.png)

---

### Example Question 7: Do you see any correlated columns in the data?

```text
# User input:
Do you see any correlated columns in the data?
```

**Note:** As you can see, the agent has the feature of **self-correction by looping** if there’s an error.

![Example 7](demo_7.png)

---

### Example Question 8: Can you do some clustering analysis on the data? (you could ask for some particular variables as well)

```text
# User input:
Can you do some clustering analysis on the data? (you could ask for some particular variables as well)
```

![Example 8](demo_8.png)
```text
Here is the cluster graph stored under result (Do not interpret the results, just for demo 😄)
```
![Example 8](cluster_0.png)

