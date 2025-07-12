import streamlit as st

st.title("Algorithms Analyzer Tool")

# Main Menu for Algorithm Strategies
strategy = st.sidebar.selectbox("Choose Strategy", ["Divide & Conquer", "Greedy", "Dynamic Programming", "Backtracking", "Branch & Bound"])

# Initialize empty variable for problem selection
problem = None

if strategy == "Divide & Conquer":
    problem = st.selectbox("Select a Problem", [
        "Binary Search",
        "Quick Sort",
        "Merge Sort",
        "Integer Arithmetic",
        "Maximum Sub-array"
    ])

elif strategy == "Greedy":
    problem = st.selectbox("Select a Problem", [
        "Knapsack Problem",
        "Job Scheduling",
        "Single Source Shortest Path (Dijkstra's Algorithm)"
    ])

elif strategy == "Dynamic Programming":
    problem = st.selectbox("Select a Problem", [
        "Binomial Coefficients",
        "Multistage Graphs",
        "0/1 Knapsack",
        "All Pair Shortest Path (Floyd-Warshall)",
        "Optimal Binary Search Tree (OBST)",
        "Bellman-Ford",
        "Subset Sum Problem"
    ])

elif strategy == "Backtracking":
    problem = st.selectbox("Select a Problem", [
        "8-Queens Problem",
        "Graph Coloring",
        "Subset Sum Problem",
        "Hamiltonian Cycle"
    ])

elif strategy == "Branch & Bound":
    problem = st.selectbox("Select a Problem", [
        "Knapsack Problem",
        "Traveling Salesman Problem (TSP)"
    ])

def edge_list_to_adjacency_list(edge_list):
        adjacency_list = {}
        for u, v, w in edge_list:
            if u not in adjacency_list:
                adjacency_list[u] = []
            adjacency_list[u].append((v, w))  # Add edge (u, v, w)
            # If it's an undirected graph, add (v, u, w) as well
            # if v not in adjacency_list:
            #     adjacency_list[v] = []
            # adjacency_list[v].append((u, w)) 
        return adjacency_list

if problem:
    st.write(f"## Solving {problem} using {strategy}")
    
    input_type = st.radio("Choose Input Method", ["User Input", "Random Input"])

    # Divide & Conquer Problems
    if strategy == "Divide & Conquer":
        if problem == "Binary Search":
            if input_type == "User Input":
                arr = st.text_input("Enter Array (comma-separated)").split(',')
                arr = list(map(int, arr)) if arr else []
                target = st.number_input("Enter Target Value", min_value=0)
            else:
                arr = list(range(1, 101))  # Random array
                target = 50  # Random target
            
        elif problem == "Quick Sort" or problem == "Merge Sort":
            if input_type == "User Input":
                arr = st.text_input("Enter Array (comma-separated)").split(',')
                arr = list(map(int, arr)) if arr else []
            else:
                arr = [10, 7, 8, 9, 1, 5]  # Example random array
                
        elif problem == "Integer Arithmetic":
            if input_type == "User Input":
                x = st.number_input("Enter First Integer", min_value=0)
                y = st.number_input("Enter Second Integer", min_value=0)
            else:
                x, y = 1234, 5678  # Random example

        elif problem == "Maximum Sub-array":
            if input_type == "User Input":
                arr = st.text_input("Enter Array (comma-separated)").split(',')
                arr = list(map(int, arr)) if arr else []
            else:
                arr = [-2, -3, 4, -1, -2, 1, 5, -3]  # Example random array
    
    # Greedy Problems
    elif strategy == "Greedy":
        if problem == "Knapsack Problem":
            if input_type == "User Input":
                max_weight = st.number_input("Enter Max Weight", min_value=1)
                weights = st.text_input("Enter Weights (comma-separated)").split(',')
                values = st.text_input("Enter Values (comma-separated)").split(',')
                weights = list(map(int, weights)) if weights else []
                values = list(map(int, values)) if values else []
            else:
                max_weight = 50
                weights = [10, 20, 30]
                values = [60, 100, 120]

        elif problem == "Job Scheduling":
            if input_type == "User Input":
                profits = st.text_input("Enter Profits (comma-separated)").split(',')
                deadlines = st.text_input("Enter Deadlines (comma-separated)").split(',')
                profits = list(map(int, profits)) if profits else []
                deadlines = list(map(int, deadlines)) if deadlines else []
            else:
                profits = [20, 15, 10, 5, 1]
                deadlines = [2, 2, 1, 3, 3]

        elif problem == "Single Source Shortest Path (Dijkstra's Algorithm)":
            if input_type == "User Input":
                num_nodes = st.number_input("Enter Number of Nodes", min_value=2)
                edges = st.text_area("Enter Edges (format: 'node1,node2,weight' on each line)")
                graph = []
                for line in edges.splitlines():
                    u, v, w = map(int, line.split(','))
                    graph.append((u, v, w))
                graph_dict = edge_list_to_adjacency_list(graph)
                source = st.number_input("Enter Source Node", min_value=0)
            else:
                graph = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
                graph_dict = edge_list_to_adjacency_list(graph)
                source = 0
    
    # Dynamic Programming Problems
    elif strategy == "Dynamic Programming":
        if problem == "Binomial Coefficients":
            if input_type == "User Input":
                n = st.number_input("Enter n", min_value=0)
                k = st.number_input("Enter k", min_value=0)
            else:
                n, k = 5, 2

        elif problem == "Multistage Graphs":
            if input_type == "User Input":
                stages = st.number_input("Enter Number of Stages", min_value=2)
                edges = st.text_area("Enter Edges with Costs (format: 'stage1,node1,stage2,node2,cost' on each line)")
                graph = []
                for line in edges.splitlines():
                    s1, u, s2, v, cost = map(int, line.split(','))
                    graph.append((s1, u, s2, v, cost))
            else:
                graph = [(1, 1, 2, 2, 2), (1, 1, 2, 3, 3), (2, 2, 3, 4, 4), (2, 3, 3, 4, 1)]
                stages = 3

        elif problem == "0/1 Knapsack":
            if input_type == "User Input":
                max_weight = st.number_input("Enter Max Weight", min_value=1)
                weights = st.text_input("Enter Weights (comma-separated)").split(',')
                values = st.text_input("Enter Values (comma-separated)").split(',')
                weights = list(map(int, weights)) if weights else []
                values = list(map(int, values)) if values else []
            else:
                max_weight = 50
                weights = [10, 20, 30]
                values = [60, 100, 120]

        elif problem == "All Pair Shortest Path (Floyd-Warshall)":
            if input_type == "User Input":
                num_nodes = st.number_input("Enter Number of Nodes", min_value=2)
                matrix = st.text_area("Enter Adjacency Matrix (comma-separated rows)")
                graph = []
                for row in matrix.splitlines():
                    graph.append(list(map(int, row.split(','))))
            else:
                graph = [[0, 3, float('inf'), 7], [8, 0, 2, float('inf')], [5, float('inf'), 0, 1], [2, float('inf'), float('inf'), 0]]

        elif problem == "Optimal Binary Search Tree (OBST)":
            if input_type == "User Input":
                keys = st.text_input("Enter Keys (comma-separated)").split(',')
                freq = st.text_input("Enter Frequencies (comma-separated)").split(',')
                keys = list(map(int, keys)) if keys else []
                freq = list(map(float, freq)) if freq else []
            else:
                keys = [10, 12, 20]
                freq = [0.34, 0.33, 0.33]

        elif problem == "Bellman-Ford":
            if input_type == "User Input":
                num_nodes = st.number_input("Enter Number of Nodes", min_value=2)
                edges = st.text_area("Enter Edges (format: 'node1,node2,weight' on each line)")
                graph = []
                for line in edges.splitlines():
                    u, v, w = map(int, line.split(','))
                    graph.append((u, v, w))
                source = st.number_input("Enter Source Node", min_value=0)
            else:
                graph = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
                source = 0

        elif problem == "Subset Sum Problem":
            if input_type == "User Input":
                target_sum = st.number_input("Enter Target Sum", min_value=0)
                nums = st.text_input("Enter Numbers (comma-separated)").split(',')
                nums = list(map(int, nums)) if nums else []
            else:
                target_sum = 10
                nums = [2, 3, 7, 8, 10]

    # Backtracking Problems
    elif strategy == "Backtracking":
        if problem == "8-Queens Problem":
            # No input required for this problem
            st.write("No additional input required for 8-Queens Problem.")
            
        elif problem == "Graph Coloring":
            if input_type == "User Input":
                num_nodes = st.number_input("Enter Number of Nodes", min_value=2)
                edges = st.text_area("Enter Edges (format: 'node1,node2' on each line)")
                num_colors = st.number_input("Enter Number of Colors", min_value=1)
            else:
                num_nodes = 4
                edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
                num_colors = 3

        elif problem == "Subset Sum Problem":
            if input_type == "User Input":
                target_sum = st.number_input("Enter Target Sum", min_value=0)
                nums = st.text_input("Enter Numbers (comma-separated)").split(',')
                nums = list(map(int, nums)) if nums else []
            else:
                target_sum = 10
                nums = [2, 3, 7, 8, 10]

        elif problem == "Hamiltonian Cycle":
            if input_type == "User Input":
                num_nodes = st.number_input("Enter Number of Nodes", min_value=2)
                edges = st.text_area("Enter Edges (format: 'node1,node2' on each line)")
                graph = []
                for line in edges.splitlines():
                    u, v = map(int, line.split(','))
                    graph.append((u, v))
            else:
                num_nodes = 5
                graph = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (1, 4)]

    # Branch & Bound Problems
    elif strategy == "Branch & Bound":
        if problem == "Knapsack Problem":
            if input_type == "User Input":
                max_weight = st.number_input("Enter Max Weight", min_value=1)
                weights = st.text_input("Enter Weights (comma-separated)").split(',')
                values = st.text_input("Enter Values (comma-separated)").split(',')
                weights = list(map(int, weights)) if weights else []
                values = list(map(int, values)) if values else []
            else:
                max_weight = 50
                weights = [10, 20, 30]
                values = [60, 100, 120]

        elif problem == "Traveling Salesman Problem (TSP)":
            if input_type == "User Input":
                num_cities = st.number_input("Enter Number of Cities", min_value=2)
                distances = st.text_area("Enter Distance Matrix (comma-separated rows)")
                graph = []
                for row in distances.splitlines():
                    graph.append(list(map(int, row.split(','))))
            else:
                graph = [
                    [0, 10, 15, 20],
                    [10, 0, 35, 25],
                    [15, 35, 0, 30],
                    [20, 25, 30, 0]
                ]

import numpy as np

# Divide & Conquer Functions

def binary_search(arr, target):
    """Binary Search implementation using Divide & Conquer."""
    def search_recursive(arr, target, left, right):
        if left > right:
            return -1
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return search_recursive(arr, target, mid + 1, right)
        else:
            return search_recursive(arr, target, left, mid - 1)

    return search_recursive(arr, target, 0, len(arr) - 1)


def quick_sort(arr):
    """Quick Sort implementation using Divide & Conquer."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr):
    """Merge Sort implementation using Divide & Conquer."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def integer_multiplication(x, y):
    """Integer multiplication using Divide & Conquer (Karatsuba's algorithm)."""
    if x < 10 or y < 10:
        return x * y
    n = max(len(str(x)), len(str(y)))
    half = n // 2
    a, b = divmod(x, 10**half)
    c, d = divmod(y, 10**half)
    ac = integer_multiplication(a, c)
    bd = integer_multiplication(b, d)
    ad_plus_bc = integer_multiplication(a + b, c + d) - ac - bd
    return ac * 10**(2 * half) + (ad_plus_bc * 10**half) + bd

def max_subarray(arr):
    """Maximum Subarray problem using Divide & Conquer."""
    def max_crossing_sum(arr, left, mid, right):
        left_sum = right_sum = float('-inf')
        temp_sum = 0
        for i in range(mid, left - 1, -1):
            temp_sum += arr[i]
            left_sum = max(left_sum, temp_sum)
        temp_sum = 0
        for i in range(mid + 1, right + 1):
            temp_sum += arr[i]
            right_sum = max(right_sum, temp_sum)
        return left_sum + right_sum

    def max_subarray_recursive(arr, left, right):
        if left == right:
            return arr[left]
        mid = (left + right) // 2
        return max(
            max_subarray_recursive(arr, left, mid),
            max_subarray_recursive(arr, mid + 1, right),
            max_crossing_sum(arr, left, mid, right)
        )
    
    return max_subarray_recursive(arr, 0, len(arr) - 1)


# Greedy Functions

def knapsack_greedy(weights, values, max_weight):
    """Fractional Knapsack Problem solved using Greedy strategy."""
    index = list(range(len(values)))
    ratio = [v / w for v, w in zip(values, weights)]
    index.sort(key=lambda i: ratio[i], reverse=True)
    max_value = 0
    for i in index:
        if weights[i] <= max_weight:
            max_value += values[i]
            max_weight -= weights[i]
        else:
            max_value += values[i] * (max_weight / weights[i])
            break
    return max_value


def job_scheduling(profits, deadlines):
    """Job Scheduling Problem solved using Greedy strategy."""
    jobs = sorted(zip(profits, deadlines), key=lambda x: x[0], reverse=True)
    max_deadline = max(deadlines)
    slots = [-1] * max_deadline
    total_profit = 0
    for profit, deadline in jobs:
        for j in range(min(deadline, max_deadline) - 1, -1, -1):
            if slots[j] == -1:
                slots[j] = profit
                total_profit += profit
                break
    return total_profit


def dijkstra(graph_dict, source):
    """Single Source Shortest Path using Dijkstra's Algorithm."""
    n = len(graph_dict)  # Assuming node numbering starts from 0
    visited = [False] * n
    distance = [float('inf')] * n
    distance[source] = 0  # Initialize source distance
    for _ in range(n):
        u = min(graph_dict.keys(), key=lambda v: distance[v] if not visited[v] else float('inf'))
        visited[u] = True
        for v, weight in graph_dict.get(u, []):  # Safe edge access
            if distance[u] + weight < distance[v]:  # Direct index access
                distance[v] = distance[u] + weight
    return distance


# Dynamic Programming Functions

def binomial_coefficient(n, k):
    """Calculate binomial coefficient C(n, k) using Dynamic Programming."""
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            if j == 0 or j == i:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
    return dp[n][k]

def multistage_graph(stages, graph):
    """Multistage graph problem using Dynamic Programming."""
    INF = float('inf')

    # Identify unique nodes from the graph
    nodes = set([u for s1, u, s2, v, cost in graph] + [v for s1, u, s2, v, cost in graph])
    costs = {node: INF for node in nodes}
    parent = {}

    # Base case: Cost of last stage is 0
    costs[stages] = 0  # Assuming the last node is represented by 'stages'

    # Recursively calculate costs for other stages
    for stage in range(stages - 1, 0, -1):
        for s1, u, s2, v, cost in graph:
            if s1 == stage:
                if costs[u] > cost + costs[v]:
                    costs[u] = cost + costs[v]
                    parent[u] = v  # Store parent for path reconstruction

    return costs, parent

def knapsack_dp(weights, values, max_weight):
    """0/1 Knapsack Problem using Dynamic Programming."""
    n = len(weights)
    dp = [[0] * (max_weight + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(max_weight + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][max_weight]


def floyd_warshall(graph):
    """All-Pairs Shortest Path using Floyd-Warshall algorithm."""
    n = len(graph)
    dist = [[graph[i][j] if i != j else 0 for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def obst(keys, freq):
    """Optimal Binary Search Tree (OBST) using Dynamic Programming."""
    n = len(keys)
    cost = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        cost[i][i] = freq[i]

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            cost[i][j] = float('inf')
            for r in range(i, j + 1):
                c = (cost[i][r - 1] if r > i else 0) + \
                    (cost[r + 1][j] if r < j else 0) + \
                    sum(freq[i:j + 1])
                cost[i][j] = min(cost[i][j], c)
    return cost[0][n - 1]

def bellman_ford(graph, source, num_nodes):
    """Bellman-Ford algorithm for Single Source Shortest Path with negative weights."""
    distance = [float('inf')] * num_nodes
    distance[source] = 0
    for _ in range(num_nodes - 1):
        for u, v, w in graph:
            if distance[u] != float('inf') and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
    return distance

def sum_of_subsets_dp(nums, target_sum):
    """Subset Sum problem using Dynamic Programming."""
    n = len(nums)
    dp = [[False] * (target_sum + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = True

    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            dp[i][j] = dp[i - 1][j]
            if nums[i - 1] <= j:
                dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
    
    return dp[n][target_sum]

# Functions for Backtracking

def solve_8_queens():
    """8-Queens Problem using Backtracking."""
    result = []
    board = [-1] * 8

    def is_safe(row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    def backtrack(row):
        if row == 8:
            result.append(board[:])
            return
        for col in range(8):
            if is_safe(row, col):
                board[row] = col
                backtrack(row)
                board[row] = -1

    backtrack(0)
    return result

def graph_coloring(edges, num_colors):
    """Graph Coloring Problem using Backtracking."""
    colors = [-1] * len(edges)

    def is_safe(v, color):
        for i in range(len(edges)):
            if edges[v][i] == 1 and colors[i] == color:
                return False
        return True

    def color_graph(v):
        if v == len(edges):
            return True
        for c in range(num_colors):
            if is_safe(v, c):
                colors[v] = c
                if color_graph(v + 1):
                    return True
                colors[v] = -1
        return False

    return color_graph(0), colors if colors[0] != -1 else None

def sum_of_subsets(nums, target_sum):
    """Sum of Subsets Problem using Backtracking."""
    result = []

    def backtrack(index, current_sum, subset):
        if current_sum == target_sum:
            result.append(subset[:])
            return
        if current_sum > target_sum or index >= len(nums):
            return
        subset.append(nums[index])
        backtrack(index + 1, current_sum + nums[index], subset)
        subset.pop()
        backtrack(index + 1, current_sum, subset)

    backtrack(0, 0, [])
    return result


def hamiltonian_cycle(graph, num_nodes):
    """Hamiltonian Cycle using Backtracking."""
    path = [-1] * num_nodes
    path[0] = 0  # start at the first node

    def is_safe(v, pos):
        if graph[path[pos - 1]][v] == 0:
            return False
        if v in path[:pos]:
            return False
        return True

    def backtrack(pos):
        if pos == num_nodes:
            return graph[path[pos - 1]][path[0]] == 1  # check if last vertex connects to the first
        for v in range(1, num_nodes):
            if is_safe(v, pos):
                path[pos] = v
                if backtrack(pos + 1):
                    return True
                path[pos] = -1
        return False

    return backtrack(1), path if path[0] != -1 else None

# Branch & Bound Functions

from queue import PriorityQueue

def knapsack_branch_bound(weights, values, max_weight):
    """0/1 Knapsack Problem using Branch & Bound."""
    n = len(values)

    class Node:
        def __init__(self, level, value, weight, bound):
            self.level = level
            self.value = value
            self.weight = weight
            self.bound = bound

        def __lt__(self, other):
            return self.bound > other.bound

    def bound(u):
        if u.weight >= max_weight:
            return 0
        profit_bound = u.value
        j = u.level + 1
        total_weight = u.weight
        while j < n and total_weight + weights[j] <= max_weight:
            total_weight += weights[j]
            profit_bound += values[j]
            j += 1
        if j < n:
            profit_bound += (max_weight - total_weight) * values[j] / weights[j]
        return profit_bound

    Q = PriorityQueue()
    u = Node(-1, 0, 0, 0)
    u.bound = bound(u)
    Q.put(u)
    max_profit = 0
    while not Q.empty():
        u = Q.get()
        if u.bound > max_profit:
            v = Node(u.level + 1, u.value + values[u.level + 1], u.weight + weights[u.level + 1], 0)
            if v.weight <= max_weight and v.value > max_profit:
                max_profit = v.value
            v.bound = bound(v)
            if v.bound > max_profit:
                Q.put(v)
            v = Node(u.level + 1, u.value, u.weight, 0)
            v.bound = bound(v)
            if v.bound > max_profit:
                Q.put(v)
    return max_profit


def tsp_branch_bound(graph):
    """Traveling Salesman Problem (TSP) using Branch & Bound."""
    n = len(graph)
    min_cost = float('inf')
    visited = [False] * n

    def branch_bound(curr_pos, count, cost, path):
        nonlocal min_cost
        if count == n and graph[curr_pos][0] > 0:
            min_cost = min(min_cost, cost + graph[curr_pos][0])
            return
        for i in range(n):
            if not visited[i] and graph[curr_pos][i] > 0:
                visited[i] = True
                branch_bound(i, count + 1, cost + graph[curr_pos][i], path + [i])
                visited[i] = False

    visited[0] = True
    branch_bound(0, 1, 0, [0])
    return min_cost

def visualize_binary_search(arr, target):
    left, right = 0, len(arr) - 1
    step = 1
    st.write("### Binary Search Visualization")
    
    while left <= right:
        mid = (left + right) // 2
        st.write(f"**Step {step}:** Checking middle element at index {mid}")
        
        # Display current state of array
        st.write(f"Array: {arr}")
        st.write(f"Left index: {left} ({arr[left]}) | Mid index: {mid} ({arr[mid]}) | Right index: {right} ({arr[right]})")
        
        if arr[mid] == target:
            st.success(f"Target found at index {mid}!")
            return mid
        elif arr[mid] < target:
            st.write(f"Target is greater than {arr[mid]}, so we search the right half.")
            left = mid + 1
        else:
            st.write(f"Target is less than {arr[mid]}, so we search the left half.")
            right = mid - 1
        
        step += 1
    st.error("Target not found.")
    return -1

def visualize_quick_sort(arr, start=0, end=None, depth=0):
    if end is None:
        end = len(arr) - 1
    
    if start < end:
        st.write(f"**Depth {depth}:** Quick Sort on array segment {arr[start:end+1]}")
        
        # Partition the array and get pivot index
        pi = partition(arr, start, end)
        st.write(f"Pivot chosen: {arr[pi]} at index {pi}")
        st.write(f"Array after partition: {arr}")
        
        # Recursively sort elements
        visualize_quick_sort(arr, start, pi - 1, depth + 1)
        visualize_quick_sort(arr, pi + 1, end, depth + 1)

def partition(arr, start, end):
    pivot = arr[end]
    i = start - 1
    for j in range(start, end):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[end] = arr[end], arr[i + 1]
    return i + 1

def visualize_merge_sort(arr):
    """Visualize Merge Sort using Divide and Conquer."""
    st.write("### Merge Sort Visualization")
    
    def merge_sort(arr, level=0):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]

            st.write(f"Level {level}: Split array into {left_half} and {right_half}")

            merge_sort(left_half, level + 1)
            merge_sort(right_half, level + 1)

            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1

            st.write(f"Level {level}: Merged into {arr}")
    
    merge_sort(arr)
    st.success(f"Sorted array: {arr}")

def visualize_integer_multiplication(x, y):
    """Visualize Integer Multiplication using Karatsuba�s Algorithm."""
    st.write("### Integer Multiplication Visualization")

    def karatsuba(x, y, level=0):
        st.write(f"Level {level}: Multiplying {x} by {y}")

        if x < 10 or y < 10:
            st.write(f"Level {level}: Base case multiplication = {x * y}")
            return x * y

        n = max(len(str(x)), len(str(y)))
        half = n // 2

        a, b = divmod(x, 10**half)
        c, d = divmod(y, 10**half)

        ac = karatsuba(a, c, level + 1)
        bd = karatsuba(b, d, level + 1)
        ad_plus_bc = karatsuba(a + b, c + d, level + 1) - ac - bd

        result = ac * 10**(2 * half) + (ad_plus_bc * 10**half) + bd
        st.write(f"Level {level}: Result of {x} * {y} = {result}")
        return result

    result = karatsuba(x, y)
    st.success(f"Multiplication result: {result}")

def visualize_max_subarray(arr):
    """Visualize Maximum Sub-array using Divide and Conquer."""
    st.write("### Maximum Sub-array Visualization")
    
    def max_crossing_sum(arr, left, mid, right):
        left_sum = right_sum = float('-inf')
        total = 0

        for i in range(mid, left - 1, -1):
            total += arr[i]
            left_sum = max(left_sum, total)

        total = 0
        for i in range(mid + 1, right + 1):
            total += arr[i]
            right_sum = max(right_sum, total)

        st.write(f"Crossing sum: Left sum = {left_sum}, Right sum = {right_sum}")
        return left_sum + right_sum

    def max_subarray(arr, left, right):
        if left == right:
            st.write(f"Base case at index {left} with value {arr[left]}")
            return arr[left]

        mid = (left + right) // 2
        st.write(f"Divide at index {mid}")

        left_sum = max_subarray(arr, left, mid)
        right_sum = max_subarray(arr, mid + 1, right)
        cross_sum = max_crossing_sum(arr, left, mid, right)

        result = max(left_sum, right_sum, cross_sum)
        st.write(f"Conquer: Max of {left_sum}, {right_sum}, {cross_sum} = {result}")
        return result

    max_sum = max_subarray(arr, 0, len(arr) - 1)
    st.success(f"Maximum sub-array sum: {max_sum}")

def visualize_greedy_knapsack(weights, values, max_weight):
    """Visualize Knapsack using Greedy Approach."""
    n = len(values)
    items = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
    items.sort(reverse=True)

    st.write("### Greedy Knapsack Visualization")
    total_value = 0
    for i, (ratio, weight, value) in enumerate(items):
        if max_weight == 0:
            break
        if weight <= max_weight:
            st.write(f"Step {i + 1}: Taking full item with weight {weight} and value {value}")
            max_weight -= weight
            total_value += value
        else:
            st.write(f"Step {i + 1}: Taking fraction of item with weight {max_weight} and value {value * (max_weight / weight)}")
            total_value += value * (max_weight / weight)
            max_weight = 0

    st.success(f"Total value achieved: {total_value}")

def visualize_job_scheduling(profits, deadlines):
    """Visualize Job Scheduling using Greedy Approach."""
    st.write("### Job Scheduling Problem Visualization")
    jobs = sorted([(i + 1, deadlines[i], profits[i]) for i in range(len(profits))], key=lambda x: x[2], reverse=True)  # Sort by profit

    max_deadline = max(job[1] for job in jobs)
    schedule = [None] * max_deadline
    total_profit = 0

    for job in jobs:
        job_id, deadline, profit = job
        st.write(f"Attempting to schedule job {job_id} with deadline {deadline} and profit {profit}")

        for j in range(min(deadline - 1, max_deadline - 1), -1, -1):
            if schedule[j] is None:
                schedule[j] = job_id
                total_profit += profit
                st.write(f"Scheduled job {job_id} at slot {j+1}")
                break
            else:
                st.write(f"Slot {j+1} already occupied.")

    st.success(f"Total profit: {total_profit}")
    st.write(f"Final Job Schedule: {schedule}")

import numpy as np

def visualize_dijkstra(graph, source):
    n = len(graph)
    visited = [False] * n
    distance = [float('inf')] * n
    distance[source] = 0
    st.write("### Dijkstra's Algorithm Visualization")
    
    for i in range(n):
        u = min(range(n), key=lambda v: distance[v] if not visited[v] else float('inf'))
        visited[u] = True
        st.write(f"**Step {i+1}:** Mark node {u} as visited with shortest known distance: {distance[u]}")
        
        for v, weight in graph[u]:
            if not visited[v] and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                st.write(f"Updated distance of node {v} to {distance[v]} via node {u}.")
    
    st.write("### Final Shortest Distances from Source Node")
    for i, d in enumerate(distance):
        st.write(f"Node {i}: {d}")
    return distance

def visualize_binomial_coeff(n, k):
    """Visualize Binomial Coefficient Calculation using Dynamic Programming."""
    st.write("### Binomial Coefficient Calculation Visualization")
    dp = [[0] * (k + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            if j == 0 or j == i:
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
            st.write(f"C({i}, {j}) = {dp[i][j]}")

    st.success(f"Binomial Coefficient C({n}, {k}) = {dp[n][k]}")
    return dp[n][k]

def visualize_multistage_graph(stages, graph):
    """Visualize Multistage Graph using Dynamic Programming."""
    st.write("### Multistage Graph Visualization")
    
    n = len(graph)
    dp = [float('inf')] * n
    dp[-1] = 0

    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if graph[i][j] != float('inf'):
                dp[i] = min(dp[i], graph[i][j] + dp[j])
                st.write(f"Updated cost for stage {i}: {dp[i]}")
    
    st.success(f"Minimum cost from start to end: {dp[0]}")
    return dp[0]

def visualize_knapsack_dp(weights, values, max_weight):
    n = len(weights)
    st.write("### 0/1 Knapsack Problem (DP) Visualization")
    
    # Recursive function to determine optimal value with tracking of choices
    def knapsack_recursive(i, remaining_weight):
        if i == 0 or remaining_weight == 0:
            return 0

        if weights[i - 1] > remaining_weight:
            st.write(f"Item {i-1} too heavy (Weight: {weights[i-1]}, Remaining Capacity: {remaining_weight}) - Skipping")
            return knapsack_recursive(i - 1, remaining_weight)
        
        # Calculate values for including or excluding current item
        exclude = knapsack_recursive(i - 1, remaining_weight)
        include = values[i - 1] + knapsack_recursive(i - 1, remaining_weight - weights[i - 1])
        
        # Choose the option with the maximum value
        if include > exclude:
            st.write(f"Including item {i-1}: Value: {values[i-1]}, Weight: {weights[i-1]}, Remaining Capacity: {remaining_weight}")
            return include
        else:
            st.write(f"Excluding item {i-1}: Value: {values[i-1]}, Weight: {weights[i-1]}, Remaining Capacity: {remaining_weight}")
            return exclude
    
    # Start solving the knapsack problem
    optimal_value = knapsack_recursive(n, max_weight)
    st.success(f"Optimal value for the given weight limit ({max_weight}) is: {optimal_value}")
    return optimal_value

def visualize_floyd_warshall(graph):
    """Visualize Floyd-Warshall Algorithm using Dynamic Programming."""
    st.write("### Floyd-Warshall Algorithm Visualization")
    n = len(graph)
    dist = [row[:] for row in graph]  # Copy of graph for distance updates

    for k in range(n):
        st.write(f"Considering intermediate vertex {k}")
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    st.write(f"Updated dist[{i}][{j}] to {dist[i][j]} via vertex {k}")

    st.write("### Final Shortest Distances Matrix")
    for row in dist:
        st.write(row)
    return dist

def visualize_bellman_ford(graph, source):
    """Visualize Bellman-Ford Algorithm using Dynamic Programming."""
    st.write("### Bellman-Ford Algorithm Visualization")
    n = len(graph)
    distance = [float('inf')] * n
    distance[source] = 0

    for i in range(n - 1):
        st.write(f"Iteration {i + 1}")
        for u, v, weight in graph:
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                st.write(f"Updated distance of {v} to {distance[v]} via {u}")

    # Check for negative-weight cycles
    for u, v, weight in graph:
        if distance[u] != float('inf') and distance[u] + weight < distance[v]:
            st.warning("Graph contains negative weight cycle")
            return

    st.success("Final distances from source:")
    for i, dist in enumerate(distance):
        st.write(f"Node {i}: {dist}")

def visualize_obst(keys, freq):
    """Visualize Optimal Binary Search Tree construction using Dynamic Programming."""
    n = len(keys)
    dp = [[0] * n for _ in range(n)]
    cost = [[0] * n for _ in range(n)]

    st.write("### Optimal Binary Search Tree (OBST) Visualization")
    for i in range(n):
        dp[i][i] = freq[i]
        st.write(f"Initial cost for single key {keys[i]} with frequency {freq[i]}: {dp[i][i]}")

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            st.write(f"Calculating cost for keys {keys[i:j + 1]}")

            for r in range(i, j + 1):
                left = dp[i][r - 1] if r > i else 0
                right = dp[r + 1][j] if r < j else 0
                total_cost = left + right + sum(freq[i:j + 1])

                if total_cost < dp[i][j]:
                    dp[i][j] = total_cost
                    st.write(f"Optimal cost with root {keys[r]}: {dp[i][j]}")

    st.success(f"Minimum cost for OBST: {dp[0][n - 1]}")
    return dp[0][n - 1]

def visualize_sum_of_subsets(nums, target_sum):
    """Visualize Sum of Subsets using Dynamic Programming."""
    st.write("### Sum of Subsets Problem (DP) Visualization")
    n = len(nums)
    dp = [[False] * (target_sum + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        dp[i][0] = True

    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            if j < nums[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            st.write(f"Subset sum for first {i} elements with sum {j}: {dp[i][j]}")

    st.success(f"Is sum {target_sum} possible: {dp[n][target_sum]}")
    return dp[n][target_sum]

def visualize_8_queens():
    board = [-1] * 8
    solutions = []

    def is_safe(row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    def backtrack(row):
        if row == 8:
            solutions.append(board[:])
            st.write(f"Solution found: {board}")
            return
        for col in range(8):
            if is_safe(row, col):
                board[row] = col
                st.write(f"Placing queen at row {row}, col {col}")
                backtrack(row)
                board[row] = -1
                st.write(f"Backtracking from row {row}, col {col}")

    backtrack(0)
    st.success(f"Total solutions found: {len(solutions)}")
    return solutions

def visualize_graph_coloring(graph, m):
    """Visualize Graph Coloring using Backtracking."""
    n = len(graph)
    colors = [-1] * n

    st.write("### Graph Coloring Visualization")

    def is_safe(v, c):
        for i in range(n):
            if graph[v][i] == 1 and colors[i] == c:
                return False
        return True

    def graph_coloring(v):
        if v == n:
            st.success(f"All nodes colored: {colors}")
            return True

        for c in range(1, m + 1):
            if is_safe(v, c):
                colors[v] = c
                st.write(f"Colored node {v} with color {c}")
                if graph_coloring(v + 1):
                    return True
                colors[v] = -1
                st.write(f"Backtracked at node {v}, removing color {c}")

        return False

    if not graph_coloring(0):
        st.warning("No solution exists.")
    return colors

def visualize_hamiltonian_cycle(graph):
    """Visualize Hamiltonian Cycle using Backtracking."""
    n = len(graph)
    path = [-1] * n
    path[0] = 0

    st.write("### Hamiltonian Cycle Visualization")

    def is_safe(v, pos):
        if graph[path[pos - 1]][v] == 0:
            return False
        if v in path:
            return False
        return True

    def hamiltonian_cycle(pos):
        if pos == n:
            if graph[path[pos - 1]][path[0]] == 1:
                st.success(f"Hamiltonian Cycle found: {path + [path[0]]}")
                return True
            return False

        for v in range(1, n):
            if is_safe(v, pos):
                path[pos] = v
                st.write(f"Path so far: {path[:pos + 1]}")
                if hamiltonian_cycle(pos + 1):
                    return True
                path[pos] = -1
                st.write(f"Backtracking from vertex {v}")

        return False

    if not hamiltonian_cycle(1):
        st.warning("No Hamiltonian Cycle found")

def visualize_sum_of_subsets_backtracking(nums, target_sum):
    """Visualize Sum of Subsets using Backtracking."""
    st.write("### Sum of Subsets Problem (Backtracking) Visualization")
    solution = []
    found = False

    def subset_sum_recursive(i, current_sum, subset):
        nonlocal found
        if current_sum == target_sum:
            st.success(f"Subset found: {subset}")
            found = True
            return
        if i >= len(nums) or current_sum > target_sum:
            return

        subset_sum_recursive(i + 1, current_sum + nums[i], subset + [nums[i]])
        if not found:
            subset_sum_recursive(i + 1, current_sum, subset)

    subset_sum_recursive(0, 0, [])
    if not found:
        st.warning("No subset with the given sum found.")

import queue

def visualize_knapsack_branch_bound(weights, values, max_weight):
    """Visualize Knapsack using Branch and Bound."""
    n = len(weights)
    st.write("### Knapsack Problem (Branch and Bound) Visualization")

    class Node:
        def __init__(self, level, profit, weight, bound):
            self.level = level
            self.profit = profit
            self.weight = weight
            self.bound = bound

    def bound(node):
        if node.weight >= max_weight:
            return 0
        profit_bound = node.profit
        j = node.level + 1
        total_weight = node.weight

        while j < n and total_weight + weights[j] <= max_weight:
            total_weight += weights[j]
            profit_bound += values[j]
            j += 1

        if j < n:
            profit_bound += (max_weight - total_weight) * (values[j] / weights[j])
        
        return profit_bound

    max_profit = 0
    Q = queue.Queue()
    Q.put(Node(-1, 0, 0, 0))

    while not Q.empty():
        node = Q.get()
        if node.level == n - 1:
            continue

        next_level = node.level + 1
        include = Node(next_level, node.profit + values[next_level],
                       node.weight + weights[next_level], 0)
        include.bound = bound(include)
        
        st.write(f"Evaluating node with profit {include.profit} and weight {include.weight}")

        if include.weight <= max_weight and include.profit > max_profit:
            max_profit = include.profit
            st.write(f"Updated max profit to {max_profit}")

        if include.bound > max_profit:
            Q.put(include)

        exclude = Node(next_level, node.profit, node.weight, 0)
        exclude.bound = bound(exclude)
        if exclude.bound > max_profit:
            Q.put(exclude)

    st.success(f"Maximum profit: {max_profit}")

from queue import PriorityQueue

def visualize_tsp_branch_bound(graph):
    n = len(graph)
    min_cost = float('inf')
    visited = [False] * n
    st.write("### Traveling Salesman Problem (TSP) using Branch and Bound")

    def branch_bound(curr_pos, count, cost, path):
        nonlocal min_cost
        if count == n and graph[curr_pos][0] > 0:
            min_cost = min(min_cost, cost + graph[curr_pos][0])
            st.write(f"Complete tour found with cost: {cost + graph[curr_pos][0]}")
            return
        for i in range(n):
            if not visited[i] and graph[curr_pos][i] > 0:
                visited[i] = True
                st.write(f"Visit node {i} from {curr_pos}, partial path cost: {cost + graph[curr_pos][i]}")
                branch_bound(i, count + 1, cost + graph[curr_pos][i], path + [i])
                visited[i] = False
                st.write(f"Backtrack from node {i} to {curr_pos}")

    visited[0] = True
    branch_bound(0, 1, 0, [0])
    st.success(f"Minimum cost for the TSP is: {min_cost}")
    return min_cost

# Binary Search
if problem == "Binary Search":
    if st.button("Run Binary Search"):
        binary_search(arr, target)
        visualize_binary_search(arr, target)

# Quick Sort
elif problem == "Quick Sort":
    if st.button("Run Quick Sort"):
        quick_sort(arr)
        visualize_quick_sort(arr, start=0, end=None, depth=0)

# Merge Sort
elif problem == "Merge Sort":
    if st.button("Run Merge Sort"):
        merge_sort(arr)
        visualize_merge_sort(arr)

# Integer Arithmetic (example: exponentiation)
elif problem == "Integer Arithmetic":
    if st.button("Run Integer Arithmetic (Multiplication)"):
        result = integer_multiplication(x, y)
        visualize_integer_multiplication(x, y)

# Maximum Subarray (using Kadane's Algorithm)
elif problem == "Maximum Sub-array":
    if st.button("Run Maximum Subarray"):
        max_sum = max_subarray(arr)
        visualize_max_subarray(arr)

# Knapsack Problem (Greedy)
elif problem == "Knapsack Problem":
    if st.button("Run Knapsack (Greedy)"):
        max_profit = knapsack_greedy(weights, values, max_weight)
        visualize_greedy_knapsack(weights, values, max_weight)

# Job Scheduling Problem (Greedy)
elif problem == "Job Scheduling":
    if st.button("Run Job Scheduling"):
        schedule = job_scheduling(profits, deadlines)
        visualize_job_scheduling(profits, deadlines)

# Single Source Shortest Path (Dijkstra�s Algorithm)
elif problem == "Single Source Shortest Path (Dijkstra's Algorithm)":
    if st.button("Run Dijkstra's Algorithm"):
        distances = dijkstra(graph_dict, source)
        visualize_dijkstra(graph, source)

# Binomial Coefficients (Dynamic Programming)
elif problem == "Binomial Coefficients":
    if st.button("Run Binomial Coefficients"):
        min_cost = binomial_coefficient(n, k)
        visualize_binomial_coeff(n, k)

# Multistage Graph (Dynamic Programming)
elif problem == "Multistage Graphs":
    if st.button("Run Multistage Graphs"):
        min_cost = multistage_graph(stages, graph)
        visualize_multistage_graph(stages, graph)

# 0/1 Knapsack (Dynamic Programming)
elif problem == "0/1 Knapsack":
    if st.button("Run 0/1 Knapsack"):
        max_value = knapsack_dp(weights, values, max_weight)
        visualize_knapsack_dp(weights, values, max_weight)

# Floyd-Warshall (All Pair Shortest Path)
elif problem == "All Pair Shortest Path (Floyd-Warshall)":
    if st.button("Run Floyd-Warshall"):
        shortest_paths = floyd_warshall(graph)
        visualize_floyd_warshall(graph)

# Bellman-Ford Algorithm
elif problem == "Bellman-Ford":
    if st.button("Run Bellman-Ford"):
        distances = bellman_ford(graph, source, num_nodes)
        visualize_bellman_ford(graph, source)

# OBST (Dynamic Programming)
elif problem == "Optimal Binary Search Tree (OBST)":
    if st.button("Run Optimal Binary Search Tree (OBST)"):
        obst(keys, freq)
        visualize_obst(keys, freq)

# Sum of Subsets (Dynamic Programming)
elif problem == "Subset Sum Problem":
    if st.button("Run Subset Sum Problem"):
        sum_of_subsets_dp(nums, target_sum)
        visualize_sum_of_subsets(nums, target_sum)

# 8-Queens Problem (Backtracking)
elif problem == "8-Queens Problem":
    if st.button("Run 8-Queens Problem"):
        solve_8_queens()
        visualize_8_queens()

# Graph Coloring (Backtracking)
elif problem == "Graph Coloring":
    if st.button("Run Graph Coloring"):
        graph_coloring(edges, num_colors)
        visualize_graph_coloring(graph, m)

# Subset Sum Problem (Backtracking)
elif problem == "Subset Sum Problem":
    if st.button("Run Subset Sum Problem"):
        sum_of_subsets(nums, target_sum)
        visualize_sum_of_subsets_backtracking(nums, target_sum)

# Hamiltonian Cycle (Backtracking)
elif problem == "Hamiltonian Cycle":
    if st.button("Run Hamiltonian Cycle"):
        cycle = hamiltonian_cycle(graph, num_nodes)
        visualize_hamiltonian_cycle(graph)

# # TSP (Branch & Bound)
# elif problem == "TSP using Branch and Bound":
#     if st.button("Run TSP Branch & Bound"):
#         min_cost = tsp_branch_and_bound(graph)
#         visualize_tsp(graph)
