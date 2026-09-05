
import heapq
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def heuristic(a, b):
 
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(position, grid):
 
    row, col = position

    # Up, Down, Left, Right
    directions = [
        (-1, 0),   # Up
        (1, 0),    # Down
        (0, -1),   # Left
        (0, 1)     # Right
    ]

    rows = len(grid)
    cols = len(grid[0])

    neighbors = []

    for dr, dc in directions:

        new_row = row + dr
        new_col = col + dc

        # Check grid boundaries
        if 0 <= new_row < rows and 0 <= new_col < cols:

            # Do not move through obstacles
            if grid[new_row][new_col] != '#':
                neighbors.append((new_row, new_col))

    return neighbors

def find_start_goal(grid):

    start = None
    goal = None

    for row in range(len(grid)):

        for col in range(len(grid[row])):

            if grid[row][col] == 'S':
                start = (row, col)

            elif grid[row][col] == 'G':
                goal = (row, col)

    if start is None:
        raise ValueError("Starting position 'S' not found.")

    if goal is None:
        raise ValueError("Goal position 'G' not found.")

    return start, goal


def reconstruct_path(came_from, current):

    path = [current]

    while current in came_from:

        current = came_from[current]

        path.append(current)

    # Reverse so that path goes Start -> Goal
    path.reverse()

    return path

def astar(grid):
    # Start timer
    start_time = time.perf_counter()

    # Find S and G
    start, goal = find_start_goal(grid)

    open_set = []

    counter = 0

    # Initial heuristic
    initial_h = heuristic(start, goal)

    heapq.heappush(
        open_set,
        (initial_h, counter, start)
    )

    g_score = {
        start: 0
    }
    came_from = {}
    closed_set = set()

    explored = []
    while open_set:

        # Get cell with smallest f-score
        f_score, _, current = heapq.heappop(open_set)

        # Ignore duplicate entries
        if current in closed_set:
            continue

        # Mark current cell as processed
        closed_set.add(current)

        explored.append(current)

        if current == goal:

            path = reconstruct_path(
                came_from,
                current
            )

            path_cost = g_score[current]

            execution_time = (
                time.perf_counter() - start_time
            )

            return (
                path,
                explored,
                path_cost,
                execution_time
            )
        for neighbor in get_neighbors(current, grid):

            # Every movement costs 1
            tentative_g = g_score[current] + 1

            # Check whether this is a better path
            if (
                neighbor not in g_score
                or tentative_g < g_score[neighbor]
            ):

                # Store parent
                came_from[neighbor] = current

                # Store new cost
                g_score[neighbor] = tentative_g

                # Calculate heuristic
                h_score = heuristic(
                    neighbor,
                    goal
                )

                # A* formula:
                #
                # f(n) = g(n) + h(n)

                new_f_score = tentative_g + h_score

                counter += 1

                # Add to priority queue
                heapq.heappush(
                    open_set,
                    (
                        new_f_score,
                        counter,
                        neighbor
                    )
                )

    execution_time = (
        time.perf_counter() - start_time
    )

    return (
        None,
        explored,
        None,
        execution_time
    )


def print_results(
    test_name,
    grid,
    path,
    explored,
    path_cost,
    execution_time
):

    print()
    print("=" * 65)
    print(test_name)
    print("=" * 65)

    print("\nGrid:")

    for row in grid:
        print(" ".join(row))

    print()

    if path is not None:

        print("Path Found       : YES")

        print("Path Cost        :", path_cost)

        print("Nodes Explored   :", len(explored))

        print(
            "Execution Time   :",
            f"{execution_time:.8f}",
            "seconds"
        )

        print("\nFinal Path:")

        print(
            " -> ".join(
                "(" + str(r) + "," + str(c) + ")"
                for r, c in path
            )
        )

    else:

        print("Path Found       : NO")

        print("Path Cost        : N/A")

        print("Nodes Explored   :", len(explored))

        print(
            "Execution Time   :",
            f"{execution_time:.8f}",
            "seconds"
        )

        print("\nFinal Path:")

        print("No valid path exists.")

    print("=" * 65)

def visualize(
    grid,
    path,
    explored,
    filename
):
    rows = len(grid)
    cols = len(grid[0])


    visual = [
        [0 for _ in range(cols)]
        for _ in range(rows)
    ]

    start = None
    goal = None

    for r in range(rows):

        for c in range(cols):

            if grid[r][c] == '#':

                visual[r][c] = 1

            elif grid[r][c] == 'S':

                start = (r, c)

            elif grid[r][c] == 'G':

                goal = (r, c)

    for cell in explored:

        r, c = cell

        if visual[r][c] == 0:
            visual[r][c] = 2
            
    if path is not None:

        for cell in path:

            r, c = cell

            if cell != start and cell != goal:
                visual[r][c] = 3

    if start is not None:
        visual[start[0]][start[1]] = 4

    if goal is not None:
        visual[goal[0]][goal[1]] = 5

    plt.figure(figsize=(8, 6))

    plt.imshow(
        visual,
        interpolation="nearest"
    )

    # Grid coordinates
    plt.xticks(range(cols))
    plt.yticks(range(rows))

    # Grid lines
    plt.grid(
        True,
        linewidth=0.8
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.title("A* Path Planning")

    if start is not None:

        plt.text(
            start[1],
            start[0],
            "S",
            ha="center",
            va="center",
            fontweight="bold"
        )
        
    if goal is not None:

        plt.text(
            goal[1],
            goal[0],
            "G",
            ha="center",
            va="center",
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=150
    )

    # Close figure
    plt.close()

    print("\nVisualization saved as:", filename)
    
def main():
    grid1 = [
        list("S......."),
        list(".###...."),
        list("...#...."),
        list("...#...."),
        list("...###.."),
        list("........"),
        list("......G.")
    ]

    path, explored, cost, execution_time = astar(grid1)

    print_results(
        "TEST CASE 1 - PATH EXISTS",
        grid1,
        path,
        explored,
        cost,
        execution_time
    )

    visualize(
        grid1,
        path,
        explored,
        "testcase1.png"
    )

    grid2 = [
        list("S...."),
        list("#####"),
        list("....G"),
        list("....."),
        list(".....")
    ]

    path, explored, cost, execution_time = astar(grid2)

    print_results(
        "TEST CASE 2 - NO PATH",
        grid2,
        path,
        explored,
        cost,
        execution_time
    )

    visualize(
        grid2,
        path,
        explored,
        "testcase2.png"
    )


    grid3 = [
        list("S........."),
        list(".#####...."),
        list(".....#...."),
        list(".....#...."),
        list("..####...."),
        list(".........."),
        list(".......G..")
    ]

    path, explored, cost, execution_time = astar(grid3)

    print_results(
        "TEST CASE 3 - LARGER GRID",
        grid3,
        path,
        explored,
        cost,
        execution_time
    )

    visualize(
        grid3,
        path,
        explored,
        "testcase3.png"
    )

if __name__ == "__main__":
    main()