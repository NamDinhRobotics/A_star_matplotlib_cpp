#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

#include "matplotlibcpp.h"

using namespace std;

struct Point {
    int x, y;

    Point(int _x, int _y) : x(_x), y(_y) {}
};


float euclideanDistance(const Point &a, const Point &b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

struct Node {
    Point pos;
    float g, h;
    Node *parent;

    Node(Point _pos, float _g, float _h, Node *_parent = nullptr) : pos(_pos), g(_g), h(_h), parent(_parent) {}

    [[nodiscard]] float getF() const {
        return g + h;
    }
};

// Function to find a path from start to goal on the grid,
vector<Point> findPath(Point start, Point goal, vector<vector<int>> &grid) {
    // Priority queue to traverse nodes based on the lowest cost
    auto cmp = [](Node *left, Node *right) { return left->getF() > right->getF(); };
    priority_queue<Node *, vector<Node *>, decltype(cmp)> openSet(cmp);

    // Create the initial node with cost g = 0 and estimated cost h from start to goal
    Node *startNode = new Node(start, 0.0, euclideanDistance(start, goal));
    openSet.push(startNode);

    // List to store visited nodes
    vector<Node *> closedSet;

    while (!openSet.empty()) {
        // Retrieve the node with the lowest total cost from the priority queue
        Node *currentNode = openSet.top();
        openSet.pop();// Remove the node from the open set that has been selected

        // Check if the current node is the goal, if so, return the path
        if (currentNode->pos.x == goal.x && currentNode->pos.y == goal.y) {
            vector<Point> path;
            // Trace back the path from goal to start based on the parent pointers
            while (currentNode != nullptr) {
                path.push_back(currentNode->pos);
                currentNode = currentNode->parent;
            }
            reverse(path.begin(), path.end()); // Reverse to get the path from start to goal
            return path;
        }

        // Add the current node to the visited set
        closedSet.push_back(currentNode);

        // List neighboring points
        vector<Point> neighbors = {
                {currentNode->pos.x + 1, currentNode->pos.y},
                {currentNode->pos.x,     currentNode->pos.y + 1},
                {currentNode->pos.x - 1, currentNode->pos.y},
                {currentNode->pos.x,     currentNode->pos.y - 1}
        };

        for (const Point &neighbor: neighbors) {
            // Check if the neighbor point is within the grid and not an obstacle
            if (neighbor.x >= 0 && neighbor.x < grid.size() &&
                neighbor.y >= 0 && neighbor.y < grid[0].size() &&
                grid[neighbor.x][neighbor.y] != 1) {

                float newG = currentNode->g + euclideanDistance(currentNode->pos, neighbor);

                Node *neighborNode = new Node(neighbor, newG, euclideanDistance(neighbor, goal), currentNode);

                // Check if the neighbor node has been visited
                auto it = find_if(closedSet.begin(), closedSet.end(), [&](const Node *n) {
                    return n->pos.x == neighborNode->pos.x && n->pos.y == neighborNode->pos.y;
                });

                // If the node has not been visited, add it to the open set
                if (it == closedSet.end()) {
                    openSet.push(neighborNode);
                } else if (newG < (*it)->g) { // If visited and a better path is found, update the cost and parent
                    (*it)->g = newG;
                    (*it)->parent = currentNode;
                }
            }
        }
    }

    return {}; // Return an empty vector if no path is found
}

namespace plt = matplotlibcpp;

// Main function to execute the program
int main() {
    // Initialize a 10x10 grid and start and goal points
    vector<vector<int>> grid = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 1, 1, 1, 1, 1, 0},
            {0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 1, 1, 1, 1},
            {0, 0, 0, 0, 1, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 1, 0, 1, 0},
            {0, 1, 1, 1, 1, 0, 1, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    Point start = {0, 0};
    Point goal = {5, 5};

    // Find a path from start to goal on the given grid
    vector<Point> path = findPath(start, goal, grid);

    // Print the found path
    if (!path.empty()) {
        for (const auto &point: path) {
            cout << "(" << point.x << "," << point.y << ") ";
        }
        cout << endl;
    } else {
        cout << "No path found!" << endl;
    }
    //create a grid copy from the original grid
    vector<vector<int>> grid_copy = grid;
    //chance the grid copy to - if the path on the grid
    for (const auto &point: path) {
        grid_copy[point.x][point.y] = 2;
    }
    //print the grid copy
    for (const auto &row: grid_copy) {
        for (const auto &col: row) {
            //if = 2 prints > else print the number
            if (col == 2) {
                cout << "* ";
            } else {
                cout << col << " ";
            }

        }
        cout << endl;
    }

    // Plot the grid using matplotlibcpp
    plt::title("A* Path Finding");
    int size = grid.size();
    //create a plot to show the grid
    plt::xlim(-1, size + 1);
    plt::ylim(-1, size + 1);
    //convert grid to coordinates
    vector<double> x, y;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < grid[i].size(); j++) {
            x.push_back(i);
            y.push_back(j);
        }
    }
    //plot the grid back if = 1 else green
    for (int i = 0; i < x.size(); i++) {
        if (grid[x[i]][y[i]] == 1) {
            plt::plot({x[i]}, {y[i]}, "ks");
        } else {
            plt::plot({x[i]}, {y[i]}, "gs");
        }
    }

    //convert the path to coordinates
    vector<double> path_x, path_y;
    for (const auto &point: path) {
        path_x.push_back(point.x);
        path_y.push_back(point.y);
    }
    //plot the path with arrows
    plt::plot(path_x, path_y, "b-");
    //plt::quiver(path_x, path_y, path_x, path_y);
    //start and goal points
    double start_x = start.x;
    double start_y = start.y;
    double goal_x = goal.x;
    double goal_y = goal.y;
    plt::plot({start_x}, {start_y}, "go");
    plt::plot({goal_x}, {goal_y}, "ro");
    //text for start and goal
    plt::text(start_x, start_y, "Start");
    plt::text(goal_x, goal_y, "Goal");




    //plot the path
    plt::show();


    return 0;
}