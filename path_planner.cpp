//
// Created by dinhnambkhn on 14/10/2024.
//
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <sstream>
#include <Eigen/Geometry>
#include <chrono>

#include "matplotlibcpp.h"

using namespace matplotlibcpp;

class LocalPath {
public:
    // Vehicle pose (x, y, theta)
    struct Pose {
        double x, y, theta;
    };

    // Constructor
    LocalPath(const std::vector<std::tuple<double, double, double>> &global_path, const Pose& vehicle_pose)
            : global_path_(global_path), vehicle_pose_(vehicle_pose), vehicle_index_(0) {}

    // Update vehicle pose
    void updateVehiclePose(const Pose &new_pose) {
        vehicle_pose_ = new_pose;
    }

    // Find the index of the closest waypoint ahead of the vehicle in the global path
    void findClosestWaypointAhead() {
        double min_distance = std::numeric_limits<double>::max();
        for (size_t i = vehicle_index_; i < global_path_.size(); ++i) {
            double gx = std::get<0>(global_path_[i]);
            double gy = std::get<1>(global_path_[i]);
            double dx = gx - vehicle_pose_.x;
            double dy = gy - vehicle_pose_.y;

            // Calculate the distance from the vehicle pose to the waypoint
            double distance = sqrt(dx * dx + dy * dy);

            // Find the first waypoint ahead of the vehicle (x > 0 in vehicle frame)
            if (distance < min_distance && dx * cos(vehicle_pose_.theta) + dy * sin(vehicle_pose_.theta) > 0) {
                min_distance = distance;
                vehicle_index_ = i;
            }
        }
    }

    // Convert the global path to a local path in the vehicle's coordinate system, taking a number of poses ahead of the vehicle
    std::vector<std::tuple<double, double, double>> getLocalPathAhead(int num_poses_ahead) {
        std::vector<std::tuple<double, double, double>> local_path;
        local_path.emplace_back(0, 0, 0);  // Ego-point (0, 0, 0)

        findClosestWaypointAhead();

        for (size_t i = vehicle_index_; i < vehicle_index_ + num_poses_ahead && i < global_path_.size(); ++i) {
            double gx = std::get<0>(global_path_[i]);
            double gy = std::get<1>(global_path_[i]);
            double gtheta = std::get<2>(global_path_[i]);

            double dx = gx - vehicle_pose_.x;
            double dy = gy - vehicle_pose_.y;

            // Convert global coordinates to local coordinates
            double local_x = dx * cos(-vehicle_pose_.theta) - dy * sin(-vehicle_pose_.theta);
            double local_y = dx * sin(-vehicle_pose_.theta) + dy * cos(-vehicle_pose_.theta);
            double local_theta = normalizeAngle(gtheta - vehicle_pose_.theta);

            local_path.emplace_back(local_x, local_y, local_theta);
        }

        // If not enough points, add the last point repeatedly to match the required number of points
        while (local_path.size() < static_cast<size_t>(num_poses_ahead + 1)) {  // +1 because of ego-point
            local_path.push_back(local_path.back());
        }


        return local_path;
    }

    // Get the global path points ahead of the vehicle
    std::vector<std::tuple<double, double, double>> getGlobalPathAhead(int num_poses_ahead) {
        std::vector<std::tuple<double, double, double>> global_path_ahead;
        global_path_ahead.emplace_back(vehicle_pose_.x, vehicle_pose_.y, vehicle_pose_.theta);

        findClosestWaypointAhead();

        for (size_t i = vehicle_index_; i < vehicle_index_ + num_poses_ahead && i < global_path_.size(); ++i) {
            global_path_ahead.push_back(global_path_[i]);
        }
        // If not enough points, add the last point repeatedly to match the required number of points
        while (global_path_ahead.size() < static_cast<size_t>(num_poses_ahead + 1)) {  // +1 because of ego-point
            global_path_ahead.push_back(global_path_ahead.back());
        }

        return global_path_ahead;
    }

    // Convert the local path back to global coordinates using the vehicle's pose
    [[nodiscard]] std::vector<std::tuple<double, double, double>>
    convertLocalToGlobal(const std::vector<std::tuple<double, double, double>> &local_path) const {
        std::vector<std::tuple<double, double, double>> global_path;

        for (const auto &point : local_path) {
            double local_x = std::get<0>(point);
            double local_y = std::get<1>(point);
            double local_theta = std::get<2>(point);

            double global_x = vehicle_pose_.x + local_x * cos(vehicle_pose_.theta) - local_y * sin(vehicle_pose_.theta);
            double global_y = vehicle_pose_.y + local_x * sin(vehicle_pose_.theta) + local_y * cos(vehicle_pose_.theta);
            double global_theta = normalizeAngle(local_theta + vehicle_pose_.theta);

            global_path.emplace_back(global_x, global_y, global_theta);
        }

        return global_path;
    }

    // Fit a polynomial to a set of waypoints
    static Eigen::VectorXd fitPolynomial(const std::vector<std::tuple<double, double>> &waypoints, int order = 3) {
        size_t n = waypoints.size();
        if (n < order + 1) {
            std::cerr << "Not enough points to fit a polynomial of order " << order << "." << std::endl;
            return Eigen::VectorXd::Zero(order + 1);
        }

        Eigen::MatrixXd A(n, order + 1);
        Eigen::VectorXd b(n);

        for (size_t i = 0; i < n; ++i) {
            double x = std::get<0>(waypoints[i]);
            double y = std::get<1>(waypoints[i]);
            for (int j = 0; j < order + 1; ++j) {
                A(i, j) = pow(x, j);
            }
            b(i) = y;
        }

        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);
        return coeffs;
    }

    // Generate points with heading angle (theta)
    static std::vector<std::tuple<double, double, double>>
    generatePointsWithHeading(const Eigen::VectorXd &coeffs, double start_x, int num_points, double step) {
        std::vector<std::tuple<double, double, double>> points_with_heading;
        double x = start_x;
        for (int i = 0; i < num_points; ++i) {
            double y = evaluatePolynomial(coeffs, x);
            double dy_dx = evaluateDerivative(coeffs, x);
            double theta = atan(dy_dx);  // Heading angle in radians

            points_with_heading.emplace_back(x, y, theta);
            x += step;
        }
        return points_with_heading;
    }

private:
    std::vector<std::tuple<double, double, double>> global_path_;  // Global path (x, y, theta)
    Pose vehicle_pose_;  // Vehicle's pose (x, y, theta)
    size_t vehicle_index_;  // Index of the closest waypoint ahead of the vehicle

    static double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    static double evaluatePolynomial(const Eigen::VectorXd &coeffs, double x) {
        double y = 0.0;
        for (int i = 0; i < coeffs.size(); ++i) {
            y += coeffs[i] * pow(x, i);
        }
        return y;
    }

    static double evaluateDerivative(const Eigen::VectorXd &coeffs, double x) {
        double dy_dx = 0.0;
        for (int i = 1; i < coeffs.size(); ++i) {
            dy_dx += i * coeffs[i] * pow(x, i - 1);
        }
        return dy_dx;
    }
};

// Example usage
int main() {
    std::ifstream file("/home/dinhnambkhn/A_star_matplotlib_cpp/path.txt");
    std::vector<std::tuple<double, double, double>> global_path;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double time, x, y, z, qx, qy, qz, qw, gear;
        if (!(iss >> time >> x >> y >> z >> qx >> qy >> qz >> qw >> gear)) break;
        double yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
        global_path.emplace_back(x, y, yaw);
    }

    LocalPath::Pose vehicle_pose{std::get<0>(global_path[60]), std::get<1>(global_path[60]), std::get<2>(global_path[60])};
    LocalPath local_path(global_path, vehicle_pose);

    auto global_path_ahead = local_path.getGlobalPathAhead(20);
    //time start
    auto local_path_ahead = local_path.getLocalPathAhead(20);

    for (const auto &point : global_path_ahead) {
        std::cout << "Global Path: x = " << std::get<0>(point) << ", y = " << std::get<1>(point)
                  << ", theta = " << std::get<2>(point) << std::endl;
    }

    for (const auto &point : local_path_ahead) {
        std::cout << "Local Path: x = " << std::get<0>(point) << ", y = " << std::get<1>(point)
                  << ", theta = " << std::get<2>(point) << std::endl;
    }

    //fit 5-order polynomial to local_path_ahead
    std::vector<std::tuple<double, double>> waypoints;
    for (const auto &point : local_path_ahead) {
        waypoints.emplace_back(std::get<0>(point), std::get<1>(point));
    }
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd coeffs = LocalPath::fitPolynomial(waypoints, 5);
    //time end
    auto end = std::chrono::high_resolution_clock::now();
    // time ms
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time taken for getLocalPathAhead: " << elapsed.count() << " ms" << std::endl;


    std::cout << "Fitted Polynomial Coefficients:\n";
    for (int i = 0; i < coeffs.size(); ++i) {
        std::cout << "a" << i << " = " << coeffs[i] << std::endl;
    }
    // Generate points with heading angles

    auto points_with_heading = LocalPath::generatePointsWithHeading(coeffs, 0, 20, 0.5);


    std::cout << "\nFitted Points with Heading Angles:\n";
    for (const auto &point : points_with_heading) {
        std::cout << "x: " << std::get<0>(point) << ", y: " << std::get<1>(point)
                  << ", theta: " << std::get<2>(point) * 180 / M_PI << " deg"
                  << std::endl; // Convert radians to degrees for readability
    }
    // Convert local path back to global path
    auto global_path_converted = local_path.convertLocalToGlobal(points_with_heading);
    for (const auto &point : global_path_converted) {
        std::cout << "Global Path Converted: x = " << std::get<0>(point) << ", y = " << std::get<1>(point)
                  << ", theta = " << std::get<2>(point) << std::endl;
    }

    // Plot the global path
    std::vector<double> x, y;
    for (const auto &point : global_path) {
        x.push_back(std::get<0>(point));
        y.push_back(std::get<1>(point));
    }
    plot(x, y, "b*");

    // Plot the global path ahead
    std::vector<double> x_ahead, y_ahead;
    for (const auto &point : global_path_ahead) {
        x_ahead.push_back(std::get<0>(point));
        y_ahead.push_back(std::get<1>(point));
    }
    plot(x_ahead, y_ahead, "r*");

    // Plot vehicle pose at the start - square marker yellow-vehicle_pose
    plot({vehicle_pose.x}, {vehicle_pose.y}, "ys");


    // Plot the global_path_converted
    std::vector<double> x_converted, y_converted;
    for (const auto &point : global_path_converted) {
        x_converted.push_back(std::get<0>(point));
        y_converted.push_back(std::get<1>(point));
    }
    plot(x_converted, y_converted, "g*");

    show();



    return 0;
}