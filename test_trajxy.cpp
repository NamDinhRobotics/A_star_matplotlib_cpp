//
// Created by dinhnambkhn on 17/10/2024.
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

namespace plt = matplotlibcpp;


class LocalPath
{
public:
    // Vehicle pose (x, y, theta)
    struct Pose
    {
        double x, y, theta;
    };

    // Constructor
    LocalPath(const std::vector<std::tuple<double, double, double>>& global_path, const Pose& vehicle_pose, const int order = 3)
        : global_path_(global_path), vehicle_pose_(vehicle_pose), vehicle_index_(0), order_(order)
    {
    }

    // Update vehicle pose
    void updateVehiclePose(const Pose& new_pose)
    {
        vehicle_pose_ = new_pose;
    }

    // Limit the search window to reduce redundant searches for waypoints
    void findClosestWaypointAhead()
    {
        constexpr int search_window = 50; // Limit the search to the next 50 points
        double min_distance = std::numeric_limits<double>::max();

        for (size_t i = vehicle_index_; i < std::min(vehicle_index_ + search_window, global_path_.size()); ++i)
        {
            const double gx = std::get<0>(global_path_[i]);
            const double gy = std::get<1>(global_path_[i]);
            const double dx = gx - vehicle_pose_.x;
            const double dy = gy - vehicle_pose_.y;

            // Ensure the waypoint is ahead of the vehicle
            const double dis = sqrt(dx * dx + dy * dy);
            if (dis < min_distance && dx * cos(vehicle_pose_.theta) + dy *
                sin(vehicle_pose_.theta) > 0)
            {
                min_distance = dis;
                vehicle_index_ = i;
            }
        }
    }

    // Convert the global path to a local path in the vehicle's coordinate system, taking a number of poses ahead of the vehicle
    std::vector<std::tuple<double, double, double>> getLocalPathAhead(const int num_poses_ahead)
    {
        std::vector<std::tuple<double, double, double>> local_path;
        local_path.emplace_back(0, 0, 0); // Ego-point (0, 0, 0)

        findClosestWaypointAhead();

        for (size_t i = vehicle_index_; i < vehicle_index_ + num_poses_ahead && i < global_path_.size(); ++i)
        {
            const double gx = std::get<0>(global_path_[i]);
            const double gy = std::get<1>(global_path_[i]);
            const double gTheta = std::get<2>(global_path_[i]);

            const double dx = gx - vehicle_pose_.x;
            const double dy = gy - vehicle_pose_.y;

            // Convert global coordinates to local coordinates
            double local_x = dx * cos(-vehicle_pose_.theta) - dy * sin(-vehicle_pose_.theta);
            double local_y = dx * sin(-vehicle_pose_.theta) + dy * cos(-vehicle_pose_.theta);
            double local_theta = normalizeAngle(gTheta - vehicle_pose_.theta);

            local_path.emplace_back(local_x, local_y, local_theta);
        }

        // If not enough points, add the last point repeatedly to match the required number of points
        while (local_path.size() < static_cast<size_t>(num_poses_ahead) + 1)
        {
            // +1 because of ego-point
            local_path.push_back(local_path.back());
        }


        return local_path;
    }

    // Get the global path points ahead of the vehicle
    std::vector<std::tuple<double, double, double>> getGlobalPathAhead(const int num_poses_ahead)
    {
        std::vector<std::tuple<double, double, double>> global_path_ahead;
        global_path_ahead.emplace_back(vehicle_pose_.x, vehicle_pose_.y, vehicle_pose_.theta);

        findClosestWaypointAhead();

        for (size_t i = vehicle_index_; i < vehicle_index_ + num_poses_ahead && i < global_path_.size(); ++i)
        {
            global_path_ahead.push_back(global_path_[i]);
        }
        // If not enough points, add the last point to match the required number of points
        while (global_path_ahead.size() < static_cast<size_t>(num_poses_ahead) + 1)
        {
            // +1 because of ego-point
            global_path_ahead.push_back(global_path_ahead.back());
        }

        return global_path_ahead;
    }

    // Convert the local path back to global coordinates using the vehicle's pose
    [[nodiscard]] std::vector<std::tuple<double, double, double>>
    convertLocalToGlobal(const std::vector<std::tuple<double, double, double>>& local_path) const
    {
        std::vector<std::tuple<double, double, double>> global_path;

        for (const auto& point : local_path)
        {
            const double local_x = std::get<0>(point);
            const double local_y = std::get<1>(point);
            const double local_theta = std::get<2>(point);

            double global_x = vehicle_pose_.x + local_x * cos(vehicle_pose_.theta) - local_y * sin(vehicle_pose_.theta);
            double global_y = vehicle_pose_.y + local_x * sin(vehicle_pose_.theta) + local_y * cos(vehicle_pose_.theta);
            double global_theta = normalizeAngle(local_theta + vehicle_pose_.theta);

            global_path.emplace_back(global_x, global_y, global_theta);
        }

        return global_path;
    }

    // Fit a polynomial to a set of waypoints
    [[nodiscard]] Eigen::VectorXd fitPolynomial(const std::vector<std::tuple<double, double>>& waypoints) const
    {
        const int order = this->order_;

        const size_t n = waypoints.size();
        if (n < order + 1)
        {
            std::cerr << "Not enough points to fit a polynomial of order " << order << "." << std::endl;
            return Eigen::VectorXd::Zero(order + 1);
        }

        Eigen::MatrixXd A(n, order + 1);
        Eigen::VectorXd b(n);

        for (auto i = 0; i < n; ++i)
        {
            const double x = std::get<0>(waypoints[i]);
            const double y = std::get<1>(waypoints[i]);
            for (auto j = 0; j < order + 1; ++j)
            {
                A(i, j) = pow(x, j);
            }
            b(i) = y;
        }

        Eigen::VectorXd coefficients = A.colPivHouseholderQr().solve(b);

        return coefficients;
    }

    // Generate points with heading angle (theta)
    static std::vector<std::tuple<double, double, double>>
    generatePointsWithHeading(const Eigen::VectorXd& coefficient, const double start_x, const int num_points, const double step)
    {
        std::vector<std::tuple<double, double, double>> points_with_heading;
        double x = start_x;
        for (int i = 0; i < num_points; ++i)
        {
            const double y = evaluatePolynomial(coefficient, x);
            const double dy_dx = evaluateDerivative(coefficient, x);
            const double theta = atan(dy_dx); // Heading angle in radians

            points_with_heading.emplace_back(x, y, theta);
            x += step;
        }
        //chance the first point to ego point (0,0,0)
        points_with_heading[0] = std::make_tuple(0, 0, 0);
        return points_with_heading;
    }

    std::vector<std::tuple<double, double, double>> genLocalPathInter(const Pose& vehicle_pose,
                                                                      const int num_poses_ahead = 20,
                                                                      const double start_x = 0.0,
                                                                      const int num_points = 15,
                                                                      const double step = 0.3)
    {
        // Update the vehicle's pose
        updateVehiclePose(vehicle_pose);

        // Get the local path ahead of the vehicle using the specified number of poses
        auto local_path_ahead = getLocalPathAhead(num_poses_ahead);

        // Convert the local path to (x, y) points to fit the polynomial
        std::vector<std::tuple<double, double>> waypoints;
        for (const auto& point : local_path_ahead)
        {
            waypoints.emplace_back(std::get<0>(point), std::get<1>(point));
        }

        // Fit a polynomial to the waypoints (5th order)
        const Eigen::VectorXd coefficients = fitPolynomial(waypoints);

        // Generate points with heading using the internally computed coefficients
        return generatePointsWithHeading(coefficients, start_x, num_points, step);
    }

private:
    std::vector<std::tuple<double, double, double>> global_path_; // Global path (x, y, theta)
    Pose vehicle_pose_; // Vehicle's pose (x, y, theta)
    size_t vehicle_index_; // Index of the closest waypoint ahead of the vehicle
    int order_; // Polynomial order

    static double normalizeAngle(double angle)
    {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    static double evaluatePolynomial(const Eigen::VectorXd& coefficients, const double x)
    {
        double y = 0.0;
        for (int i = 0; i < coefficients.size(); ++i)
        {
            y += coefficients[i] * pow(x, i);
        }
        return y;
    }

    static double evaluateDerivative(const Eigen::VectorXd& coefficients, const double x)
    {
        double dy_dx = 0.0;
        for (int i = 1; i < coefficients.size(); ++i)
        {
            dy_dx += i * coefficients[i] * pow(x, i - 1);
        }
        return dy_dx;
    }
};

int main()
{
    // Step 1: Read global path from file
    std::ifstream file("/home/dinhnambkhn/Documents/A_star_matplotlib_cpp/trajectory0.csv");
    std::vector<std::tuple<double, double, double>> global_path;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return 1;
    }

    //std::vector<double> x_values, y_values;
    std::string line, header;

    // Read and ignore the header
    std::getline(file, header);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str;

        // Split the line by comma and extract x and y values
        std::getline(ss, x_str, ',');
        std::getline(ss, y_str, ',');

        // Convert the strings to double and store them in the vectors
        double x = std::stod(x_str);
        double y = std::stod(y_str);
        //global_path
        global_path.emplace_back(x, y, 0.0);

    }

    file.close();

    // Step 2: Initialize vehicle pose at a starting point
    LocalPath::Pose vehicle_pose{std::get<0>(global_path[0]), std::get<1>(global_path[0]), std::get<2>(global_path[0])};
    LocalPath local_path(global_path, vehicle_pose, 3); // 5th order polynomial

    // Step 3: Animate vehicle movement along the path using genLocalPathInter
    for (size_t i = 0; i < global_path.size() - 30; ++i)
    {
        plt::clf(); // Clear previous plot

        // Update vehicle pose to simulate movement along the path
        vehicle_pose.x = std::get<0>(global_path[i]);
        vehicle_pose.y = std::get<1>(global_path[i]);
        vehicle_pose.theta = std::get<2>(global_path[i]);

        // Use genLocalPathInter to get points with heading
        auto points_with_heading = local_path.genLocalPathInter(vehicle_pose, 30, 0, 15, 0.3);

        // Convert the generated local path back to global coordinates
        auto global_path_converted = local_path.convertLocalToGlobal(points_with_heading);

        // Plot the global_path_converted and arrows for heading
        std::vector<double> x_converted, y_converted;
        for (const auto& point : global_path_converted)
        {
            double x = std::get<0>(point);
            double y = std::get<1>(point);
            double theta = std::get<2>(point); // Heading angle in radians
            x_converted.push_back(x);
            y_converted.push_back(y);

            // Draw an arrow at each point to indicate heading
            double arrow_length = 1.5; // Adjust the arrow length as needed
            double arrow_dx = arrow_length * cos(theta);
            double arrow_dy = arrow_length * sin(theta);
            plt::arrow(x, y, arrow_dx, arrow_dy, "green");
        }
        plt::plot(x_converted, y_converted, "r*"); // Red squares for converted global path

        // Plot global path
        std::vector<double> x_global, y_global;
        for (const auto& point : global_path)
        {
            x_global.push_back(std::get<0>(point));
            y_global.push_back(std::get<1>(point));
        }
        plt::plot(x_global, y_global, "b."); // Blue dashed line for global path

        // Plot current vehicle position
        plt::plot({vehicle_pose.x}, {vehicle_pose.y}, "yo"); // Yellow circle for vehicle

        // Set plot labels and titles
        plt::title("Vehicle Movement Along Path with Heading Indication");
        plt::xlabel("X");
        plt::ylabel("Y");
        plt::grid(true);

        plt::pause(0.15); // Pause for animation effect (adjust for real-time movement)
    }

    plt::show(); // Final plot
    return 0;
}
