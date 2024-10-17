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
    LocalPath(const std::vector<std::tuple<double, double, double>>& global_path, const Pose& vehicle_pose)
        : global_path_(global_path), vehicle_pose_(vehicle_pose), vehicle_index_(0)
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
        const int search_window = 50;  // Limit the search to the next 50 points
        double min_distance = std::numeric_limits<double>::max();

        for (size_t i = vehicle_index_; i < std::min(vehicle_index_ + search_window, global_path_.size()); ++i)
        {
            const double gx = std::get<0>(global_path_[i]);
            const double gy = std::get<1>(global_path_[i]);
            const double dx = gx - vehicle_pose_.x;
            const double dy = gy - vehicle_pose_.y;

            double distance = sqrt(dx * dx + dy * dy);

            // Ensure the waypoint is ahead of the vehicle
            if (distance < min_distance && dx * cos(vehicle_pose_.theta) + dy * sin(vehicle_pose_.theta) > 0)
            {
                min_distance = distance;
                vehicle_index_ = i;
            }
        }
    }

    // Convert the global path to a local path in the vehicle's coordinate system, taking a number of poses ahead of the vehicle
    std::vector<std::tuple<double, double, double>> getLocalPathAhead(int num_poses_ahead)
    {
        std::vector<std::tuple<double, double, double>> local_path;
        local_path.emplace_back(0, 0, 0); // Ego-point (0, 0, 0)

        findClosestWaypointAhead();

        for (size_t i = vehicle_index_; i < vehicle_index_ + num_poses_ahead && i < global_path_.size(); ++i)
        {
            const double gx = std::get<0>(global_path_[i]);
            const double gy = std::get<1>(global_path_[i]);
            const double gtheta = std::get<2>(global_path_[i]);

            const double dx = gx - vehicle_pose_.x;
            const double dy = gy - vehicle_pose_.y;

            // Convert global coordinates to local coordinates
            double local_x = dx * cos(-vehicle_pose_.theta) - dy * sin(-vehicle_pose_.theta);
            double local_y = dx * sin(-vehicle_pose_.theta) + dy * cos(-vehicle_pose_.theta);
            double local_theta = normalizeAngle(gtheta - vehicle_pose_.theta);

            local_path.emplace_back(local_x, local_y, local_theta);
        }

        // If not enough points, add the last point repeatedly to match the required number of points
        while (local_path.size() < static_cast<size_t>(num_poses_ahead + 1))
        {
            // +1 because of ego-point
            local_path.push_back(local_path.back());
        }


        return local_path;
    }

    // Get the global path points ahead of the vehicle
    std::vector<std::tuple<double, double, double>> getGlobalPathAhead(int num_poses_ahead)
    {
        std::vector<std::tuple<double, double, double>> global_path_ahead;
        global_path_ahead.emplace_back(vehicle_pose_.x, vehicle_pose_.y, vehicle_pose_.theta);

        findClosestWaypointAhead();

        for (size_t i = vehicle_index_; i < vehicle_index_ + num_poses_ahead && i < global_path_.size(); ++i)
        {
            global_path_ahead.push_back(global_path_[i]);
        }
        // If not enough points, add the last point to match the required number of points
        while (global_path_ahead.size() < static_cast<size_t>(num_poses_ahead + 1))
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
    static Eigen::VectorXd fitPolynomial(const std::vector<std::tuple<double, double>>& waypoints, int order = 3)
    {
        size_t n = waypoints.size();
        if (n < order + 1)
        {
            std::cerr << "Not enough points to fit a polynomial of order " << order << "." << std::endl;
            return Eigen::VectorXd::Zero(order + 1);
        }

        Eigen::MatrixXd A(n, order + 1);
        Eigen::VectorXd b(n);

        for (auto i = 0; i < n; ++i)
        {
            double x = std::get<0>(waypoints[i]);
            double y = std::get<1>(waypoints[i]);
            for (auto j = 0; j < order + 1; ++j)
            {
                A(i, j) = pow(x, j);
            }
            b(i) = y;
        }

        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);
        return coeffs;
    }

    // Generate points with heading angle (theta)
    static std::vector<std::tuple<double, double, double>>
    generatePointsWithHeading(const Eigen::VectorXd& coeffs, double start_x, int num_points, double step)
    {
        std::vector<std::tuple<double, double, double>> points_with_heading;
        double x = start_x;
        for (int i = 0; i < num_points; ++i)
        {
            double y = evaluatePolynomial(coeffs, x);
            double dy_dx = evaluateDerivative(coeffs, x);
            double theta = atan(dy_dx); // Heading angle in radians

            points_with_heading.emplace_back(x, y, theta);
            x += step;
        }
        return points_with_heading;
    }

private:
    std::vector<std::tuple<double, double, double>> global_path_; // Global path (x, y, theta)
    Pose vehicle_pose_; // Vehicle's pose (x, y, theta)
    size_t vehicle_index_; // Index of the closest waypoint ahead of the vehicle

    static double normalizeAngle(double angle)
    {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    static double evaluatePolynomial(const Eigen::VectorXd& coeffs, const double x)
    {
        double y = 0.0;
        for (int i = 0; i < coeffs.size(); ++i)
        {
            y += coeffs[i] * pow(x, i);
        }
        return y;
    }

    static double evaluateDerivative(const Eigen::VectorXd& coeffs, const double x)
    {
        double dy_dx = 0.0;
        for (int i = 1; i < coeffs.size(); ++i)
        {
            dy_dx += i * coeffs[i] * pow(x, i - 1);
        }
        return dy_dx;
    }
};

int main()
{
    // Step 1: Read global path from file
    std::ifstream file("/home/dinhnambkhn/Documents/A_star_matplotlib_cpp/path.txt");
    std::vector<std::tuple<double, double, double>> global_path;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        double time, x, y, z, qx, qy, qz, qw, gear;
        if (!(iss >> time >> x >> y >> z >> qx >> qy >> qz >> qw >> gear)) break;
        double yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
        global_path.emplace_back(x, y, yaw);
    }

    // Step 2: Initialize vehicle pose at a starting point
    LocalPath::Pose vehicle_pose{std::get<0>(global_path[0]), std::get<1>(global_path[0]), std::get<2>(global_path[0])};
    LocalPath local_path(global_path, vehicle_pose);

    // Step 3: Animate vehicle movement along the path
    for (size_t i = 0; i < global_path.size() - 20; ++i)
    {
        plt::clf(); // Clear previous plot

        // Update vehicle pose to simulate movement along the path
        vehicle_pose.x = std::get<0>(global_path[i]);
        vehicle_pose.y = std::get<1>(global_path[i]);
        vehicle_pose.theta = std::get<2>(global_path[i]);
        local_path.updateVehiclePose(vehicle_pose);

        // Get global path ahead of the vehicle
        auto global_path_ahead = local_path.getGlobalPathAhead(20);
        auto local_path_ahead = local_path.getLocalPathAhead(20);

        //fit 5-order polynomial to local_path_ahead
        std::vector<std::tuple<double, double>> waypoints;
        for (const auto& point : local_path_ahead)
        {
            waypoints.emplace_back(std::get<0>(point), std::get<1>(point));
        }
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd coeffs = LocalPath::fitPolynomial(waypoints, 5);
        //time end
        auto end = std::chrono::high_resolution_clock::now();
        // time ms
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Time taken for getLocalPathAhead: " << elapsed.count() << " ms" << std::endl;

        auto points_with_heading = LocalPath::generatePointsWithHeading(coeffs, 0, 15, 0.3);

        // Convert local path back to global path
        auto global_path_converted = local_path.convertLocalToGlobal(points_with_heading);



        // Plot the global_path_converted
        std::vector<double> x_converted, y_converted;
        for (const auto& point : global_path_converted)
        {
            x_converted.push_back(std::get<0>(point));
            y_converted.push_back(std::get<1>(point));
        }
        plt::plot(x_converted, y_converted, "rs"); // Red squares for converted global path

        // Plot global path
        std::vector<double> x_global, y_global;
        for (const auto& point : global_path)
        {
            x_global.push_back(std::get<0>(point));
            y_global.push_back(std::get<1>(point));
        }
        plt::plot(x_global, y_global, "b*"); // Blue dashed line for global path

        //plot the global path ahead
        std::vector<double> x_global_ahead, y_global_ahead;
        for (const auto& point : global_path_ahead)
        {
            x_global_ahead.push_back(std::get<0>(point));
            y_global_ahead.push_back(std::get<1>(point));
        }
        plt::plot(x_global_ahead, y_global_ahead, "r--"); // Red dashed line for global path ahead

        // Plot current vehicle position
        plt::plot({vehicle_pose.x}, {vehicle_pose.y}, "yo" ); // Yellow circle for vehicle

        // Set plot labels and titles
        plt::title("Vehicle Movement Along Path");
        plt::xlabel("X");
        plt::ylabel("Y");
        plt::grid(true);

        plt::pause(0.15); // Pause for animation effect (adjust for real-time movement)
    }

    plt::show(); // Final plot
    return 0;
}
