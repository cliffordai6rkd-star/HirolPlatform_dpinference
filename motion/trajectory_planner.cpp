#include <pybind11/pybind11.h>
#include "time_optimal_trajectory/Trajectory.h"
#include "time_optimal_trajectory/Path.h"

namespace py = pybind11;

Eigen::VectorXd l_to_e(const py::list &l){
    int len = py::len(l);
    Eigen::VectorXd ret(len);
    for(int k=0; k<len; ++k)
        ret(k) = l[k].cast<double>();
    return ret;
}

py::list e_to_l(const Eigen::VectorXd &in){
    int v_len = in.size();
    py::list ret;
    for(int k=0; k<v_len; ++k)
        ret.append(in(k));
    return ret;
}

std::list<Eigen::VectorXd> ll_to_le(const py::list &ll){
    int len = py::len(ll);
    std::list<Eigen::VectorXd> ret;
    ret.clear();
    for(int k=0; k<len; ++k)
        ret.push_back(l_to_e(ll[k]));
    return ret;
}

class TimeOptimalTrajectoryWrapper : public Trajectory
{
public:
    TimeOptimalTrajectoryWrapper(
        const py::list &path, double max_dev,
        const py::list &max_vel, const py::list &max_acc):Trajectory(
            Path(ll_to_le(path), max_dev), l_to_e(max_vel), l_to_e(max_acc)){};
    py::list getPosition(double t){
        return e_to_l(Trajectory::getPosition(t));
    };
    py::list getVelocity(double t){
        return e_to_l(Trajectory::getVelocity(t));
    };
};


PYBIND11_MODULE(trajectory_planner, m) {
    py::class_<TimeOptimalTrajectoryWrapper>(m, "TimeOptimalTrajectoryWrapper")
        .def(py::init<const py::list&, double, const py::list&, const py::list&>())
        .def("is_valid", &Trajectory::isValid, "Returns True if valid.")
        .def("get_position", &TimeOptimalTrajectoryWrapper::getPosition, "Returns position at t.")
        .def("get_velocity", &TimeOptimalTrajectoryWrapper::getVelocity, "Returns velocity at t.")
        .def("get_duration", &Trajectory::getDuration, "Returns the duration.");
}
