#include <Eigen/Core>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>

void get_file_contents(std::ostringstream& contents, const char *filename) {
	std::ifstream in(filename);
	if (in) {
		contents << in.rdbuf();
		in.close();
	} else
		throw(errno);
}

int main()
{
	// create a compute context and command queue
	auto ctx = boost::compute::system::default_context();
	auto queue = boost::compute::system::default_queue();

	// create program and kernels
	std::ostringstream source;
	get_file_contents(source, "svd3.cl");
	auto program = boost::compute::program::build_with_source(source.str(), ctx);
	auto svdArrayTestKernel = program.create_kernel("svdArrayTest");

	EigenMatN A_host = EigenMatN::Identity();
	boost::compute::vector<cl_float> A_dev(A_host.data(), A_host.data() + N_*N_, queue);
	boost::compute::vector<cl_float> U_dev(N_*N_), S_dev(N_*N_), V_dev(N_*N_);

	svdArrayTestKernel.set_args(A_dev, U_dev, S_dev, V_dev);
	queue.enqueue_1d_range_kernel(svdArrayTestKernel, 0, 1, 0);

	EigenMatN U_host,S_host,V_host;
	boost::compute::copy(U_dev.begin(), U_dev.end(), U_host.data());
	boost::compute::copy(S_dev.begin(), S_dev.end(), S_host.data());
	boost::compute::copy(V_dev.begin(), V_dev.end(), V_host.data());

	std::cout << A_host << std::endl;
	std::cout << U_host << std::endl;
	std::cout << S_host << std::endl;
	std::cout << V_host << std::endl;
	
    return 0;
}

