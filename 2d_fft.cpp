#include <afx.h>
#include <omp.h>
#include <amp.h>
#include <amp_math.h>
#include <immintrin.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

concurrency::accelerator get_gpu() {
	std::vector<concurrency::accelerator> vAcs = concurrency::accelerator::get_all();
	std::vector<concurrency::accelerator> vNoEmurate;
	for (auto &i : vAcs) {
		if (i.is_emulated == false) {
			vNoEmurate.push_back(i);
		}
	}
	for (auto &i : vNoEmurate) {
		std::wcout << i.get_description() << "\n";
	}
	return vNoEmurate.front();
}

std::unique_ptr<std::vector<std::vector<float>>> mat2array(const cv::Mat &src_img) {
	size_t row = src_img.rows;
	size_t col = src_img.cols;
	std::unique_ptr<std::vector<std::vector<float>>> img(new std::vector<std::vector<float>>);
	img->resize(row, std::vector<float>(col,0));
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < row; y++) {
		for (size_t x = 0; x != col; x++) {
			img->at(y).at(x) = src_img.at<uint8_t>(y, x);;
		}
	}
	return img;
}

std::unique_ptr<std::vector<std::vector<float>>> get_w(const int length, const concurrency::accelerator gpu){
	std::unique_ptr<std::vector<std::vector<float>>> w(new std::vector<std::vector<float>>);
	w->resize(2,std::vector<float>(length,0));
	const float inner = -2 * M_PI / length;
	concurrency::array<float, 1> gpu_real(length, w->at(0).begin(), w->at(0).end());
	concurrency::array<float, 1> gpu_imag(length, w->at(1).begin(), w->at(1).end());
	concurrency::parallel_for_each(
		gpu.get_default_view(),
		gpu_real.extent,
		[=, &gpu_real, &gpu_imag](concurrency::index<1> idx)restrict(amp) {
		auto i = inner*idx[0];
		gpu_real[idx] = concurrency::fast_math::cosf(i);
		gpu_imag[idx] = concurrency::fast_math::sinf(i);
	}
	);
	w->at(0) = gpu_real;
	w->at(1) = gpu_imag;
	return w;
}

int bit_r(const int x, const int lengh){
	int x_r = 0;
	int log2_length = log2(lengh);
	for (int i = 0; i < log2_length; i++) {
		((x &(0b1 << i)) != 0) ? x_r += (1 << ((log2_length - 1) - i)) : 0;
	}
	return x_r;
}

int bit_r_gpu(const int x, const int lengh) restrict(amp) {
	using namespace concurrency::fast_math;
	int x_r = 0;
	int log2_length = log2(lengh);
	for (int i = 0; i < log2_length; i++) {
		((x &(0b1 << i)) != 0) ? x_r += (1 << ((log2_length - 1) - i)) : 0;
	}
	return x_r;
}

std::unique_ptr<std::vector<std::vector<std::vector<float>>>> fft(const cv::Mat &src_img, concurrency::accelerator gpu) {
	int row = src_img.rows;
	int col = src_img.cols;
	std::unique_ptr<std::vector<std::vector<std::vector<float>>>> fft_img(new std::vector<std::vector<std::vector<float>>>);
	fft_img->resize(2, std::vector<std::vector<float>>(row, std::vector<float>(col, 0)));
	std::vector<std::vector<std::vector<float>>, std::vector<std::vector<float>>> inner_fft(2,std::vector<std::vector<float>>(col, std::vector<float>(row, 0)));
	auto src_mat = std::move(mat2array(src_img));
	auto w_x = get_w(col, gpu);
	auto w_y = get_w(row, gpu);
#ifdef _OPENMP
#pragma omp parallel
#endif
	{
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for
#endif
		for (int y_2 = 0; y_2 < (int)row; y_2++) {
			int half = col / 2;
			std::vector<float>real(src_mat->at(y_2));
			std::vector<float>imag(col, 0);
			for (int i = 1; i < col; i = i << 1) {
				auto half_2 = half << 1;
				for (int j = 0; j < col; j += half_2) {
					int m = 0;
					for (int k = j; k < j + half; k++) {
						int a = k;
						int b = k + half;
						auto tmp_real = real.at(a) + real.at(b);
						auto tmp_imag = imag.at(a) + imag.at(b);
						auto sub_real = (real.at(a) - real.at(b));
						auto sub_imag = (imag.at(a) - imag.at(b));
						real.at(a) = tmp_real;
						imag.at(a) = tmp_imag;
						real.at(b) = sub_real*w_x->at(0).at(m) - sub_imag*w_x->at(1).at(m);
						imag.at(b) = sub_real*w_x->at(1).at(m) + sub_imag*w_x->at(0).at(m);
						m = (m + i);
					}
				}
				half = half >> 1;
			}
			for (int x = 0; x < col; x++) {
				inner_fft.at(0).at(x).at(y_2) = real.at(bit_r(x, col));
				inner_fft.at(1).at(x).at(y_2) = imag.at(bit_r(x, col));
			}
		}
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for
#endif
		for (int x = 0; x < (int)col; x++) {
			int half = row / 2;
			for (int i = 1; i < row; i = i << 1) {
				auto half_2 = half << 1;
				for (int j = 0; j < row; j += half_2) {
					int m = 0;
					for (int k = j; k < j + half; k++) {
						int a = k;
						int b = k + half;
						auto tmp_real = inner_fft.at(0).at(x).at(a) + inner_fft.at(0).at(x).at(b);
						auto tmp_imag = inner_fft.at(1).at(x).at(a) + inner_fft.at(1).at(x).at(b);
						auto sub_real = (inner_fft.at(0).at(x).at(a) - inner_fft.at(0).at(x).at(b));
						auto sub_imag = (inner_fft.at(1).at(x).at(a) - inner_fft.at(1).at(x).at(b));
						inner_fft.at(0).at(x).at(a) = tmp_real;
						inner_fft.at(1).at(x).at(a) = sub_real*w_y->at(0).at(m) - sub_imag*w_y->at(1).at(m);
						inner_fft.at(0).at(x).at(b) = tmp_imag;
						inner_fft.at(1).at(x).at(b) = sub_real*w_y->at(1).at(m) + sub_imag*w_y->at(0).at(m);
						m = (m + i);
					}
				}
				half = half >> 1;
			}
			for (int y = 0; y < row; y++) {
				fft_img->at(0).at(y).at(x) = inner_fft.at(0).at(x).at(bit_r(y, row));
				fft_img->at(1).at(y).at(x) = inner_fft.at(1).at(x).at(bit_r(y, row));
			}
		}
	}
	return fft_img;
}

std::unique_ptr<std::vector<std::vector<std::vector<float>>>> fft_gpu(const cv::Mat &src_img, const concurrency::accelerator gpu) {
	const int row = src_img.rows;
	const int col = src_img.cols;
	std::unique_ptr<std::vector<std::vector<std::vector<float>>>> fft_img(new std::vector<std::vector<std::vector<float>>>);
	*fft_img = { *std::move(mat2array(src_img)), std::vector<std::vector<float>>(row, std::vector<float>(col, 0)) };
	std::vector<float> for_gpu_real(row*col, 0), for_gpu_imag(row*col, 0);
	std::unique_ptr<std::vector<std::vector<float>>> w_x(get_w(col, gpu));
	std::unique_ptr<std::vector<std::vector<float>>> w_y(get_w(row, gpu));
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < row; i++) {
		std::copy(fft_img->at(0).at(i).begin(), fft_img->at(0).at(i).end(), &for_gpu_real.at(i*col));
	}
	concurrency::array<float, 2> gpu_real(row, col, for_gpu_real.begin(), for_gpu_real.end());
	concurrency::array<float, 2> gpu_imag(row, col, for_gpu_imag.begin(), for_gpu_imag.end());
	concurrency::array<float, 2> gpu_inner_real(col, row), gpu_inner_imag(col, row);
	concurrency::array<int, 1> gpu_bit_r_col(col), gpu_bit_r_row(row);
	concurrency::array<float, 1> gpu_w_x_real(col, w_x->at(0).begin(), w_x->at(0).end());
	concurrency::array<float, 1> gpu_w_x_imag(col, w_x->at(1).begin(), w_x->at(1).end());
	concurrency::array<float, 1> gpu_w_y_real(row, w_y->at(0).begin(), w_y->at(0).end());
	concurrency::array<float, 1> gpu_w_y_imag(row, w_y->at(1).begin(), w_y->at(1).end());
	{
		int half = col >> 1;
		concurrency::extent<2> iter;
		iter = { row,half };
		for (int i = 1; i < col; i = i << 1) {
			auto half_2 = half << 1;
			concurrency::parallel_for_each(
				gpu.get_default_view(),
				iter,
				[=, &gpu_w_x_real, &gpu_w_x_imag, &gpu_real, &gpu_imag](concurrency::index<2> idx)restrict(amp)
			{
				using namespace concurrency::fast_math;
				int j = fmodf(idx[1], half);
				concurrency::index<2> a = { idx[0],j + (int)floorf(idx[1] / half)*half_2 };
				concurrency::index<2> b = { a[0],a[1] + half };
				auto m = j * i;
				auto add_real = gpu_real[a] + gpu_real[b];
				auto add_imag = gpu_imag[a] + gpu_imag[b];
				auto sub_real = (gpu_real[a] - gpu_real[b]);
				auto sub_imag = (gpu_imag[a] - gpu_imag[b]);
				gpu_real[a] = add_real;
				gpu_imag[a] = add_imag;
				gpu_real[b] = sub_real*gpu_w_x_real[m] - sub_imag*gpu_w_x_imag[m];
				gpu_imag[b] = sub_real*gpu_w_x_imag[m] + sub_imag*gpu_w_x_real[m];
			}
			);
			half = half >> 1;
		}
		concurrency::parallel_for_each(
			gpu.get_default_view(),
			gpu_bit_r_col.extent,
			[=, &gpu_bit_r_col](concurrency::index<1> idx)restrict(amp)
		{
			gpu_bit_r_col[idx] = bit_r_gpu((int)idx[0], col);
		}
		);
		concurrency::extent<2> iter2;
		iter2 = { row,col };
		concurrency::parallel_for_each(
			gpu.get_default_view(),
			iter2,
			[=, &gpu_bit_r_col, &gpu_inner_real, &gpu_inner_imag, &gpu_real, &gpu_imag](concurrency::index<2> idx)restrict(amp)
		{
			concurrency::index<2> r = { gpu_bit_r_col[idx[1]],idx[0] };
			gpu_inner_real[r] = gpu_real[idx];
			gpu_inner_imag[r] = gpu_imag[idx];
		}
		);
	}
	{
		int half = row >> 1;
		concurrency::extent<2> iter3;
		iter3 = { col,half };
		for (int i = 1; i < row; i = i << 1) {
			auto half_2 = half << 1;
			concurrency::parallel_for_each(
				gpu.get_default_view(),
				iter3,
				[=, &gpu_inner_real, &gpu_inner_imag, &gpu_w_y_real, &gpu_w_y_imag](concurrency::index<2> idx)restrict(amp)
			{
				using namespace concurrency::fast_math;
				int j = fmodf(idx[1], half);
				concurrency::index<2> a = { idx[0],j + (int)floorf(idx[1] / half)*half_2 };
				concurrency::index<2> b = { a[0],a[1] + half };
				auto m = j * i;
				auto add_real = gpu_inner_real[a] + gpu_inner_real[b];
				auto add_imag = gpu_inner_imag[a] + gpu_inner_imag[b];
				auto sub_real = (gpu_inner_real[a] - gpu_inner_real[b]);
				auto sub_imag = (gpu_inner_imag[a] - gpu_inner_imag[b]);
				gpu_inner_real[a] = add_real;
				gpu_inner_imag[a] = add_imag;
				gpu_inner_real[b] = sub_real*gpu_w_y_real[m] - sub_imag*gpu_w_y_imag[m];
				gpu_inner_imag[b] = sub_real*gpu_w_y_imag[m] + sub_imag*gpu_w_y_real[m];
			}
			);
			half = half >> 1;
		}
		concurrency::parallel_for_each(
			gpu.get_default_view(),
			gpu_bit_r_row.extent,
			[=, &gpu_bit_r_row](concurrency::index<1> idx)restrict(amp)
		{
			gpu_bit_r_row[idx] = bit_r_gpu((int)idx[0], row);
		}
		);
		concurrency::extent<2> iter4;
		iter4 = { col,row };
		concurrency::parallel_for_each(
			gpu.get_default_view(),
			iter4,
			[=, &gpu_bit_r_row, &gpu_inner_real, &gpu_inner_imag, &gpu_real, &gpu_imag](concurrency::index<2> idx)restrict(amp)
		{
			concurrency::index<2> r = { gpu_bit_r_row[idx[1]],idx[0] };
			gpu_real[r] = gpu_inner_real[idx];
			gpu_imag[r] = gpu_inner_imag[idx];
		}
		);
	}
	concurrency::copy(gpu_real, for_gpu_real.data());
	concurrency::copy(gpu_imag, for_gpu_imag.data());
#ifdef _OPENMP
#pragma omp parallel for 
#endif
		for (int y = 0; y < row; y++) {
			std::copy(&for_gpu_real.at(y*col), &for_gpu_real.at(y*col) + col, fft_img->at(1).at(y).begin());
			std::copy(&for_gpu_imag.at(y*col), &for_gpu_imag.at(y*col) + col, fft_img->at(0).at(y).begin());
		}
	return fft_img;
}

cv::Mat array2mat(const std::unique_ptr<std::vector<std::vector<std::vector<float>>>> fft_img,const cv::Mat &src_img, const concurrency::accelerator gpu) {
	int row = src_img.rows;
	int col = src_img.cols;
	cv::Mat compute_img = cv::Mat::zeros(cv::Size(col, row), CV_8U);
	std::vector<float> ft_power(row*col,0),max_list(std::max(col, row), FLT_MIN);
	concurrency::array<float, 1> gpu_max(std::max(col, row), max_list.begin(), max_list.end());
	concurrency::array<float, 2> gpu_power(row, col, ft_power.begin(), ft_power.end());
	concurrency::array<float, 2> gpu_img(row, col);
	std::vector<float> for_gpu_real(row*col, 0), for_gpu_imag(row*col, 0);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < row; i++) {
		std::copy(fft_img->at(0).at(i).begin(), fft_img->at(0).at(i).end(), &for_gpu_real.at(i*col));
		std::copy(fft_img->at(1).at(i).begin(), fft_img->at(1).at(i).end(), &for_gpu_imag.at(i*col));
	}

	concurrency::array<float, 2> gpu_fft_real(row, col, for_gpu_real.begin(), for_gpu_real.end());
	concurrency::array<float, 2> gpu_fft_imag(row, col, for_gpu_imag.begin(), for_gpu_imag.end());
	concurrency::parallel_for_each(
		gpu.get_default_view(),
		gpu_power.extent,
		[=, &gpu_power, &gpu_fft_real, &gpu_fft_imag](concurrency::index<2> idx)restrict(amp) {
		using namespace concurrency;
		gpu_power[idx] = fast_math::logf(fast_math::powf(gpu_fft_real[idx], 2) + fast_math::powf(gpu_fft_imag[idx], 2));
	}
	);
	for (int i = 0; i < std::min(col, row); i++) {
		concurrency::extent<1> iter;
		iter[0] = std::max(col, row);
		concurrency::parallel_for_each(
			gpu.get_default_view(),
			iter,
			[=, &gpu_power, &gpu_max](concurrency::index<1> idx)restrict(amp) {
			using namespace concurrency;
			concurrency::index<2> a = { i,idx[0] };
			auto max = fast_math::fmaxf(gpu_max[idx], gpu_power[a]);
			gpu_max[idx] = max;
		}
		);
	}
	max_list = gpu_max;
	{
		std::vector<float> tmp_max(omp_get_max_threads(), FLT_MIN);
#ifdef _OPENMP
#pragma  omp parallel for
#endif
		for (int i = 0; i < std::max(col, row); i++) {
			tmp_max.at(omp_get_thread_num()) = std::max(tmp_max.at(omp_get_thread_num()), max_list.at(i));
		}
		tmp_max.swap(max_list);
	}
	auto max = *std::max_element(max_list.begin(), max_list.end()) / 255.0;
	concurrency::parallel_for_each(
		gpu.get_default_view(),
		gpu_power.extent,
		[=, &gpu_power](concurrency::index<2> idx)restrict(amp) {
		using namespace concurrency;
		auto ans = (int)fast_math::round(gpu_power[idx] / max);
		gpu_power[idx] = (ans >= 255) ? 255 : (ans <= 0) ? 0 : ans;
	}
	);
	concurrency::extent<2> iter2;
	iter2 = { row >>1,col >>1 };
	concurrency::parallel_for_each(
		gpu.get_default_view(),
		iter2,
		[=, &gpu_power, &gpu_img](concurrency::index<2> idx)restrict(amp) {
		gpu_img[idx] = gpu_power[idx[0] + row / 2][idx[1] + col / 2];
		gpu_img[idx[0] + row / 2][idx[1]] = gpu_power[idx[0]][idx[1] + col / 2];
		gpu_img[idx[0]][idx[1] + col / 2] = gpu_power[idx[0] + row / 2][idx[1]];
		gpu_img[idx[0] + row / 2][idx[1] + col / 2] = gpu_power[idx];
	}
	);
	ft_power = gpu_img;

#ifdef _OPENMP
#pragma  omp parallel for
#endif
	for (int y = 0; y < row ; y++) {
		for (size_t x = 0; x != col; x++) {
			compute_img.at<uint8_t>(y, x) = ft_power.at(y*col+x);
		}
	}
	return compute_img;
}

cv::Mat load_img(std::string path, int n) {
	cv::Mat src = cv::imread(path, n);
	auto to2base = [](size_t x)-> size_t {return pow(2, ceil(log2(x))); };
	cv::Rect roi_rect;
	roi_rect.width = src.cols;
	roi_rect.height = src.rows;
	cv::Mat padded = cv::Mat::zeros(cv::Size(to2base(src.cols), to2base(src.rows)), CV_8U);
	cv::Mat roi(padded, roi_rect);
	src.copyTo(roi);
	return padded;
}

void fft_time_count(std::string path, concurrency::accelerator gpu) {
	cv::Mat src_img = std::move(load_img(path, 0));
	std::chrono::system_clock::time_point gpu_start = std::chrono::system_clock::now();
	fft_gpu(src_img, gpu);
	std::chrono::system_clock::time_point gpu_end = std::chrono::system_clock::now();
	std::cout << "GPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count() << "[ms]" << std::endl;
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	fft(src_img, gpu);
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::cout << "CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms]" << std::endl;
}

int main() { 
	concurrency::accelerator gpu = get_gpu();
	cv::namedWindow("src", CV_WINDOW_OPENGL);
	cv::Mat src_img = std::move(load_img(".//lena.jpg",0));
	std::cout << src_img.rows << " * " << src_img.cols << "pixel\n" << src_img.rows*src_img.cols << "pixel" << std::endl;
	cv::imshow("src", src_img);
	cv::waitKey(1);
	auto fft_data = std::move(fft_gpu(src_img,gpu));
	cv::Mat fft_img = std::move(array2mat(std::move(fft_data), src_img,gpu));
	cv::namedWindow("dst", CV_WINDOW_OPENGL);
	cv::imshow("dst", fft_img);
	cv::waitKey(1);
	fft_time_count(".//lena.jpg",gpu);
	cv::waitKey(0);
	return 0; 
}