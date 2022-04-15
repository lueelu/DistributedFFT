/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#define HIP_ENABLE_PRINTF

#include "heffte_backend_rocm.h"

#ifdef Heffte_ENABLE_ROCM

namespace heffte {

namespace gpu {

int device_count(){
    int count;
    rocm::check_error(hipGetDeviceCount(&count), "hipGetDeviceCount()" );
    return count;
}

void device_set(int active_device){
    if (active_device < 0 or active_device > device_count())
        throw std::runtime_error("device_set() called with invalid rocm device id");
    rocm::check_error(hipSetDevice(active_device), "hipSetDevice()");
}

void synchronize_default_stream(){
    rocm::check_error(hipStreamSynchronize(nullptr), "device synch"); // synch the default stream
}

}

namespace rocm {

/*
 * Launch with one thread per entry.
 *
 * If to_complex is true, convert one real number from source to two real numbers in destination.
 * If to_complex is false, convert two real numbers from source to one real number in destination.
 */
template<typename scalar_type, int num_threads, bool to_complex, typename index>
__global__ __launch_bounds__(num_threads) void real_complex_convert(index num_entries, scalar_type const source[], scalar_type destination[]){
    index i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        if (to_complex){
            destination[2*i] = source[i];
            destination[2*i + 1] = 0.0;
        }else{
            destination[i] = source[2*i];
        }
        i += num_threads * gridDim.x;
    }
}

/*
 * Launch this with one block per line.
 */
template<typename scalar_type, int num_threads, int tuple_size, bool pack, typename index>
__global__ __launch_bounds__(num_threads) void direct_packer(index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                                          scalar_type const source[], scalar_type destination[]){
    index block_index = blockIdx.x;
    while(block_index < nmid * nslow){

        index mid = block_index % nmid;
        index slow = block_index / nmid;

        scalar_type const *block_source = (pack) ?
                            &source[tuple_size * (mid * line_stride + slow * plane_stide)] :
                            &source[block_index * nfast * tuple_size];
        scalar_type *block_destination = (pack) ?
                            &destination[block_index * nfast * tuple_size] :
                            &destination[tuple_size * (mid * line_stride + slow * plane_stide)];

        index i = threadIdx.x;
        while(i < nfast * tuple_size){
            block_destination[i] = block_source[i];
            i += num_threads;
        }

        block_index += gridDim.x;
    }
}

/*
 * Launch this with one block per line of the destination.
 */
template<typename scalar_type, int num_threads, int tuple_size, int map0, int map1, int map2, typename index>
__global__ __launch_bounds__(num_threads) void transpose_unpacker(index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                                   index buff_line_stride, index buff_plane_stride,
                                   scalar_type const source[], scalar_type destination[]){

    index block_index = blockIdx.x;
    while(block_index < nmid * nslow){

        index j = block_index % nmid;
        index k = block_index / nmid;

        index i = threadIdx.x;
        while(i < nfast){
            if (map0 == 0 and map1 == 1 and map2 == 2){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (k * buff_plane_stride + j * buff_line_stride + i)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (k * buff_plane_stride + j * buff_line_stride + i) + 1];
            }else if (map0 == 0 and map1 == 2 and map2 == 1){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (j * buff_plane_stride + k * buff_line_stride + i)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (j * buff_plane_stride + k * buff_line_stride + i) + 1];
            }else if (map0 == 1 and map1 == 0 and map2 == 2){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (k * buff_plane_stride + i * buff_line_stride + j)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (k * buff_plane_stride + i * buff_line_stride + j) + 1];
            }else if (map0 == 1 and map1 == 2 and map2 == 0){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (i * buff_plane_stride + k * buff_line_stride + j)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (i * buff_plane_stride + k * buff_line_stride + j) + 1];
            }else if (map0 == 2 and map1 == 1 and map2 == 0){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (i * buff_plane_stride + j * buff_line_stride + k)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (i * buff_plane_stride + j * buff_line_stride + k) + 1];
            }else if (map0 == 2 and map1 == 0 and map2 == 1){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (j * buff_plane_stride + i * buff_line_stride + k)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (j * buff_plane_stride + i * buff_line_stride + k) + 1];
            }
            i += num_threads;
        }

        block_index += gridDim.x;
    }
}

/*
 * Call with one thread per entry.
 */
template<typename scalar_type, int num_threads, typename index>
__global__ __launch_bounds__(num_threads) void simple_scal(index num_entries, scalar_type data[], scalar_type scaling_factor){
    index i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        data[i] *= scaling_factor;
        i += num_threads * gridDim.x;
    }
}

/*
 * Create a 1-D CUDA thread grid using the total_threads and number of threads per block.
 * Basically, computes the number of blocks but no more than 65536.
 */
struct thread_grid_1d{
    // Compute the threads and blocks.
    thread_grid_1d(int total_threads, int num_per_block) :
        threads(num_per_block),
        blocks(std::min(total_threads / threads + ((total_threads % threads == 0) ? 0 : 1), 65536))
    {}
    // number of threads
    int const threads;
    // number of blocks
    int const blocks;
};

// max number of cuda threads (Volta supports more, but I don't think it matters)
constexpr int max_threads  = 1024;
// allows expressive calls to_complex or not to_complex
constexpr bool to_complex  = true;
// allows expressive calls to_pack or not to_pack
constexpr bool to_pack     = true;

template<typename precision_type, typename index>
void convert(hipStream_t stream, index num_entries, precision_type const source[], std::complex<precision_type> destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    real_complex_convert<precision_type, max_threads, to_complex><<<grid.blocks, grid.threads, 0, stream>>>(num_entries, source, reinterpret_cast<precision_type*>(destination));
}

template<typename precision_type, typename index>
void convert(hipStream_t stream, index num_entries, std::complex<precision_type> const source[], precision_type destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    real_complex_convert<precision_type, max_threads, not to_complex><<<grid.blocks, grid.threads, 0, stream>>>(num_entries, reinterpret_cast<precision_type const*>(source), destination);
}

#define heffte_instantiate_convert(precision, index) \
    template void convert<precision, index>(hipStream_t, index num_entries, precision const source[], std::complex<precision> destination[]); \
    template void convert<precision, index>(hipStream_t, index num_entries, std::complex<precision> const source[], precision destination[]); \

heffte_instantiate_convert(float, int)
heffte_instantiate_convert(double, int)
heffte_instantiate_convert(float, long long)
heffte_instantiate_convert(double, long long)

/*
 * For float and double, defines type = <float/double> and tuple_size = 1
 * For complex float/double, defines type <float/double> and typle_size = 2
 */
template<typename scalar_type> struct precision{
    using type = scalar_type;
    static const int tuple_size = 1;
};
template<typename precision_type> struct precision<std::complex<precision_type>>{
    using type = precision_type;
    static const int tuple_size = 2;
};

template<typename scalar_type, typename index>
void direct_pack(hipStream_t stream, index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                 scalar_type const source[], scalar_type destination[]){
    constexpr index max_blocks = 65536;
    using prec = typename precision<scalar_type>::type;
    direct_packer<prec, max_threads, precision<scalar_type>::tuple_size, to_pack>
            <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>(nfast, nmid, nslow, line_stride, plane_stide,
            reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
}

template<typename scalar_type, typename index>
void direct_unpack(hipStream_t stream, index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                   scalar_type const source[], scalar_type destination[]){
    constexpr index max_blocks = 65536;
    using prec = typename precision<scalar_type>::type;
    direct_packer<prec, max_threads, precision<scalar_type>::tuple_size, not to_pack>
            <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>(nfast, nmid, nslow, line_stride, plane_stide,
            reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
}

template<typename scalar_type, typename index>
void transpose_unpack(hipStream_t stream, index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                      index buff_line_stride, index buff_plane_stride, int map0, int map1, int map2,
                      scalar_type const source[], scalar_type destination[]){
    constexpr index max_blocks = 65536;
    using prec = typename precision<scalar_type>::type;
    if (map0 == 0 and map1 == 1 and map2 == 2){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 0, 1, 2>
                <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 0 and map1 == 2 and map2 == 1){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 0, 2, 1>
                <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 1 and map1 == 0 and map2 == 2){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 1, 0, 2>
                <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 1 and map1 == 2 and map2 == 0){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 1, 2, 0>
                <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 2 and map1 == 0 and map2 == 1){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 2, 0, 1>
                <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 2 and map1 == 1 and map2 == 0){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 2, 1, 0>
                <<<std::min(nmid * nslow, max_blocks), max_threads, 0, stream>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }
}

#define heffte_instantiate_packers(index) \
template void direct_pack<float, index>(hipStream_t, index, index, index, index, index, float const source[], float destination[]); \
template void direct_pack<double, index>(hipStream_t, index, index, index, index, index, double const source[], double destination[]); \
template void direct_pack<std::complex<float>, index>(hipStream_t, index, index, index, index, index, \
                                                      std::complex<float> const source[], std::complex<float> destination[]); \
template void direct_pack<std::complex<double>, index>(hipStream_t, index, index, index, index, index, \
                                                       std::complex<double> const source[], std::complex<double> destination[]); \
\
template void direct_unpack<float, index>(hipStream_t, index, index, index, index, index, float const source[], float destination[]); \
template void direct_unpack<double, index>(hipStream_t, index, index, index, index, index, double const source[], double destination[]); \
template void direct_unpack<std::complex<float>, index>(hipStream_t, index, index, index, index, index, \
                                                        std::complex<float> const source[], std::complex<float> destination[]); \
template void direct_unpack<std::complex<double>, index>(hipStream_t, index, index, index, index, index, \
                                                         std::complex<double> const source[], std::complex<double> destination[]); \
\
template void transpose_unpack<float, index>(hipStream_t, index, index, index, index, index, index, index, int, int, int, \
                                             float const source[], float destination[]); \
template void transpose_unpack<double, index>(hipStream_t, index, index, index, index, index, index, index, int, int, int, \
                                              double const source[], double destination[]); \
template void transpose_unpack<std::complex<float>, index>(hipStream_t, index, index, index, index, index, index, index, int, int, int, \
                                                           std::complex<float> const source[], std::complex<float> destination[]); \
template void transpose_unpack<std::complex<double>, index>(hipStream_t, index, index, index, index, index, index, index, int, int, int, \
                                                            std::complex<double> const source[], std::complex<double> destination[]); \

heffte_instantiate_packers(int)
heffte_instantiate_packers(long long)


template<typename scalar_type, typename index>
void scale_data(hipStream_t stream, index num_entries, scalar_type *data, double scale_factor){
    thread_grid_1d grid(num_entries, max_threads);
    simple_scal<scalar_type, max_threads><<<grid.blocks, grid.threads, 0, stream>>>(num_entries, data, static_cast<scalar_type>(scale_factor));
}

template void scale_data<float, int>(hipStream_t, int num_entries, float *data, double scale_factor);
template void scale_data<double, int>(hipStream_t, int num_entries, double *data, double scale_factor);
template void scale_data<float, long long>(hipStream_t, long long num_entries, float *data, double scale_factor);
template void scale_data<double, long long>(hipStream_t, long long num_entries, double *data, double scale_factor);

} // namespace rocm

} // namespace heffte

#endif
