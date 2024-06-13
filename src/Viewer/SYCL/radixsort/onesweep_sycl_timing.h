#ifndef MULTISENSE_VIEWER_ONESWEEP_SYCL
#define MULTISENSE_VIEWER_ONESWEEP_SYCL


#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#define RADIX               256     // Number of digit bins
#define RADIX_MASK          255     // Mask of digit bins, to extract digits
#define RADIX_LOG           8       // log2(RADIX)

#define SEC_RADIX_START     256
#define THIRD_RADIX_START   512
#define FOURTH_RADIX_START  768

// For the upfront global histogram kernel
#define G_HIST_PART_SIZE    65536
#define G_HIST_VEC_SIZE     16384

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0                                       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1                                       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2                                       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3                                       //Mask used to retrieve flag values
#define LANE_COUNT 32
#define LANE_MASK 31
#define LANE_LOG 5

//For the digit binning
#define BIN_PART_SIZE       7680                                     //Partition tile size in k_DigitBinning
#define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
#define BIN_SUB_PART_SIZE   960                                      //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_WARPS           16                                      //Warps per threadblock in k_DigitBinning
#define BIN_KEYS_PER_THREAD 30                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)        //Starting offset of a partition tile

namespace VkRender::crl {

    struct uint4 {
        unsigned int x, y, z, w;
    };

    inline uint32_t getLaneId(sycl::sub_group &sg) {
        return sg.get_local_id();
    }

    uint32_t getLaneMaskLt(sycl::sub_group sg) {
        return (1 << sg.get_local_id()) - 1;
    }

    uint32_t ActiveExclusiveWarpScan(sycl::sub_group sg, uint32_t val, bool debug = false) {
        // Create a mask of active threads in the subgroup
        uint32_t mask = 255; // TODO hardcoded?
        // Perform the warp scan operation considering only active threads
        for (int i = 1; i <= 16; i <<= 1) {
            uint32_t t = sycl::select_from_group(sg, val, sg.get_local_id() - i);
            t = (sg.get_local_id() >= i && (mask & (1 << (sg.get_local_id() - i)))) ? t : 0;
            val += t;
        }
        uint32_t t = sycl::select_from_group(sg, val, sg.get_local_id() - 1);
        t = (sg.get_local_id() >= 1 && (mask & (1 << (sg.get_local_id() - 1)))) ? t : 0;
        return getLaneId(sg) ? t : 0;
    }


    inline uint32_t InclusiveWarpScanCircularShift(sycl::sub_group &subGroup, uint32_t val) {
        for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
        {
            uint32_t t = sycl::shift_group_right(subGroup, val, i);
            if (getLaneId(subGroup) >= i)
                val += t;
        }
        return sycl::select_from_group(subGroup, val,
                                       getLaneId(subGroup) + LANE_MASK & LANE_MASK);
    }

//Test for correctness
    void ValidationTest(std::vector<uint32_t> &keys, uint32_t size, uint32_t testIterations) {
        printf("Beginning VALIDATION tests at size %u and %u iterations. \n", size, testIterations);
        int testsPassed = 0;
        for (uint32_t i = 1; i <= testIterations; ++i) {
            bool isCorrect = true;
            for (uint32_t k = 1; k < keys.size(); ++k) {
                if (keys[k] < keys[k - 1]) {
                    isCorrect = false;
                    break;
                }
            }
            if (isCorrect)
                testsPassed++;
            else
                printf("Test iteration %d failed. \n", i);
        }
        printf("%d/%d tests passed.\n", testsPassed, testIterations);
    }

    void printKeyValue(const std::vector<uint32_t> &key, const std::vector<uint32_t> &value, uint32_t num = 128) {
        // Print the keys
        std::cout << "Key:   ";
        for (size_t i = 0; i < key.size(); ++i) {
            std::cout << std::setw(4) << key[i] << " ";
            if (i >= num - 1 || i >= key.size()) {
                break;
            }
        }
        std::cout << std::endl;
        // Print the values
        std::cout << "Value: ";
        for (size_t i = 0; i < value.size(); ++i) {
            std::cout << std::setw(4) << value[i] << " ";
            if (i >= num - 1 || i >= key.size()) {
                break;
            }
        }
        std::cout << std::endl;
    }


    void DigitBinningPassFunc(sycl::queue &queue, sycl::buffer<uint32_t, 1> &sortBuffer,
                              sycl::buffer<uint32_t, 1> &sortAltBuffer, sycl::buffer<uint32_t, 1> &valuesBuffer,
                              sycl::buffer<uint32_t, 1> &valuesAltBuffer,
                              sycl::buffer<uint32_t, 1> &passHistogramBuffer,
                              sycl::buffer<uint32_t, 1> &indexBuffer,
                              uint32_t radixShift, uint32_t binningThreadBlocks, uint32_t binningThreads, uint32_t size) {


        queue.submit([&](sycl::handler &h) {
            sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
            sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);

            auto indexAcc = indexBuffer.get_access<sycl::access::mode::read_write>(h);
            auto sortAcc = sortBuffer.get_access<sycl::access::mode::read_write>(h);
            auto sortAltAcc = sortAltBuffer.get_access<sycl::access::mode::read_write>(h);
            auto valuesAcc = valuesBuffer.get_access<sycl::access::mode::read_write>(h);
            auto valuesAltAcc = valuesAltBuffer.get_access<sycl::access::mode::read_write>(h);
            auto passHistogramAcc = passHistogramBuffer.get_access<sycl::access::mode::read_write>(h);

            // Kernel code
            h.parallel_for<class digit_binning>(
                    sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadBlocks),
                                      sycl::range<1>(binningThreads)),
                    [=](sycl::nd_item<1> item) {

                        uint32_t threadIdx = item.get_local_id(0);
                        uint32_t blockIdx = item.get_group(0);
                        uint32_t blockDim = item.get_local_range(0);
                        uint32_t gridDim = item.get_group_range(0);

                        auto subGroup = item.get_sub_group();
                        uint32_t warpIndex = subGroup.get_group_id().get(0);
                        uint32_t *s_warpHist = &s_warpHistograms[warpIndex << RADIX_LOG];
                        // Clear shared memory
                        for (uint32_t i = threadIdx; i < BIN_HISTS_SIZE; i += blockDim)
                            s_warpHistograms[i] = 0;
                        // Atomically assign partition tiles
                        if (threadIdx == 0)
                            s_warpHistograms[BIN_PART_SIZE -
                                             1] = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(
                                    indexAcc[radixShift >> 3]).fetch_add(1);
                        item.barrier(sycl::access::fence_space::local_space);

                        const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];


                        //To handle input sizes not perfect multiples of the partition tile size
                        if (partitionIndex < gridDim - 1) {
                            uint32_t keys[BIN_KEYS_PER_THREAD];
                            uint32_t binSubPartStart = warpIndex * BIN_SUB_PART_SIZE;
                            uint32_t binPartStart = partitionIndex * BIN_PART_SIZE;

                            for (uint32_t i = 0, t = getLaneId(subGroup) + binSubPartStart + binPartStart;
                                 i < BIN_KEYS_PER_THREAD; ++i, t += subGroup.get_local_range().get(0))
                                keys[i] = sortAcc[t];

                            uint16_t offsets[BIN_KEYS_PER_THREAD];

                            for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
                                unsigned warpFlags = 0xffffffff;

                                for (uint32_t k = 0; k < RADIX_LOG; ++k) {
                                    const bool t = (keys[i] >> (k + radixShift)) & 1;
                                    uint32_t t_mask = t ? 0 : 0xffffffff;
                                    uint32_t mask;
                                    sycl::ext::oneapi::group_ballot(subGroup, t).extract_bits(mask);
                                    warpFlags &= mask ^ t_mask;
                                }

                                const uint32_t bits = sycl::popcount(warpFlags & getLaneMaskLt(subGroup));

                                uint32_t preIncrementVal;
                                if (bits == 0) {
                                    uint32_t bin = keys[i] >> radixShift & RADIX_MASK;
                                    uint32_t val = sycl::popcount(warpFlags);

                                    // Perform atomic add on local memory
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomicRef(
                                            s_warpHist[bin]);
                                    preIncrementVal = atomicRef.fetch_add(val);

                                }
                                uint32_t firstSetBit = __builtin_ctz(warpFlags);
                                offsets[i] = sycl::select_from_group(subGroup, preIncrementVal,
                                                                     firstSetBit) + bits;
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                            //exclusive prefix sum up the warp histograms
                            if (threadIdx < RADIX) {
                                uint32_t reduction = s_warpHistograms[threadIdx];
                                for (uint32_t i = threadIdx + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
                                    reduction += s_warpHistograms[i];
                                    s_warpHistograms[i] = reduction - s_warpHistograms[i];
                                }
                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomicRef(
                                        passHistogramAcc[threadIdx + (partitionIndex + 1) * RADIX]);
                                atomicRef.fetch_add(FLAG_REDUCTION | reduction << 2);
                                s_localHistogram[threadIdx] = InclusiveWarpScanCircularShift(subGroup, reduction);
                            }
                            {
                                item.barrier(sycl::access::fence_space::local_space);
                                // Ensure all threads participate in this section
                                uint32_t reduction = (threadIdx < (RADIX >> LANE_LOG)) ? s_localHistogram[threadIdx
                                        << LANE_LOG]
                                                                                       : 0;
                                uint32_t result = ActiveExclusiveWarpScan(subGroup, reduction);
                                if (threadIdx < (RADIX >> LANE_LOG)) {
                                    s_localHistogram[threadIdx << LANE_LOG] = result;
                                }
                                item.barrier(sycl::access::fence_space::local_space);

                                uint32_t res = sycl::select_from_group(subGroup, s_localHistogram[threadIdx - 1], 1);
                                if (threadIdx < RADIX && getLaneId(subGroup)) {
                                    s_localHistogram[threadIdx] += res;


                                }
                                item.barrier(sycl::access::fence_space::local_space);
                                //update offsets
                                if (warpIndex) {
                                    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
                                        const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
                                        offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
                                    }
                                } else {
                                    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
                                        offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
                                }
                                item.barrier(sycl::access::fence_space::local_space);

                                for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
                                    s_warpHistograms[offsets[i]] = keys[i];
                            }

                            if (threadIdx < RADIX) {
                                uint32_t reduction = 0;
                                for (uint32_t k = partitionIndex; k >= 0;) {
                                    const uint32_t flagPayload = passHistogramAcc[threadIdx + k * RADIX];

                                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                                        reduction += flagPayload >> 2;

                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomicRef(
                                                passHistogramAcc[threadIdx + (partitionIndex + 1) * RADIX]);
                                        atomicRef.fetch_add(1 | (reduction << 2));

                                        s_localHistogram[threadIdx] = reduction - s_localHistogram[threadIdx];
                                        break;
                                    }

                                    if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
                                        reduction += flagPayload >> 2;
                                        k--;
                                    }
                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);

                            //scatter runs of keys into device memory
                            for (uint32_t i = threadIdx; i < BIN_PART_SIZE; i += blockDim) {
                                sortAltAcc[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] +
                                           i] = s_warpHistograms[i];
                            }
                        }
                        if (partitionIndex == gridDim - 1) {
                            // Immediately begin lookback
                            if (threadIdx < RADIX) {
                                if (partitionIndex) {

                                    uint32_t reduction = 0;
                                    for (uint32_t k = partitionIndex; k >= 0;) {
                                        const uint32_t flagPayload = passHistogramAcc[threadIdx + k * RADIX];
                                        if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                                            reduction += flagPayload >> 2;
                                            s_localHistogram[threadIdx] = reduction;
                                            break;
                                        }
                                        if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
                                            reduction += flagPayload >> 2;
                                            k--;
                                        }
                                    }
                                } else {
                                    s_localHistogram[threadIdx] = passHistogramAcc[threadIdx] >> 2;

                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);

                            const uint32_t partEnd = BIN_PART_START + BIN_PART_SIZE;
                            for (uint32_t i = threadIdx + BIN_PART_START; i < partEnd; i += blockDim) {
                                uint32_t key, value;
                                uint32_t offset;
                                uint32_t warpFlags = 0xffffffff;
                                if (i < size) {
                                    key = sortAcc[i];
                                    value = valuesAcc[i];
                                }
                                for (uint32_t k = 0; k < RADIX_LOG; ++k) {
                                    const bool t = (key >> (k + radixShift)) & 1;
                                    uint32_t t_mask = t ? 0 : 0xffffffff;
                                    uint32_t mask;
                                    sycl::ext::oneapi::group_ballot(subGroup, t).extract_bits(mask);
                                    warpFlags &= mask ^ t_mask;
                                }
                                const uint32_t bits = sycl::popcount(warpFlags & getLaneMaskLt(subGroup));

                                uint32_t dummy;
                                uint32_t preIncrementVal;

                                for (uint32_t k = 0; k < BIN_WARPS; ++k) {
                                    bool increment = warpIndex == k && bits == 0 &&
                                                     i < size;
                                    if (increment) {
                                        uint32_t bin = key >> radixShift & RADIX_MASK;
                                        uint32_t val = sycl::popcount(warpFlags);
                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> atomicRef(
                                                s_localHistogram[bin]);
                                        preIncrementVal = atomicRef.fetch_add(val);

                                    }
                                    uint32_t firstSetBit = __builtin_ctz(warpFlags);
                                    dummy = sycl::select_from_group(subGroup, preIncrementVal,
                                                                    firstSetBit);
                                    if (warpIndex == k) {
                                        offset = dummy + bits;
                                    }
                                    item.barrier(sycl::access::fence_space::local_space);
                                }
                                if (i < size) {
                                    sortAltAcc[offset] = key;
                                    valuesAltAcc[offset] = value;
                                }
                            }
                        }
                    });

        }).wait();

    }

    void performOneSweep(sycl::queue &queue, uint32_t size, sycl::buffer<uint32_t, 1>& sortBuffer, sycl::buffer<uint32_t, 1>& valuesBuffer) {
        uint32_t iterations = 1;

        const uint32_t testIterations = 1;

        const uint32_t radix = 256;
        const uint32_t radixPasses = 4;
        const uint32_t partitionSize = 7680;
        const uint32_t globalHistPartitionSize = 65536;
        const uint32_t globalHistThreads = 128;
        const uint32_t binningThreads = 256;            //2080 super seems to really like 512
        const uint32_t binningThreadblocks = (size + partitionSize - 1) / partitionSize;
        const uint32_t globalHistThreadblocks = (size + globalHistPartitionSize - 1) / globalHistPartitionSize;

        std::vector<uint32_t> sortAlt(size);
        std::vector<uint32_t> valuesAlt(size);
        std::vector<uint32_t> globalHistogram(radix * radixPasses, 0);
        std::vector<uint32_t> firstPassHistogram(RADIX * binningThreadblocks, 0);
        std::vector<uint32_t> secPassHistogram(RADIX * binningThreadblocks, 0);
        std::vector<uint32_t> thirdPassHistogram(RADIX * binningThreadblocks, 0);
        std::vector<uint32_t> fourthPassHistogram(RADIX * binningThreadblocks, 0);
        std::vector<uint32_t> index(RADIX, 0);


        {// Create SYCL Buffers
            //sycl::buffer<uint32_t, 1> sortBuffer(key, sycl::range<1>(size));
            //sycl::buffer<uint32_t, 1> valuesBuffer(value, sycl::range<1>(size));

            sycl::buffer<uint32_t, 1> sortAltBuffer(sortAlt.data(), sycl::range<1>(size));
            sycl::buffer<uint32_t, 1> valuesAltBuffer(valuesAlt.data(), sycl::range<1>(size));
            sycl::buffer<uint32_t, 1> globalHistogramBuffer(globalHistogram.data(),
                                                            sycl::range<1>(radix * radixPasses));
            sycl::buffer<uint32_t, 1> firstPassHistogramBuffer(firstPassHistogram.data(),
                                                               sycl::range<1>(firstPassHistogram.size()));
            sycl::buffer<uint32_t, 1> secPassHistogramBuffer(secPassHistogram.data(),
                                                             sycl::range<1>(secPassHistogram.size()));
            sycl::buffer<uint32_t, 1> thirdPassHistogramBuffer(thirdPassHistogram.data(),
                                                               sycl::range<1>(thirdPassHistogram.size()));
            sycl::buffer<uint32_t, 1> fourthPassHistogramBuffer(fourthPassHistogram.data(),
                                                                sycl::range<1>(fourthPassHistogram.size()));
            sycl::buffer<uint32_t, 1> indexBuffer(index.data(), sycl::range<1>(index.size()));

                // GlobalHistogram
                queue.submit([&](sycl::handler &h) {
                    // Shared memory allocations
                    sycl::local_accessor<uint32_t, 1> s_globalHistFirst(sycl::range<1>(RADIX * 2), h);
                    sycl::local_accessor<uint32_t, 1> s_globalHistSec(sycl::range<1>(RADIX * 2), h);
                    sycl::local_accessor<uint32_t, 1> s_globalHistThird(sycl::range<1>(RADIX * 2), h);
                    sycl::local_accessor<uint32_t, 1> s_globalHistFourth(sycl::range<1>(RADIX * 2), h);
                    auto sortAcc = sortBuffer.get_access<sycl::access_mode::read_write>(h);
                    auto globalHistAcc = globalHistogramBuffer.get_access<sycl::access_mode::read_write>(h);
                    // Kernel code
                    h.parallel_for<class global_histogram_kernel>(
                            sycl::nd_range<1>(sycl::range<1>(globalHistThreads * globalHistThreadblocks),
                                              sycl::range<1>(globalHistThreads)),
                            [=](sycl::nd_item<1> item) {
                                uint32_t threadIdx = item.get_local_id(0);
                                uint32_t blockDim = item.get_local_range(0);
                                uint32_t blockIdx = item.get_group(0);
                                uint32_t gridDim = item.get_group_range(0);

                                // Clear shared memory
                                for (uint32_t i = threadIdx; i < RADIX * 2; i += blockDim) {
                                    s_globalHistFirst[i] = 0;
                                    s_globalHistSec[i] = 0;
                                    s_globalHistThird[i] = 0;
                                    s_globalHistFourth[i] = 0;
                                }
                                item.barrier(sycl::access::fence_space::local_space);

                                // Histogram
                                {
                                    // 64 threads : 1 histogram in shared memory
                                    uint32_t *s_wavesHistFirst = &s_globalHistFirst[threadIdx / 64 * RADIX];
                                    uint32_t *s_wavesHistSec = &s_globalHistSec[threadIdx / 64 * RADIX];
                                    uint32_t *s_wavesHistThird = &s_globalHistThird[threadIdx / 64 * RADIX];
                                    uint32_t *s_wavesHistFourth = &s_globalHistFourth[threadIdx / 64 * RADIX];

                                    if (blockIdx < gridDim - 1) {

                                        const uint32_t partEnd = (blockIdx + 1) * G_HIST_VEC_SIZE;

                                        for (uint32_t i = threadIdx + (blockIdx * G_HIST_VEC_SIZE);
                                             i < partEnd; i += blockDim) {

                                            uint4 t[1] = {
                                                    reinterpret_cast<uint4 *>(sortAcc.get_multi_ptr<sycl::access::decorated::no>().get())[i]};

                                            {
                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFirst_atomic(
                                                        s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[0]]);
                                                s_wavesHistFirst_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistSec_atomic(
                                                        s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[1]]);
                                                s_wavesHistSec_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistThird_atomic(
                                                        s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[2]]);
                                                s_wavesHistThird_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFourth_atomic(
                                                        s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[3]]);
                                                s_wavesHistFourth_atomic.fetch_add(1);
                                            }
                                            {
                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFirst_atomic(
                                                        s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[4]]);
                                                s_wavesHistFirst_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistSec_atomic(
                                                        s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[5]]);
                                                s_wavesHistSec_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistThird_atomic(
                                                        s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[6]]);
                                                s_wavesHistThird_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFourth_atomic(
                                                        s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[7]]);
                                                s_wavesHistFourth_atomic.fetch_add(1);
                                            }
                                            {
                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFirst_atomic(
                                                        s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[8]]);
                                                s_wavesHistFirst_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistSec_atomic(
                                                        s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[9]]);
                                                s_wavesHistSec_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistThird_atomic(
                                                        s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[10]]);
                                                s_wavesHistThird_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFourth_atomic(
                                                        s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[11]]);
                                                s_wavesHistFourth_atomic.fetch_add(1);
                                            }
                                            {
                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFirst_atomic(
                                                        s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[12]]);
                                                s_wavesHistFirst_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistSec_atomic(
                                                        s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[13]]);
                                                s_wavesHistSec_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistThird_atomic(
                                                        s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[14]]);
                                                s_wavesHistThird_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFourth_atomic(
                                                        s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[15]]);
                                                s_wavesHistFourth_atomic.fetch_add(1);
                                            }
                                        }
                                    }

                                    if (blockIdx == gridDim - 1) {
                                        for (uint32_t i = threadIdx + (blockIdx * G_HIST_PART_SIZE);
                                             i < size; i += blockDim) {
                                            uint32_t t[1] = {
                                                    reinterpret_cast<uint32_t *>(sortAcc.get_multi_ptr<sycl::access::decorated::no>().get())[i]};

                                            {
                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFirst_atomic(
                                                        s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[0]]);
                                                s_wavesHistFirst_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistSec_atomic(
                                                        s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[1]]);
                                                s_wavesHistSec_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistThird_atomic(
                                                        s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[2]]);
                                                s_wavesHistThird_atomic.fetch_add(1);

                                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFourth_atomic(
                                                        s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[3]]);
                                                s_wavesHistFourth_atomic.fetch_add(1);
                                            }
                                        }
                                    }

                                    item.barrier(sycl::access::fence_space::local_space);
                                    // Reduce and add to device
                                    for (uint32_t i = threadIdx; i < RADIX; i += blockDim) {

                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistFirst_atomic(
                                                globalHistAcc[i]);
                                        globalHistFirst_atomic.fetch_add(
                                                s_globalHistFirst[i] + s_globalHistFirst[i + RADIX]);

                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistSec_atomic(
                                                globalHistAcc[i + SEC_RADIX_START]);
                                        globalHistSec_atomic.fetch_add(s_globalHistSec[i] + s_globalHistSec[i + RADIX]);

                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistThird_atomic(
                                                globalHistAcc[i + THIRD_RADIX_START]);
                                        globalHistThird_atomic.fetch_add(
                                                s_globalHistThird[i] + s_globalHistThird[i + RADIX]);

                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistFourth_atomic(
                                                globalHistAcc[i + FOURTH_RADIX_START]);
                                        globalHistFourth_atomic.fetch_add(
                                                s_globalHistFourth[i] + s_globalHistFourth[i + RADIX]);
                                    }
                                }
                            });
                });


                /// SCAN PASS

                queue.submit([&](sycl::handler &h) {
                    // Device memory allocations
                    sycl::local_accessor<uint32_t, 1> s_scan(sycl::range<1>(RADIX), h);
                    auto globalHistogramAcc = globalHistogramBuffer.get_access<sycl::access::mode::read>(h);
                    auto firstPassHistogramAcc = firstPassHistogramBuffer.get_access<sycl::access::mode::write>(h);
                    auto secPassHistogramAcc = secPassHistogramBuffer.get_access<sycl::access::mode::write>(h);
                    auto thirdPassHistogramAcc = thirdPassHistogramBuffer.get_access<sycl::access::mode::write>(h);
                    auto fourthPassHistogramAcc = fourthPassHistogramBuffer.get_access<sycl::access::mode::write>(h);
                    // Kernel code
                    h.parallel_for<class scan_hists_kernel>(
                            sycl::nd_range<1>(sycl::range<1>(radixPasses * radix), sycl::range<1>(radix)),
                            [=](sycl::nd_item<1> item) {
                                uint32_t threadIdx = item.get_local_id(0);
                                uint32_t blockIdx = item.get_group(0);
                                auto subGroup = item.get_sub_group();

                                s_scan[threadIdx] = InclusiveWarpScanCircularShift(subGroup,
                                                                                   globalHistogramAcc[threadIdx +
                                                                                                      blockIdx *
                                                                                                      RADIX]);
                                item.barrier(sycl::access::fence_space::local_space);

                                uint32_t reduction = (threadIdx < (RADIX >> LANE_LOG)) ? s_scan[threadIdx << LANE_LOG]
                                                                                       : 0;
                                uint32_t result = ActiveExclusiveWarpScan(subGroup, reduction);
                                if (threadIdx < (RADIX >> LANE_LOG))
                                    s_scan[threadIdx << LANE_LOG] = result;
                                item.barrier(sycl::access::fence_space::local_space);


                                uint32_t res = getLaneId(subGroup) ? sycl::select_from_group(subGroup,
                                                                                             s_scan[threadIdx - 1], 1)
                                                                   : 0;
                                uint32_t histRes = s_scan[threadIdx] + res;
                                histRes = histRes << 2 | FLAG_INCLUSIVE;

                                switch (blockIdx) {
                                    case 0:
                                        firstPassHistogramAcc[threadIdx] = histRes;
                                        break;
                                    case 1:
                                        secPassHistogramAcc[threadIdx] = histRes;
                                        break;
                                    case 2:
                                        thirdPassHistogramAcc[threadIdx] = histRes;
                                        break;
                                    case 3:
                                        fourthPassHistogramAcc[threadIdx] = histRes;
                                        break;
                                    default:
                                        break;
                                }
                            });
                });

                // DIGIT BINNING PASSES

                DigitBinningPassFunc(queue, sortBuffer, sortAltBuffer, valuesBuffer, valuesAltBuffer,
                                     firstPassHistogramBuffer, indexBuffer, 0,
                                     binningThreadblocks, binningThreads, size);

                DigitBinningPassFunc(queue, sortAltBuffer, sortBuffer, valuesAltBuffer, valuesBuffer,
                                     secPassHistogramBuffer, indexBuffer, 8,
                                     binningThreadblocks, binningThreads, size);

                DigitBinningPassFunc(queue, sortBuffer, sortAltBuffer, valuesBuffer, valuesAltBuffer,
                                     thirdPassHistogramBuffer, indexBuffer, 16,
                                     binningThreadblocks, binningThreads, size);

                DigitBinningPassFunc(queue, sortAltBuffer, sortBuffer, valuesAltBuffer, valuesBuffer,
                                     fourthPassHistogramBuffer, indexBuffer, 24,
                                     binningThreadblocks, binningThreads, size);


            }
            queue.wait();

    }
}

/*
int main() {
    sycl::queue queue{sycl::property::queue::in_order()};
    performOneSweep(queue);
    return 0;
}
 */

#endif