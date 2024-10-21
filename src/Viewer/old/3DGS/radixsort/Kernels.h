//
// Created by magnus on 7/3/24.
//

#ifndef MULTISENSE_VIEWER_KERNELS_H
#define MULTISENSE_VIEWER_KERNELS_H

#include <sycl/sycl.hpp>


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
#define BIN_SUB_PART_SIZE   480 * 2                                      //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_WARPS           16  / 2                                        //Warps per threadblock in k_DigitBinning
#define BIN_KEYS_PER_THREAD 15  * 2                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)        //Starting offset of a partition tile


namespace RadixSorter {

    struct uint4 {
        unsigned int x, y, z, w;
    };

    static inline uint32_t getLaneId(sycl::sub_group &sg) {
        return sg.get_local_id();
    }

    static uint32_t getLaneMaskLt(sycl::sub_group sg) {
        return (1 << sg.get_local_id()) - 1;
    }

    static uint32_t ActiveExclusiveWarpScan(sycl::sub_group sg, uint32_t val, bool debug = false) {

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


    static inline uint32_t InclusiveWarpScanCircularShift(sycl::sub_group &subGroup, uint32_t val, bool debug = false) {

        for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
        {
            uint32_t t = sycl::shift_group_right(subGroup, val, i);
            if (getLaneId(subGroup) >= i)
                val += t;
        }

        uint32_t result = select_from_group(subGroup, val,
                                            (getLaneId(subGroup) + subGroup.get_group_linear_range() - 1) &
                                            (subGroup.get_group_linear_range() -
                                             1)); // 16 = LANE_MASK and 31 = LANE_COUNT - 1

        return result;
    }


    class GlobalHistogram {
    public:
        GlobalHistogram(
                sycl::local_accessor<uint32_t, 1> s1,
                sycl::local_accessor<uint32_t, 1> s2,
                sycl::local_accessor<uint32_t, 1> s3,
                sycl::local_accessor<uint32_t, 1> s4,
                uint32_t *sort,
                uint32_t *globalHistogram,
                uint32_t size)
                : s_globalHistFirst(s1),
                  s_globalHistSec(s2),
                  s_globalHistThird(s3),
                  s_globalHistFourth(s4),
                  m_sort(sort),
                  m_globalHist(globalHistogram),
                  m_size(size) {}

    private:
        // Add the local accessor parameters
        sycl::local_accessor<uint32_t, 1> s_globalHistFirst;
        sycl::local_accessor<uint32_t, 1> s_globalHistSec;
        sycl::local_accessor<uint32_t, 1> s_globalHistThird;
        sycl::local_accessor<uint32_t, 1> s_globalHistFourth;
        uint32_t *m_sort;
        uint32_t *m_globalHist;
        uint32_t m_size = 0;
    public:
        void operator()(sycl::nd_item<1> item) const {
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

                        uint4 t[1] = {reinterpret_cast<uint4 *>(m_sort)[i]};

                        uint8_t val1 = reinterpret_cast<uint8_t *>(t)[0];
                        uint8_t val2 = reinterpret_cast<uint8_t *>(t)[1];
                        uint8_t val3 = reinterpret_cast<uint8_t *>(t)[2];
                        uint8_t val4 = reinterpret_cast<uint8_t *>(t)[3];

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
                         i < m_size; i += blockDim) {
                        uint32_t t[1] = {(m_sort[i])};
                        uint8_t val1 = reinterpret_cast<uint8_t *>(t)[0];
                        uint8_t val2 = reinterpret_cast<uint8_t *>(t)[1];
                        uint8_t val3 = reinterpret_cast<uint8_t *>(t)[2];
                        uint8_t val4 = reinterpret_cast<uint8_t *>(t)[3];
                        {
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFirst_atomic(
                                    s_wavesHistFirst[val1]);
                            s_wavesHistFirst_atomic.fetch_add(1);

                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistSec_atomic(
                                    s_wavesHistSec[val2]);
                            s_wavesHistSec_atomic.fetch_add(1);

                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistThird_atomic(
                                    s_wavesHistThird[val3]);
                            s_wavesHistThird_atomic.fetch_add(1);

                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> s_wavesHistFourth_atomic(
                                    s_wavesHistFourth[val4]);
                            s_wavesHistFourth_atomic.fetch_add(1);
                        }
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);
                // Reduce and add to device
                for (uint32_t i = threadIdx; i < RADIX; i += blockDim) {

                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistFirst_atomic(
                            m_globalHist[i]);
                    globalHistFirst_atomic.fetch_add(
                            s_globalHistFirst[i] + s_globalHistFirst[i + RADIX]);

                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistSec_atomic(
                            m_globalHist[i + SEC_RADIX_START]);
                    globalHistSec_atomic.fetch_add(s_globalHistSec[i] + s_globalHistSec[i + RADIX]);

                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistThird_atomic(
                            m_globalHist[i + THIRD_RADIX_START]);
                    globalHistThird_atomic.fetch_add(
                            s_globalHistThird[i] + s_globalHistThird[i + RADIX]);

                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> globalHistFourth_atomic(
                            m_globalHist[i + FOURTH_RADIX_START]);
                    globalHistFourth_atomic.fetch_add(
                            s_globalHistFourth[i] + s_globalHistFourth[i + RADIX]);
                }
            }
        }
    };

    class ScanPass {
    public:
        // Constructor
        ScanPass(
                sycl::local_accessor<uint32_t, 1> s_scan,
                uint32_t *globalHistogramBuffer,
                uint32_t *firstPassHistogramBuffer,
                uint32_t *secPassHistogramBuffer,
                uint32_t *thirdPassHistogramBuffer,
                uint32_t *fourthPassHistogramBuffer)
                : s_scan(s_scan),
                  m_globalHistogramBuffer(globalHistogramBuffer),
                  m_firstPassHistogramBuffer(firstPassHistogramBuffer),
                  m_secPassHistogramBuffer(secPassHistogramBuffer),
                  m_thirdPassHistogramBuffer(thirdPassHistogramBuffer),
                  m_fourthPassHistogramBuffer(fourthPassHistogramBuffer) {}

    private:
        // Member variables to store the passed arguments
        sycl::local_accessor<uint32_t, 1> s_scan;
        uint32_t *m_globalHistogramBuffer;
        uint32_t *m_firstPassHistogramBuffer;
        uint32_t *m_secPassHistogramBuffer;
        uint32_t *m_thirdPassHistogramBuffer;
        uint32_t *m_fourthPassHistogramBuffer;
    public:

        void operator()(sycl::nd_item<1> item) const {
            auto subGroup = item.get_sub_group();
            uint32_t warpIdx = subGroup.get_local_linear_id();
            auto warpSize = subGroup.get_max_local_range()[0];

            uint32_t threadIdx = item.get_local_id(0);
            uint32_t blockIdx = item.get_group(0);

            uint32_t req = m_globalHistogramBuffer[threadIdx +
                                                   blockIdx *
                                                   RADIX];

            /*
        s_scan[threadIdx] = InclusiveWarpScanCircularShift(subGroup, req);
        item.barrier(sycl::access::fence_space::local_space);

        uint32_t idx = threadIdx * warpSize;
        uint32_t val = 0;
        if (idx < RADIX)
            val = s_scan[idx];

        val = ActiveExclusiveWarpScan(subGroup, val);
        if (idx < RADIX)
            s_scan[idx] = val;
        */

            // inclusive scan + circular shift
            for (int i = 1; i < warpSize; i <<= 1) {
                uint32_t t = sycl::shift_group_right(subGroup, req, i);
                if (warpIdx >= i) // Dont add on our first lane
                    req += t;
            }
            uint32_t circular_val;
            int last_value = sycl::permute_group_by_xor(subGroup, req, subGroup.get_max_local_range()[0] - 1);
            req = sycl::shift_group_right(subGroup, req, 1);
            if (warpIdx == 0) {
                circular_val = last_value;
            } else {
                circular_val = req;
            }

            s_scan[threadIdx] = circular_val;

            item.barrier(sycl::access::fence_space::local_space);

            uint32_t idx = threadIdx * warpSize;
            uint32_t val = 0;
            if (idx < RADIX)
                val = s_scan[idx];
            else
                val = 0;
            for (int i = 1; i < warpSize; i <<= 1) {
                uint32_t t = sycl::shift_group_right(subGroup, val, i);
                if (warpIdx >= i)
                    val += t;
            }
            val = sycl::shift_group_right(subGroup, val, 1);
            val = warpIdx ? val : 0;
            if (idx < RADIX)
                s_scan[idx] = val; // TODO multiply with warpsize for small sets?


            item.barrier(sycl::access::fence_space::local_space);

            uint32_t res = warpIdx ? sycl::select_from_group(subGroup, s_scan[threadIdx - 1], 1) : 0;
            if (warpIdx == 0)
                res = 0;

            uint32_t scan = s_scan[threadIdx];
            uint32_t histRes = scan + res;
            histRes = histRes << 2 | FLAG_INCLUSIVE;

            //if (blockIdx == 0)
            //    sycl::ext::oneapi::experimental::printf("ThreadID: %u, idx: %u, WarpID: %u, histRes %u, res: %u\n",threadIdx, idx, warpIdx, histRes, res);

            switch (blockIdx) {
                case 0:
                    m_firstPassHistogramBuffer[threadIdx] = histRes;
                    break;
                case 1:
                    m_secPassHistogramBuffer[threadIdx] = histRes;
                    break;
                case 2:
                    m_thirdPassHistogramBuffer[threadIdx] = histRes;
                    break;
                case 3:
                    m_fourthPassHistogramBuffer[threadIdx] = histRes;
                    break;
                default:
                    break;
            }
        }

    };

    class DigitBinningPass {
    public:
        DigitBinningPass(
                sycl::local_accessor<uint32_t, 1> warpHist,
                sycl::local_accessor<uint32_t, 1> warpValueHist,
                sycl::local_accessor<uint32_t, 1> localHist,
                uint32_t *sortBuffer,
                uint32_t *sortAltBuffer,
                uint32_t *valuesBuffer,
                uint32_t *valuesAltBuffer,
                uint32_t *indexBuffer,
                uint32_t *passHist,
                uint32_t radixShift,
                uint32_t size)
                : s_warpHistograms(warpHist), s_warpValueHistograms(warpValueHist),
                  s_localHistogram(localHist),
                  m_sortBuffer(sortBuffer),
                  m_sortAltBuffer(sortAltBuffer),
                  m_valuesBuffer(valuesBuffer),
                  m_valuesAltBuffer(valuesAltBuffer),
                  m_indexBuffer(indexBuffer),
                  m_passHist(passHist),
                  m_radixShift(radixShift),
                  m_size(size) {}

    private:
        // Member variables to store the passed arguments
        sycl::local_accessor<uint32_t, 1> s_warpHistograms;
        sycl::local_accessor<uint32_t, 1> s_warpValueHistograms;
        sycl::local_accessor<uint32_t, 1> s_localHistogram;
        uint32_t *m_sortBuffer;
        uint32_t *m_sortAltBuffer;
        uint32_t *m_valuesBuffer;
        uint32_t *m_valuesAltBuffer;
        uint32_t *m_indexBuffer;
        uint32_t *m_passHist;
        uint32_t m_radixShift = 0;
        uint32_t m_size = 0;
    public:


        void operator()(sycl::nd_item<1> item) const {


            auto subGroup = item.get_sub_group();
            uint32_t blockDim = item.get_local_range(0);
            uint32_t gridDim = item.get_group_range(0);
            uint32_t threadIdx = item.get_local_id(0);
            uint32_t laneId = subGroup.get_local_linear_id();
            uint32_t warpSize = subGroup.get_max_local_range()[0];

            uint32_t warpBlockIndex = threadIdx / warpSize;

            uint32_t *s_warpHist = &s_warpHistograms[warpBlockIndex << RADIX_LOG];
            // Clear shared memory
            for (uint32_t i = threadIdx; i < BIN_HISTS_SIZE; i += blockDim) {
                s_warpHistograms[i] = 0;
            }

            // Atomically assign partition tiles
            if (threadIdx == 0) {
                uint32_t globalId = item.get_global_id(0);
                s_warpHistograms[BIN_PART_SIZE -
                                 1] = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(
                        m_indexBuffer[m_radixShift >> 3]).fetch_add(1);
                //sycl::ext::oneapi::experimental::printf("GlobalID: %d, BlockID: %d, LocalID: %d\n", globalId, blockIdx, threadIdx);

            }
            item.barrier(sycl::access::fence_space::local_space);

            const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];


            //To handle input sizes not perfect multiples of the partition tile size
            if (partitionIndex < gridDim - 1) {


                uint32_t keys[BIN_KEYS_PER_THREAD];
                uint32_t values[BIN_KEYS_PER_THREAD];

                uint32_t binSubPartStart = warpBlockIndex * BIN_SUB_PART_SIZE;
                uint32_t binPartStart = partitionIndex * BIN_PART_SIZE;

                for (uint32_t i = 0, t = laneId + binSubPartStart + binPartStart;
                     i < BIN_KEYS_PER_THREAD; ++i, t += warpSize) {
                    keys[i] = m_sortBuffer[t];
                    values[i] = m_valuesBuffer[t];
                }

                uint16_t offsets[BIN_KEYS_PER_THREAD];
                for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
                    unsigned warpFlags = 0xffffffff;
                    for (uint32_t k = 0; k < RADIX_LOG; ++k) {
                        const bool t = (keys[i] >> (k + m_radixShift)) & 1;
                        uint32_t t_mask = t ? 0 : 0xffffffff;
                        uint32_t mask;
                        sycl::ext::oneapi::group_ballot(subGroup, t).extract_bits(mask);
                        warpFlags &= mask ^ t_mask;
                    }
                    const uint32_t bits = sycl::popcount(warpFlags & getLaneMaskLt(subGroup));
                    uint32_t preIncrementVal;
                    if (bits == 0) {
                        uint32_t bin = keys[i] >> m_radixShift & RADIX_MASK;
                        uint32_t val = sycl::popcount(warpFlags);

                        // Perform atomic add on local memory
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomicRef(
                                s_warpHist[bin]);
                        preIncrementVal = atomicRef.fetch_add(val);

                    }
                    uint32_t firstSetBit = warpFlags ? __builtin_ctz(warpFlags) : 0;
                    offsets[i] = sycl::select_from_group(subGroup, preIncrementVal, firstSetBit) + bits;

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
                            m_passHist[threadIdx + (partitionIndex + 1) * RADIX]);
                    atomicRef.fetch_add(FLAG_REDUCTION | reduction << 2);
                    for (int i = 1; i < warpSize; i <<= 1) {
                        uint32_t t = sycl::shift_group_right(subGroup, reduction, i);
                        if (laneId >= i) {// Dont add on our first lane
                            reduction += t;
                        }
                    }
                    uint32_t circular_val;
                    int last_value = sycl::permute_group_by_xor(subGroup, reduction,
                                                                subGroup.get_max_local_range()[0] - 1);
                    reduction = sycl::shift_group_right(subGroup, reduction, 1);
                    if (laneId == 0) {
                        circular_val = last_value;
                    } else {
                        circular_val = reduction;
                    }
                    s_localHistogram[threadIdx] = circular_val;
                }

                item.barrier(sycl::access::fence_space::local_space);
                // Active Exclusive Warp scan
                uint32_t idx = threadIdx * warpSize;

                uint32_t reduce;
                reduce = idx < RADIX ? s_localHistogram[idx] : 0;
                for (int i = 1; i <= 16; i <<= 1) { // 16 = LANE_COUNT >> 1
                    uint32_t t = sycl::shift_group_right(subGroup, reduce, i);
                    if (laneId >= i)
                        reduce += t;
                }
                uint32_t exclude_reduce = sycl::shift_group_right(subGroup, reduce, 1);
                uint32_t val = laneId ? exclude_reduce : 0;

                if (idx < RADIX) {
                    s_localHistogram[idx] = val;
                }

                item.barrier(sycl::access::fence_space::local_space);

                //uint32_t res = (laneId) ? sycl::select_from_group(subGroup, s_localHistogram[threadIdx - 1], 1) : 0;
                uint32_t res = 0;

                if (laneId) {
                    res = sycl::select_from_group(subGroup, s_localHistogram[threadIdx - 1], 1);
                } else {
                    sycl::select_from_group(subGroup, 0, 0);
                }

                // shuffl sync
                if (threadIdx < RADIX) {
                    s_localHistogram[threadIdx] += res;
                }
                item.barrier(sycl::access::fence_space::local_space);


                //update offsets

                if (warpBlockIndex) {
                    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
                        const uint32_t t2 = (keys[i] >> m_radixShift) & RADIX_MASK;
                        offsets[i] += s_warpHist[t2] + s_localHistogram[t2];

                    }
                } else {
                    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
                        offsets[i] += s_localHistogram[(keys[i] >> m_radixShift) & RADIX_MASK];
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
                    s_warpHistograms[offsets[i]] = keys[i];
                    s_warpValueHistograms[offsets[i]] = values[i];
                }

                if (threadIdx < RADIX) {
                    uint32_t reduction = 0;
                    for (uint32_t k = partitionIndex; k >= 0;) {
                        const uint32_t flagPayload = m_passHist[threadIdx + k * RADIX];

                        if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                            reduction += flagPayload >> 2;
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomicRef(
                                    m_passHist[threadIdx + (partitionIndex + 1) * RADIX]);
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
                    uint32_t altBufferIndex = s_localHistogram[s_warpHistograms[i] >> m_radixShift & RADIX_MASK] + i;

                    m_sortAltBuffer[altBufferIndex] = s_warpHistograms[i];
                    m_valuesAltBuffer[altBufferIndex] = s_warpValueHistograms[i];
                }


            }

            if (partitionIndex == gridDim - 1) {
                // Immediately begin lookback
                if (threadIdx < RADIX) {
                    if (partitionIndex) {
                        uint32_t reduction = 0;
                        for (uint32_t k = partitionIndex; k >= 0;) {
                            const uint32_t flagPayload = m_passHist[threadIdx + k * RADIX];

                            //sycl::ext::oneapi::experimental::printf(
                            //        "threadIdx: %d, k: %u, Partition: %u, index: %u, payload: %u\n",
                            //        threadIdx, k, partitionIndex, threadIdx + k * RADIX,
                            //        flagPayload);


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
                        s_localHistogram[threadIdx] = m_passHist[threadIdx] >> 2;
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                const uint32_t partEnd = partitionIndex * BIN_PART_SIZE + BIN_PART_SIZE;
                const uint32_t partStart = threadIdx + partitionIndex * BIN_PART_SIZE;
                //sycl::ext::oneapi::experimental::printf("threadIdx: %d, Start: %u, stop: %u, increment: %u\n",
                //                                        threadIdx, threadIdx + partitionIndex * BIN_PART_SIZE, partEnd,
                //                                        blockDim);

                for (uint32_t i = partStart; i < partEnd; i += blockDim) {
                    uint32_t key, value;
                    uint32_t offset;
                    uint32_t warpFlags = 0xffffffff;
                    if (i < m_size) {
                        key = m_sortBuffer[i];
                        value = m_valuesBuffer[i];
                    }
                    for (uint32_t k = 0; k < RADIX_LOG; ++k) {
                        const bool t = (key >> (k + m_radixShift)) & 1;
                        uint32_t t_mask = t ? 0 : 0xffffffff;
                        uint32_t mask;
                        sycl::ext::oneapi::group_ballot(item.get_sub_group(), t).extract_bits(mask);
                        warpFlags &= mask ^ t_mask;
                    }
                    const uint32_t bits = sycl::popcount(warpFlags & getLaneMaskLt(subGroup));

                    uint32_t dummy;
                    uint32_t preIncrementVal;
                    uint32_t warpBlockIndex = threadIdx / warpSize;

                    for (uint32_t k = 0; k < BIN_WARPS; ++k) {
                        bool increment = warpBlockIndex == k && bits == 0 &&
                                         i < m_size;
                        if (increment) {
                            uint32_t bin = key >> m_radixShift & RADIX_MASK;
                            uint32_t val = sycl::popcount(warpFlags);
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> atomicRef(
                                    s_localHistogram[bin]);
                            preIncrementVal = atomicRef.fetch_add(val);

                        }

                        uint32_t firstSetBit = warpFlags ? __builtin_ctz(warpFlags) : 0;

                        dummy = sycl::select_from_group(item.get_sub_group(), preIncrementVal,
                                                        firstSetBit);
                        if (warpBlockIndex == k) {
                            offset = dummy + bits;

                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (i < m_size) {
                        m_sortAltBuffer[offset] = key;
                        m_valuesAltBuffer[offset] = value;
                    }
                }
            }
        }
    };

}
#endif //MULTISENSE_VIEWER_KERNELS_H
