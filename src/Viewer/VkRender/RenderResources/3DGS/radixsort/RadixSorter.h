//
// Created by magnus on 6/13/24.
//

#ifndef MULTISENSE_VIEWER_RADIXSORTER_H
#define MULTISENSE_VIEWER_RADIXSORTER_H


#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#include "Kernels.h"

#include "Viewer/Tools/Logger.h"

namespace VkRender {

    class Sorter {

    public:
        Sorter(sycl::queue &_queue, uint32_t size) : queue(_queue) {
            binningThreadblocks = (size + partitionSize - 1) / partitionSize;
            globalHistThreadblocks = (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
            sortAltBuffer = sycl::malloc_device<uint32_t>(size, queue);
            valuesAltBuffer = sycl::malloc_device<uint32_t>(size, queue);

            globalHistogramBuffer = sycl::malloc_device<uint32_t>(RADIX * radixPasses, queue);
            firstPassHistogramBuffer = sycl::malloc_device<uint32_t>(RADIX * binningThreadblocks, queue);
            secPassHistogramBuffer = sycl::malloc_device<uint32_t>(RADIX * binningThreadblocks, queue);
            thirdPassHistogramBuffer = sycl::malloc_device<uint32_t>(RADIX * binningThreadblocks, queue);
            fourthPassHistogramBuffer = sycl::malloc_device<uint32_t>(RADIX * binningThreadblocks, queue);
            indexBuffer = sycl::malloc_device<uint32_t>(radixPasses, queue);

            // Fill the allocated memory with zeros
            queue.fill(globalHistogramBuffer, 0x00, RADIX * radixPasses).wait();
            queue.fill(firstPassHistogramBuffer, 0x00, RADIX * binningThreadblocks).wait();
            queue.fill(secPassHistogramBuffer, 0x00, RADIX * binningThreadblocks).wait();
            queue.fill(thirdPassHistogramBuffer, 0x00, RADIX * binningThreadblocks).wait();
            queue.fill(fourthPassHistogramBuffer, 0x00, RADIX * binningThreadblocks).wait();
            queue.fill(indexBuffer, 0x00, radixPasses).wait();

        }

        ~Sorter() {
            sycl::free(sortAltBuffer, queue);
            sycl::free(valuesAltBuffer, queue);

            sycl::free(globalHistogramBuffer, queue);
            sycl::free(firstPassHistogramBuffer, queue);
            sycl::free(secPassHistogramBuffer, queue);
            sycl::free(thirdPassHistogramBuffer, queue);
            sycl::free(fourthPassHistogramBuffer, queue);
            sycl::free(indexBuffer, queue);
        }


    private:
        const uint32_t radix = 256;
        const uint32_t radixPasses = 4;
        const uint32_t partitionSize = 7680;
        const uint32_t globalHistPartitionSize = 65536;
        const uint32_t globalHistThreads = 128;
        const uint32_t binningThreads = 256;            //2080 super seems to really like 512
        uint32_t binningThreadblocks = 0;
        uint32_t globalHistThreadblocks = 0;

        uint32_t *sortAltBuffer = nullptr;
        uint32_t *valuesAltBuffer = nullptr;
        uint32_t *globalHistogramBuffer = nullptr;
        uint32_t *firstPassHistogramBuffer = nullptr;
        uint32_t *secPassHistogramBuffer = nullptr;
        uint32_t *thirdPassHistogramBuffer = nullptr;
        uint32_t *fourthPassHistogramBuffer = nullptr;
        uint32_t *indexBuffer = nullptr;
        sycl::queue &queue;
        sycl::event resetEvent;

        void printKeys(const std::vector<uint32_t> &key, uint32_t num = 128) {
            // Print the keys
            std::cout << "Key:   ";
            for (size_t i = 0; i < key.size(); ++i) {
                std::cout << std::setw(4) << key[i] << " ";
                if (i >= num - 1 || i >= key.size()) {
                    break;
                }
            }
            std::cout << std::endl;
        }

//Test for correctness
        void validationTest(std::vector<uint32_t> &keys, uint32_t testIterations) {
            //printf("Beginning VALIDATION tests at size %u and %u iterations. \n", keys.size(), testIterations);
            int testsPassed = 0;
            uint32_t failIndex = 1;
            bool isCorrect = true;

            for (uint32_t i = 1; i <= testIterations; ++i) {
                isCorrect = true;
                for (uint32_t k = 1; k < keys.size(); ++k) {
                    if (keys[k] < keys[k - 1]) {
                        isCorrect = false;
                        failIndex = k;
                        break;
                    }
                }
                if (isCorrect)
                    testsPassed++;
                else
                    printf("Test iteration %d failed. \n", i);
            }
            Log::Logger::getInstance()->trace("Sorting: {}/{} tests passed.", testsPassed, testIterations);
            if (!isCorrect)
                Log::Logger::getInstance()->trace("Failed sorting at index: {}. Key: {} < {} (key < key - 1)", failIndex - 1,
                                                  keys[failIndex], keys[failIndex - 1]);
        }


    public:

        void resetMemory() {
            // Fill the allocated memory with zeros
            queue.fill(globalHistogramBuffer, 0x00, RADIX * radixPasses);
            queue.fill(firstPassHistogramBuffer, 0x00, RADIX * binningThreadblocks);
            queue.fill(secPassHistogramBuffer, 0x00, RADIX * binningThreadblocks);
            queue.fill(thirdPassHistogramBuffer, 0x00, RADIX * binningThreadblocks);
            queue.fill(fourthPassHistogramBuffer, 0x00, RADIX * binningThreadblocks);
            queue.fill(indexBuffer, 0x00, radixPasses);
        }

        void verifySort(uint32_t *keysDevice, uint32_t size, bool print = false, std::vector<uint32_t> cpuKeys = {}) {
            // copy keys back

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<uint32_t> keys(size);
            queue.wait();
            queue.memcpy(keys.data(), keysDevice, size * sizeof(uint32_t)).wait();

            validationTest(keys, 5);
            if(print) {
                printKeys(keys, keys.size());
                std::sort(cpuKeys.begin(), cpuKeys.end());

                // Compare the sorted cpuKeys with the keys copied back from the device
                bool identical = std::equal(keys.begin(), keys.end(), cpuKeys.begin());
                if (identical) {
                    std::cout << "The arrays are identical.\n";
                } else {
                    std::cout << "The arrays are not identical.\n";
                }
            }
            std::chrono::duration<double, std::milli> verificationDuration =
                    std::chrono::high_resolution_clock::now() - start;
            Log::Logger::getInstance()->trace("3DGS Sorting: verification duration: {}", verificationDuration.count());
        }

        void performOneSweep(uint32_t *sortBuffer, uint32_t *valuesBuffer, uint32_t numRendered) {
            if (numRendered > (1 << 25))
                return;
            binningThreadblocks = (numRendered + partitionSize - 1) / partitionSize;
            globalHistThreadblocks = (numRendered + globalHistPartitionSize - 1) / globalHistPartitionSize;

            auto startGlobalHist = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                // Shared memory allocations
                sycl::local_accessor<uint32_t, 1> s_globalHistFirst(sycl::range<1>(RADIX * 2), h);
                sycl::local_accessor<uint32_t, 1> s_globalHistSec(sycl::range<1>(RADIX * 2), h);
                sycl::local_accessor<uint32_t, 1> s_globalHistThird(sycl::range<1>(RADIX * 2), h);
                sycl::local_accessor<uint32_t, 1> s_globalHistFourth(sycl::range<1>(RADIX * 2), h);

                auto range = sycl::nd_range<1>(sycl::range<1>(globalHistThreads * globalHistThreadblocks),
                                               sycl::range<1>(globalHistThreads));
                // Kernel code
                h.parallel_for(range, RadixSorter::GlobalHistogram(s_globalHistFirst,
                                                                   s_globalHistSec,
                                                                   s_globalHistThird,
                                                                   s_globalHistFourth,
                                                                   sortBuffer,
                                                                   globalHistogramBuffer,
                                                                   numRendered));
            }).wait();

            std::chrono::duration<double, std::milli> globalHist =
                    std::chrono::high_resolution_clock::now() - startGlobalHist;
            /// SCAN PASS
            auto startScanPass = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_scan(sycl::range<1>(RADIX * 2), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(radix * radixPasses), sycl::range<1>(radix));
                RadixSorter::ScanPass functor(s_scan, globalHistogramBuffer, firstPassHistogramBuffer,
                                              secPassHistogramBuffer, thirdPassHistogramBuffer,
                                              fourthPassHistogramBuffer);

                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });

            }).wait();


            std::chrono::duration<double, std::milli> scanPass =
                    std::chrono::high_resolution_clock::now() - startScanPass;

            auto digitPassOne = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_warpValueHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));
                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_warpValueHistograms,
                                                      s_localHistogram,
                                                      sortBuffer, sortAltBuffer, valuesBuffer,
                                                      valuesAltBuffer,
                                                      indexBuffer, firstPassHistogramBuffer, 0, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });

            }).wait();
            auto digitPassTwo = std::chrono::high_resolution_clock::now();



            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_warpValueHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));
                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_warpValueHistograms,
                                                      s_localHistogram,
                                                      sortAltBuffer, sortBuffer, valuesAltBuffer,
                                                      valuesBuffer,
                                                      indexBuffer, secPassHistogramBuffer, 8, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });
            }).wait();

            auto digitPassThree = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_warpValueHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));
                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_warpValueHistograms,
                                                      s_localHistogram,
                                                      sortBuffer, sortAltBuffer, valuesBuffer,
                                                      valuesAltBuffer,
                                                      indexBuffer, thirdPassHistogramBuffer, 16, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });
            }).wait();

            auto digitPassFour = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_warpValueHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));
                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_warpValueHistograms,
                                                      s_localHistogram,
                                                      sortAltBuffer, sortBuffer, valuesAltBuffer,
                                                      valuesBuffer,
                                                      indexBuffer, fourthPassHistogramBuffer, 24, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });
            }).wait();


            std::chrono::duration<double, std::milli> digitPassOneDuration =
                    digitPassTwo - digitPassOne;
            std::chrono::duration<double, std::milli> digitPassTwoDuration =
                    digitPassThree - digitPassTwo;
            std::chrono::duration<double, std::milli> digitPassThreeDuration =
                    digitPassFour- digitPassThree;
            std::chrono::duration<double, std::milli> digitPassFourDuration =
                    std::chrono::high_resolution_clock::now() - digitPassFour;

            Log::Logger::getInstance()->trace("3DGS Sorting: GlobalHistoGram: {}", globalHist.count());
            Log::Logger::getInstance()->trace("3DGS Sorting: ScanPass: {}", scanPass.count());
            Log::Logger::getInstance()->trace("3DGS Sorting: DigitBinningPass1: {}", digitPassOneDuration.count());
            Log::Logger::getInstance()->trace("3DGS Sorting: DigitBinningPass2: {}", digitPassTwoDuration.count());
            Log::Logger::getInstance()->trace("3DGS Sorting: DigitBinningPass3: {}", digitPassThreeDuration.count());
            Log::Logger::getInstance()->trace("3DGS Sorting: DigitBinningPass4: {}", digitPassFourDuration.count());


        }

    };
}

#endif //MULTISENSE_VIEWER_RADIXSORTER_H
