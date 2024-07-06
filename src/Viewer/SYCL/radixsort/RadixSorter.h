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


    public:
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


//Test for correctness
        void validationTest(std::vector<uint32_t> &keys, uint32_t testIterations) {
            printf("Beginning VALIDATION tests at size %u and %u iterations. \n", keys.size(), testIterations);
            int testsPassed = 0;
            for (uint32_t i = 1; i <= testIterations; ++i) {
                bool isCorrect = true;
                for (uint32_t k = 1; k < keys.size(); ++k) {
                    if (keys[k] < keys[k - 1]) {
                        isCorrect = false;
                        printf("Failed at: %u, keys: %u, %u,\n", k, keys[k], keys[k - 1]);

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


    public:
        void performOneSweep(uint32_t *sortBuffer, uint32_t *valuesBuffer, uint32_t numRendered) {
            binningThreadblocks = (numRendered + partitionSize - 1) / partitionSize;
            globalHistThreadblocks = (numRendered + globalHistPartitionSize - 1) / globalHistPartitionSize;

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


            /*

            std::vector<uint32_t> globalHist(radixPasses * radix);
            // Copy to host and view the keys
            queue.memcpy(globalHist.data(), globalHistogramBuffer, sizeof(uint32_t) * globalHist.size()).wait();
            std::cout << "globalHist:  ";
            for (size_t i = 0; i < globalHist.size(); ++i) {
                std::cout << std::setw(4) << globalHist[i] << " ";
            }                // sort on device
            std::cout << std::endl;

            */

            /// SCAN PASS
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


            /*
            std::vector<uint32_t> firstPassHist(RADIX * binningThreadblocks);
            // Copy to host and view the keys
            queue.memcpy(firstPassHist.data(), firstPassHistogramBuffer,
                         sizeof(uint32_t) * firstPassHist.size()).wait();
            std::cout << "firstPassHist(" << firstPassHist.size() << "): ";
            for (unsigned int i: firstPassHist) {
                std::cout << std::setw(4) << i << " ";
            }
            std::cout << std::endl;
            std::vector<uint32_t> secPassHist(RADIX * binningThreadblocks);
            // Copy to host and view the keys
            queue.memcpy(secPassHist.data(), secPassHistogramBuffer,
                         sizeof(uint32_t) * secPassHist.size()).wait();
            std::cout << "secPassHist(" << sizeof(uint32_t) * secPassHist.size() << "): ";
            for (unsigned int i: secPassHist) {
                std::cout << std::setw(4) << i << " ";
            }
            std::cout << std::endl;
*/


            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));
                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_localHistogram,
                                                      sortBuffer, sortAltBuffer, valuesBuffer,
                                                      valuesAltBuffer,
                                                      indexBuffer, firstPassHistogramBuffer, 0, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });

            }).wait();

            //queue.memcpy(sortBuffer, sortAltBuffer, m_size * sizeof(uint32_t));


            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));

                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_localHistogram,
                                                      sortAltBuffer, sortBuffer, valuesAltBuffer,
                                                      valuesBuffer,
                                                      indexBuffer, secPassHistogramBuffer, 8, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });

            }).wait();
            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));

                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_localHistogram,
                                                      sortBuffer, sortAltBuffer, valuesBuffer,
                                                      valuesAltBuffer,
                                                      indexBuffer, thirdPassHistogramBuffer, 16, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });

            }).wait();
            queue.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> s_warpHistograms(sycl::range<1>(BIN_PART_SIZE), h);
                sycl::local_accessor<uint32_t, 1> s_localHistogram(sycl::range<1>(RADIX), h);
                auto range = sycl::nd_range<1>(sycl::range<1>(binningThreads * binningThreadblocks),
                                               sycl::range<1>(binningThreads));

                RadixSorter::DigitBinningPass functor(s_warpHistograms,
                                                      s_localHistogram,
                                                      sortAltBuffer, sortBuffer, valuesAltBuffer,
                                                      valuesBuffer,
                                                      indexBuffer, fourthPassHistogramBuffer, 24, numRendered);
                h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    functor(item);
                });
            }).wait();

        }

    };
}

#endif //MULTISENSE_VIEWER_RADIXSORTER_H
