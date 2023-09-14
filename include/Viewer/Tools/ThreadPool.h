/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/ThreadPool.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-09-27, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_THREADPOOL_H
#define MULTISENSE_VIEWER_THREADPOOL_H

#include <atomic>
#include <cassert>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace VkRender {
    /**
     * @brief Implementation of a simple thread poll in c++
     * Source: https://maidamai0.github.io/post/a-simple-thread-pool/
     * based on and highly inspired by Easy3D
     * License: ?
     */
    class ThreadPool {
        using task_type = std::function<void()>;
        std::atomic<bool> stopFlag{false}; // Flag to signal threads to stop

    public:
        /**
         * Initialize with number of threads in the threadpool
         * @param number of threads in threadpool
         */
        explicit ThreadPool(size_t num = std::thread::hardware_concurrency()) {
            for (size_t i = 0; i < num; ++i) {
                workers_.emplace_back(std::thread([this] {
                    while (true) {
                        task_type task;
                        // If the stop flag is set, return from the thread i.e. skip execution
                        if (stopFlag.load()) {
                            pushStopTask();
                            return;
                        }

                        {
                            std::unique_lock<std::mutex> lock(task_mutex_);
                            task_cond_.wait(lock, [this] { return !tasks_.empty(); });


                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        if (!task) {
                            pushStopTask();
                            return;
                        }
                        task();
                    }
                }));
            }
        }


        ~ThreadPool() {
            Stop();
        }

        void signalStop(){
            stopFlag.store(false);
        }

        void Stop() {
            stopFlag.store(true); // Set the stop flag to true
            pushStopTask();
            for (auto &worker: workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }

            // clear all pending tasks
            std::queue<task_type> empty{};
            std::swap(tasks_, empty);

        }

        /**
         * Push task to queue list
         * @tparam F
         * @tparam Args
         * @param f
         * @param args
         * @return
         */
        template<typename F, typename... Args>
        auto Push(F &&f, Args &&... args) {
            using return_type = std::invoke_result_t<F, Args...>;
            auto task
                    = std::make_shared<std::packaged_task<return_type()>>(
                            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            auto res = task->get_future();

            {
                std::lock_guard<std::mutex> lock(task_mutex_);
                tasks_.emplace([task] { (*task)(); });
            }
            task_cond_.notify_one();


            return res;
        }

        /**
         * Get the number of tasks in queue
         * @return number of tasks
         */
        size_t getTaskListSize() {
            return tasks_.size();
        }

        void pushStopTask() {
            std::lock_guard<std::mutex> lock(task_mutex_);
            tasks_.push(task_type{});
            task_cond_.notify_one();
        }

    private:

        std::vector<std::thread> workers_;
        std::queue<task_type> tasks_;
        std::mutex task_mutex_;
        std::condition_variable task_cond_;
    };
}
#endif //MULTISENSE_VIEWER_THREADPOOL_H
