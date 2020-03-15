/*
 * Copyright (c) 2017 Manzil Zaheer All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _UTILS_H
#define _UTILS_H

#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <atomic>
#include <thread>
#include <future>

#include <Eigen/Core>

#ifdef _MSC_VER

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#endif

namespace utils
{
    static inline void pause()
    {
        // Only use this function if a human is involved!
        std::cout << "Press any key to continue..." << std::flush;
        std::cin.get();
    }

    template<class InputIt, class UnaryFunction>
    UnaryFunction parallel_for_each(InputIt first, InputIt last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

        unsigned cores = std::thread::hardware_concurrency();

        auto task = [&f](InputIt start, InputIt end)->void{
            for (; start < end; ++start)
                f(*start);
        };

        const size_t total_length = std::distance(first, last);
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));
        InputIt chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = std::next(chunk_start, chunk_length);
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for(size_t first, size_t last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

        unsigned cores = std::thread::hardware_concurrency();

        auto task = [&f](size_t start, size_t end)->void{
            for (; start < end; ++start)
                f(start);
        };

        const size_t total_length = last - first;
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));
        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }

    static inline void progressbar(unsigned int x, unsigned int n, unsigned int w = 50){
        if ( (x != n) && (x % (n/10+1) != 0) ) return;

        float ratio =  x/(float)n;
        unsigned c = ratio * w;

        std::cout << std::setw(3) << (int)(ratio*100) << "% [";
        for (unsigned x=0; x<c; x++) std::cout << "=";
        for (unsigned x=c; x<w; x++) std::cout << " ";
        std::cout << "]\r" << std::flush;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for_progressbar(size_t first, size_t last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

        unsigned cores = std::thread::hardware_concurrency();
        const size_t total_length = last - first;
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));

        auto task = [&f,&chunk_length](size_t start, size_t end)->void{
            for (; start < end; ++start){
                progressbar(start%chunk_length, chunk_length);
                f(start);
            }
        };

        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        progressbar(chunk_length, chunk_length);
        std::cout << std::endl;
        return f;
    }

    template<typename T>
    void add_to_atomic(std::atomic<T>& foo, T& bar)
    {
        auto current = foo.load();
        while (!foo.compare_exchange_weak(current, current + bar));
    }

    class ParallelAddList
    {
        int left;
        int right;
        Eigen::VectorXd res;
        std::vector<Eigen::VectorXd>& pList;


        void run()
        {
            res = Eigen::VectorXd::Zero(pList[0].size());
            for(int i = left; i<right; ++i)
                res += pList[i];
        }

    public:
        explicit ParallelAddList(std::vector<Eigen::VectorXd>& pL) : pList(pL)
        {
            this->left = 0;
            this->right = pL.size();
            compute();
        }
        ParallelAddList(int left, int right, std::vector<Eigen::VectorXd>& pL) : pList(pL)
        {
            this->left = left;
            this->right = right;
        }

        ~ParallelAddList()
        {   }

        int compute()
        {
            if (right - left < 500000)
            {
                run();
                return 0;
            }

            int split = (right - left) / 2;

            ParallelAddList* t1 = new ParallelAddList(left, left + split, pList);
            ParallelAddList* t2 = new ParallelAddList(left + split, right, pList);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddList::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddList::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        Eigen::VectorXd get_result()
        {
            return res;
        }
    };

    class ParallelAddMatrix
    {
        int left;
        int right;
        Eigen::VectorXd res;
        Eigen::MatrixXd& pMatrix;


        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.rows());
            for(int i = left; i<right; ++i)
                res += pMatrix.col(i);
        }

    public:
        explicit ParallelAddMatrix(Eigen::MatrixXd& pM) : pMatrix(pM)
        {
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelAddMatrix(int left, int right, Eigen::MatrixXd& pM) : pMatrix(pM)
        {
            this->left = left;
            this->right = right;
        }

        ~ParallelAddMatrix()
        {   }

        int compute()
        {
            if (right - left < 500000)
            {
                run();
                return 0;
            }

            int split = (right - left) / 2;

            ParallelAddMatrix* t1 = new ParallelAddMatrix(left, left + split, pMatrix);
            ParallelAddMatrix* t2 = new ParallelAddMatrix(left + split, right, pMatrix);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddMatrix::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddMatrix::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        Eigen::VectorXd get_result()
        {
            return res;
        }
    };

    class ParallelAddMatrixNP
    {
        int left;
        int right;
        Eigen::VectorXd res;
        Eigen::Map<Eigen::MatrixXd>& pMatrix;


        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.rows());
            for(int i = left; i<right; ++i)
                res += pMatrix.col(i);
        }

    public:
        explicit ParallelAddMatrixNP(Eigen::Map<Eigen::MatrixXd>& pM) : pMatrix(pM)
        {
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelAddMatrixNP(int left, int right, Eigen::Map<Eigen::MatrixXd>& pM) : pMatrix(pM)
        {
            this->left = left;
            this->right = right;
        }

        ~ParallelAddMatrixNP()
        {   }

        int compute()
        {
            if (right - left < 500000)
            {
                run();
                return 0;
            }

            int split = (right - left) / 2;

            ParallelAddMatrixNP* t1 = new ParallelAddMatrixNP(left, left + split, pMatrix);
            ParallelAddMatrixNP* t2 = new ParallelAddMatrixNP(left + split, right, pMatrix);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddMatrixNP::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddMatrixNP::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        Eigen::VectorXd get_result()
        {
            return res;
        }
    };

    class ParallelDistanceComputeList
    {
        int left;
        int right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        std::vector<Eigen::VectorXd>& pList;


        void run()
        {
            res = Eigen::VectorXd::Zero(pList.size());
            for(int i = left; i<right; ++i)
                res[i] = (pList[i]-vec).norm();
        }

    public:
        ParallelDistanceComputeList(std::vector<Eigen::VectorXd>& pL, Eigen::VectorXd& v) : vec(v), pList(pL)
        {
            this->left = 0;
            this->right = pL.size();
            compute();
        }
        ParallelDistanceComputeList(int left, int right, std::vector<Eigen::VectorXd>& pL, Eigen::VectorXd& v) : vec(v), pList(pL)
        {
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceComputeList()
        {   }

        int compute()
        {
            if (right - left < 10000)
            {
                run();
                return 0;
            }

            int split = (right - left) / 2;

            ParallelDistanceComputeList* t1 = new ParallelDistanceComputeList(left, left + split, pList, vec);
            ParallelDistanceComputeList* t2 = new ParallelDistanceComputeList(left + split, right, pList, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceComputeList::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceComputeList::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        Eigen::VectorXd get_result()
        {
            return res;
        }
    };

    class ParallelDistanceCompute
    {
        int left;
        int right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        Eigen::MatrixXd& pMatrix;


        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.cols());
            for(int i = left; i<right; ++i)
                res[i] = (pMatrix.col(i)-vec).norm();
        }

    public:
        ParallelDistanceCompute(Eigen::MatrixXd& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelDistanceCompute(int left, int right, Eigen::MatrixXd& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceCompute()
        {   }

        int compute()
        {
            if (right - left < 10000)
            {
                run();
                return 0;
            }

            int split = (right - left) / 2;

            ParallelDistanceCompute* t1 = new ParallelDistanceCompute(left, left + split, pMatrix, vec);
            ParallelDistanceCompute* t2 = new ParallelDistanceCompute(left + split, right, pMatrix, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceCompute::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceCompute::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        Eigen::VectorXd get_result()
        {
            return res;
        }
    };

    class ParallelDistanceComputeNP
    {
        int left;
        int right;
        Eigen::VectorXd res;
        Eigen::VectorXd& vec;
        Eigen::Map<Eigen::MatrixXd>& pMatrix;


        void run()
        {
            res = Eigen::VectorXd::Zero(pMatrix.cols());
            for(int i = left; i<right; ++i)
                res[i] = (pMatrix.col(i)-vec).norm();
        }

    public:
        ParallelDistanceComputeNP(Eigen::Map<Eigen::MatrixXd>& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelDistanceComputeNP(int left, int right, Eigen::Map<Eigen::MatrixXd>& pM, Eigen::VectorXd& v) : vec(v), pMatrix(pM)
        {
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceComputeNP()
        {   }

        int compute()
        {
            if (right - left < 10000)
            {
                run();
                return 0;
            }

            int split = (right - left) / 2;

            ParallelDistanceComputeNP* t1 = new ParallelDistanceComputeNP(left, left + split, pMatrix, vec);
            ParallelDistanceComputeNP* t2 = new ParallelDistanceComputeNP(left + split, right, pMatrix, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceComputeNP::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceComputeNP::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        Eigen::VectorXd get_result()
        {
            return res;
        }
    };


    int read_wordmap(std::string wordmapfile, std::map<std::string, unsigned> * pword2id);
    int write_wordmap(std::string wordmapfile, std::map<std::string, unsigned> * pword2id);

}


#endif

