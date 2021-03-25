// Copyright (C) 2011  Carl Rogers
// Copyright 2021 The WebNN-native Authors
//
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php
// Copyright 2021 The WebNN-native Authors
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LIBCNPY_H_
#define LIBCNPY_H_
#define SHARED_DATA_TYPE std::shared_ptr<std::vector<char>>

#include <stdint.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

    struct NpyArray {
        NpyArray(const std::vector<int32_t>& _shape, size_t _word_size, bool _fortran_order)
            : shape(_shape), word_size(_word_size), fortran_order(_fortran_order) {
            num_vals = 1;
            for (size_t i = 0; i < shape.size(); i++)
                num_vals *= shape[i];
            data_holder = SHARED_DATA_TYPE(new std::vector<char>(num_vals * word_size));
        }

        NpyArray() : shape(0), word_size(0), fortran_order(0), num_vals(0) {
        }

        template <typename T>
        T* data() {
            return reinterpret_cast<T*>(&(*data_holder)[0]);
        }

        template <typename T>
        const T* data() const {
            return reinterpret_cast<T*>(&(*data_holder)[0]);
        }

        template <typename T>
        std::vector<T> as_vec() const {
            const T* p = data<T>();
            return std::vector<T>(p, p + num_vals);
        }

        size_t num_bytes() const {
            return data_holder->size();
        }

        SHARED_DATA_TYPE data_holder;
        std::vector<int32_t> shape;
        size_t word_size;
        bool fortran_order;
        size_t num_vals;
    };

    void parse_npy_header(FILE* fp,
                          size_t& word_size,
                          std::vector<int32_t>& shape,
                          bool& fortran_order);
    NpyArray npy_load(std::string fname);
}  // namespace cnpy

#endif
