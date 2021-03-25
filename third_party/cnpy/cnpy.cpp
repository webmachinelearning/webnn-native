// Copyright (C) 2011  Carl Rogers
// Copyright 2021 The WebNN-native Authors
//
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php
// Copyright 2021 The WebNN-native Authors
//
// SPDX-License-Identifier: Apache-2.0
//

#include "cnpy.h"
#include <stdint.h>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <regex>
#include <stdexcept>

void cnpy::parse_npy_header(FILE* fp,
                            size_t& word_size,
                            std::vector<int32_t>& shape,
                            bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11)
        std::cerr << "parse_npy_header: failed fread" << std::endl;
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        std::cerr << "parse_npy_header: failed to find header keyword: 'fortran_order" << std::endl;
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        std::cerr << "parse_npy_header: failed to find header keyword: '(' or ')'" << std::endl;
    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        std::cerr << "parse_npy_header: failed to find header keyword: 'descr'" << std::endl;
    loc1 += 9;

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

cnpy::NpyArray load_the_npy_file(FILE* fp) {
    std::vector<int32_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        std::cerr << "load_the_npy_file: failed fread" << std::endl;
    return arr;
}

cnpy::NpyArray cnpy::npy_load(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp)
        std::cerr << "npy_load: Unable to open file " << fname << std::endl;

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}
