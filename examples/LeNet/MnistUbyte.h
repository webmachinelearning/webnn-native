// Copyright (C) 2018-2020 Intel Corporation
// Copyright 2021 The WebNN-native Authors
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Mnist reader
 * \file MnistUbyte.h
 */
#pragma once

#include <memory>
#include <string>

/**
 * \class MnistUbyte
 * \brief Reader for mnist db files
 */
class MnistUbyte {
  private:
    int ReverseInt(int i);

  public:
    /**
     * \brief Constructor of Mnist reader
     * @param filename - path to input data
     * @return MnistUbyte reader object
     */
    explicit MnistUbyte(const std::string& filename);
    virtual ~MnistUbyte() {
    }

    /**
     * \brief Get size
     * @return size
     */
    size_t Size() const {
        return mWidth * mHeight * 1;
    }

    std::shared_ptr<unsigned char> GetData() {
        return mData;
    }

    bool DataInitialized() {
        return mDataInitialized;
    }

  private:
    /// \brief height
    size_t mHeight = 0;
    /// \brief width
    size_t mWidth = 0;
    /// \brief data
    std::shared_ptr<unsigned char> mData;
    bool mDataInitialized = false;
};
