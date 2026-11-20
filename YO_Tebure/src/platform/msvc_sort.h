// Tebure - msvcsort::Sort: the exact algorithm of MSVC's std::sort (VS 2022 STL 14.44: introsort = insertion sort
// up to 32 elements, Tukey ninther pivot above 40, fat partition, heap sort after 1.5 log2 N divisions), so that a
// sort with TIES (equal keys, different payload) gives the same order with libc++ (macOS) as with the Windows
// build. std::sort leaves the order of equal elements unspecified and libc++ orders them differently.
// Used for the impact trigger list (tb_plan.cpp): triggers with equal tau but different strength / id are summed in
// list order, so their order changes the last bits of the shake. tests\sort_check proves Sort == MSVC std::sort
// (identical permutations) on random tie-heavy inputs of every size 0..600 and on adversarial patterns.
// Adapted from the Microsoft STL (<algorithm>, <__msvc_heap_algorithms.hpp>), Apache License v2.0 with LLVM
// Exception, Copyright (c) Microsoft Corporation. Licence text: third_party_licenses/MicrosoftSTL.txt
#pragma once
#include <algorithm>  // std::move_backward (libc++ does not declare it in <iterator> / <utility>)
#include <cstddef>
#include <iterator>
#include <utility>

namespace msvcsort {
namespace detail {

template <class It, class Ty, class Pr>
void PushHeapByIndex(It first, std::ptrdiff_t hole, std::ptrdiff_t top, Ty&& val, Pr pred)
{
    for (std::ptrdiff_t idx = (hole - 1) >> 1; top < hole && pred(*(first + idx), val); idx = (hole - 1) >> 1)
    {
        *(first + hole) = std::move(*(first + idx));
        hole = idx;
    }
    *(first + hole) = std::forward<Ty>(val);
}

template <class It, class Ty, class Pr>
void PopHeapHoleByIndex(It first, std::ptrdiff_t hole, std::ptrdiff_t bottom, Ty&& val, Pr pred)
{
    const std::ptrdiff_t top = hole;
    std::ptrdiff_t idx = hole;
    const std::ptrdiff_t maxNonLeaf = (bottom - 1) >> 1;
    while (idx < maxNonLeaf)
    {
        idx = 2 * idx + 2;
        if (pred(*(first + idx), *(first + (idx - 1))))
            --idx;
        *(first + hole) = std::move(*(first + idx));
        hole = idx;
    }
    if (idx == maxNonLeaf && bottom % 2 == 0)
    {
        *(first + hole) = std::move(*(first + (bottom - 1)));
        hole = bottom - 1;
    }
    PushHeapByIndex(first, hole, top, std::forward<Ty>(val), pred);
}

template <class It, class Pr>
void MakeHeap(It first, It last, Pr pred)
{
    const std::ptrdiff_t bottom = last - first;
    for (std::ptrdiff_t hole = bottom >> 1; hole > 0;)
    {
        --hole;
        auto val = std::move(*(first + hole));
        PopHeapHoleByIndex(first, hole, bottom, std::move(val), pred);
    }
}

template <class It, class Pr>
void SortHeap(It first, It last, Pr pred)
{
    for (; last - first >= 2; --last)
    {
        // _Pop_heap_unchecked(first, last)
        It l = last - 1;
        auto val = std::move(*l);
        *l = std::move(*first);
        PopHeapHoleByIndex(first, std::ptrdiff_t(0), (std::ptrdiff_t)(l - first), std::move(val), pred);
    }
}

template <class It, class Pr>
void InsertionSort(It first, It last, Pr pred)
{
    if (first == last)
        return;
    for (It mid = first; ++mid != last;)
    {
        It hole = mid;
        auto val = std::move(*mid);
        if (pred(val, *first))
        {
            std::move_backward(first, mid, ++hole);
            *first = std::move(val);
        }
        else
        {
            for (It prev = hole; pred(val, *--prev); hole = prev)
                *hole = std::move(*prev);
            *hole = std::move(val);
        }
    }
}

template <class It, class Pr>
void Med3(It first, It mid, It last, Pr pred)
{
    using std::swap;
    if (pred(*mid, *first))
        swap(*mid, *first);
    if (pred(*last, *mid))
    {
        swap(*last, *mid);
        if (pred(*mid, *first))
            swap(*mid, *first);
    }
}

template <class It, class Pr>
void GuessMedian(It first, It mid, It last, Pr pred)
{
    const std::ptrdiff_t count = last - first;
    if (40 < count)
    {
        const std::ptrdiff_t step = (count + 1) >> 3;
        const std::ptrdiff_t twoStep = step << 1;
        Med3(first, first + step, first + twoStep, pred);
        Med3(mid - step, mid, mid + step, pred);
        Med3(last - twoStep, last - step, last, pred);
        Med3(first + step, mid, last - step, pred);
    }
    else
        Med3(first, mid, last, pred);
}

template <class It, class Pr>
std::pair<It, It> PartitionByMedianGuess(It first, It last, Pr pred)
{
    using std::swap;
    It mid = first + ((last - first) >> 1);
    GuessMedian(first, mid, last - 1, pred);
    It pfirst = mid;
    It plast = pfirst + 1;
    while (first < pfirst && !pred(*(pfirst - 1), *pfirst) && !pred(*pfirst, *(pfirst - 1)))
        --pfirst;
    while (plast < last && !pred(*plast, *pfirst) && !pred(*pfirst, *plast))
        ++plast;
    It gfirst = plast;
    It glast = pfirst;
    for (;;)
    {
        for (; gfirst < last; ++gfirst)
        {
            if (pred(*pfirst, *gfirst))
                continue;
            else if (pred(*gfirst, *pfirst))
                break;
            else if (plast != gfirst)
            {
                swap(*plast, *gfirst);
                ++plast;
            }
            else
                ++plast;
        }
        for (; first < glast; --glast)
        {
            const It glastPrev = glast - 1;
            if (pred(*glastPrev, *pfirst))
                continue;
            else if (pred(*pfirst, *glastPrev))
                break;
            else if (--pfirst != glastPrev)
                swap(*pfirst, *glastPrev);
        }
        if (glast == first && gfirst == last)
            return std::pair<It, It>(pfirst, plast);
        if (glast == first)
        {
            if (plast != gfirst)
                swap(*pfirst, *plast);
            ++plast;
            swap(*pfirst, *gfirst);
            ++pfirst;
            ++gfirst;
        }
        else if (gfirst == last)
        {
            if (--glast != --pfirst)
                swap(*glast, *pfirst);
            swap(*pfirst, *--plast);
        }
        else
        {
            swap(*gfirst, *--glast);
            ++gfirst;
        }
    }
}

template <class It, class Pr>
void SortUnchecked(It first, It last, std::ptrdiff_t ideal, Pr pred)
{
    for (;;)
    {
        if (last - first <= 32)
        {
            InsertionSort(first, last, pred);
            return;
        }
        if (ideal <= 0)
        {
            MakeHeap(first, last, pred);
            SortHeap(first, last, pred);
            return;
        }
        auto mid = PartitionByMedianGuess(first, last, pred);
        ideal = (ideal >> 1) + (ideal >> 2);
        if (mid.first - first < last - mid.second)
        {
            SortUnchecked(first, mid.first, ideal, pred);
            first = mid.second;
        }
        else
        {
            SortUnchecked(mid.second, last, ideal, pred);
            last = mid.first;
        }
    }
}

}  // namespace detail

template <class It, class Pr>
void Sort(It first, It last, Pr pred)
{
    detail::SortUnchecked(first, last, last - first, pred);
}

}  // namespace msvcsort
