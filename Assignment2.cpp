
#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <cstdlib>

using namespace std;

// Tạo ngẫu nhiên hai mảng (cùng dữ liệu) trong đoạn [beginIdx, endIdx]
void fill_rand_arr(int primary[], int secondary[], int beginIdx, int endIdx) {
    for (int idx = beginIdx; idx <= endIdx; ++idx) {
        int val = rand();
        primary[idx] = val;
        secondary[idx] = val;
    }
}

// In mảng trong đoạn [beginIdx, endIdx]
void print_array_range(int arr[], int beginIdx, int endIdx) {
    for (int i = beginIdx; i <= endIdx; ++i) {
        cout << arr[i] << " ";
    }
    cout << "\n";
}

// Trộn hai dãy con đã được sắp xếp: [left, mid] và [mid+1, right]
void merge_segments(int arr[], int left, int mid, int right) {
    vector<int> leftArr(arr + left, arr + mid + 1);
    vector<int> rightArr(arr + mid + 1, arr + right + 1);

    int i = 0, j = 0;
    int writePos = left;

    while (i < static_cast<int>(leftArr.size()) && j < static_cast<int>(rightArr.size())) {
        if (leftArr[i] <= rightArr[j]) {
            arr[writePos++] = leftArr[i++];
        } else {
            arr[writePos++] = rightArr[j++];
        }
    }
    // Ghi nốt phần còn lại
    while (i < static_cast<int>(leftArr.size())) {
        arr[writePos++] = leftArr[i++];
    }
    while (j < static_cast<int>(rightArr.size())) {
        arr[writePos++] = rightArr[j++];
    }
}

// Merge Sort tuần tự
void merge_sort_sequential(int arr[], int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    merge_sort_sequential(arr, left, mid);
    merge_sort_sequential(arr, mid + 1, right);
    merge_segments(arr, left, mid, right);
}

// Merge Sort song song với OpenMP (giới hạn độ sâu để tránh overhead)
void merge_sort_parallel(int arr[], int left, int right, int level = 0) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;

    if (level < 4) {
        #pragma omp parallel sections
        {
            #pragma omp section
            merge_sort_parallel(arr, left, mid, level + 1);

            #pragma omp section
            merge_sort_parallel(arr, mid + 1, right, level + 1);
        }
    } else {
        merge_sort_parallel(arr, left, mid, level + 1);
        merge_sort_parallel(arr, mid + 1, right, level + 1);
    }

    merge_segments(arr, left, mid, right);
}

int main() {
    // Demo nhỏ
    int smallSeq[] = {20, 17, 6, 3, 8, 23, 18, 10, 22, 13, 4, 9, 31, 21, 88, 2};
    int smallPar[] = {20, 17, 6, 3, 8, 23, 18, 10, 22, 13, 4, 9, 31, 21, 88, 2};
    int N = 16;

    merge_sort_sequential(smallSeq, 0, N - 1);
    merge_sort_parallel(smallPar, 0, N - 1);
    print_array_range(smallSeq, 0, N - 1);
    print_array_range(smallPar, 0, N - 1);

    // Đo hiệu năng trên dữ liệu lớn hơn
    int* largeSeq = new int[400001];
    int* largePar = new int[400001];

    fill_rand_arr(largeSeq, largePar, 0, 400000);

    auto start = chrono::steady_clock::now();
    merge_sort_sequential(largeSeq, 0, 400000);
    auto stop = chrono::steady_clock::now();
    cout << "Merge sort (uS): "
         << chrono::duration_cast<chrono::microseconds>(stop - start).count()
         << " us\n";

    start = chrono::steady_clock::now();
    merge_sort_parallel(largePar, 0, 400000);
    stop = chrono::steady_clock::now();
    cout << "Merge sort OMP (uS): "
         << chrono::duration_cast<chrono::microseconds>(stop - start).count()
         << " us\n";

    delete[] largeSeq;
    delete[] largePar;
    return 0;
}
