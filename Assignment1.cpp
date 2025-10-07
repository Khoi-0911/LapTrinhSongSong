
#include <omp.h>
#include <iostream>

using namespace std;

// Tìm vị trí xuất hiện đầu tiên của 'target' trong mảng a (không sắp xếp).
// Dùng critical (không dùng reduction(min:))
int parallel_find_first(const int a[], int n, int target) {
    int idx = -1;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (a[i] == target) {
            #pragma omp critical
            {
                if (idx == -1 || i < idx) {
                    idx = i;
                }
            }
        }
    }
    return idx;
}

// Selection Sort: song song hóa bước tìm min trong đoạn [i+1 .. n-1]
// Mỗi vòng i tạo một vùng song song để tìm min (đơn giản, phù hợp người mới).
void selection_sort_omp(int a[], int n) {
    if (n <= 1) return;

    for (int i = 0; i < n - 1; ++i) {
        int minIdx = i;
        int minVal = a[i];

        #pragma omp parallel
        {
            int localMinIdx = minIdx;
            int localMinVal = minVal;

            #pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                int val = a[j];
                if (val < localMinVal) {
                    localMinVal = val;
                    localMinIdx = j;
                }
            }

            // Gộp kết quả các luồng bằng critical
            #pragma omp critical
            {
                if (localMinVal < minVal) {
                    minVal = localMinVal;
                    minIdx = localMinIdx;
                }
            }
        } // kết thúc vùng song song của vòng i

        if (minIdx != i) {
            // Không dùng std::swap và cũng không tự tạo hàm swap
            int tmp = a[i];
            a[i] = a[minIdx];
            a[minIdx] = tmp;
        }
    }
}

int main() {
    // Demo: mảng số nguyên int a[]
    int a[] = {5, 2, 7, 2, 9, 1, 4};
    int n = static_cast<int>(sizeof(a) / sizeof(a[0]));
    int target = 2;

    // Tìm kiếm phần tử
    int pos = parallel_find_first(a, n, target);
    cout << "Vi tri dau tien cua " << target << " trong mang: " << pos << "\n";

    // Sắp xếp lựa chọn (selection sort) song song ở bước tìm min
    selection_sort_omp(a, n);

    cout << "Mang sau khi sap xep: ";
    for (int i = 0; i < n; ++i) cout << a[i] << ' ';
    cout << "\n";

    return 0;
}
