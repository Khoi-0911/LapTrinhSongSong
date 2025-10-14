
#include <omp.h>
#include <iostream>

using namespace std;

int find_first(int a[], int n, int target) {
    // Tim vi tri dau tien xuat hien target trong mang:
        // a[]: Mang dau vao
        // n: So phan tu trong mang
        // target: Gia tri cua phan tu can tim

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

void sort(int a[], int n) {
    // Sap xep mang theo gia tri tu be den lon:
        // a[]: Mang dau vao can sap xep
        // n: So luong phan tu trong mang
        
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
            #pragma omp critical
            {
                if (localMinVal < minVal) {
                    minVal = localMinVal;
                    minIdx = localMinIdx;
                }
            }
        }

        if (minIdx != i) {
            // Doi cho phan tu nho nhat vao vi tri dang xet
            int tmp = a[i];
            a[i] = a[minIdx];
            a[minIdx] = tmp;
        }
    }
}

int main() {
    int a[] = {5, 2, 7, 2, 9, 1, 4, 6};
    int n = sizeof(a) / sizeof(a[0]);
    int x = 2;

    int pos = find_first(a, n, x);
    cout << "Vi tri dau tien cua " << x << " trong mang: " << pos << "\n";

    sort(a, n);

    cout << "Mang sau khi sap xep: ";
    for (int i = 0; i < n; ++i) cout << a[i] << ' ';

    return 0;
}
