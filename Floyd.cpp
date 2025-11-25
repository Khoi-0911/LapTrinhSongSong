#include <iostream>
#include <vector>
#include <omp.h>
#include <random>  // Để tạo đầu vào ngẫu nhiên
#include <limits.h> // Để sử dụng INT_MAX
#include <chrono>  // Để tính thời gian bằng chrono
#include <stdexcept> // Để throw exception nếu input invalid

#define INF INT_MAX

using namespace std;

// Function to implement the Floyd-Warshall algorithm (tuần tự)
void floydWarshall(const vector<vector<int>>& graph, vector<vector<int>>& dist) {
    int V = graph.size();
    dist = graph; // Copy graph vào dist

    // Update the solution matrix by considering all vertices
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && 
                    (long long)dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

// Function to implement the Floyd-Warshall algorithm cach 1 (song song với OpenMP)
void parallel_floydWarshall_1(const vector<vector<int>>& graph, vector<vector<int>>& dist) {
    int V = graph.size();
    dist = graph; // Copy graph vào dist

    // Update the solution matrix by considering all vertices
    for (int k = 0; k < V; ++k) {
        // Snapshot hàng k và cột k để tránh đọc/ghi chồng chéo trong cùng bước k
        vector<int> row_k(V);
        vector<int> col_k(V);
        for (int j = 0; j < V; ++j) row_k[j] = dist[k][j];
        for (int i = 0; i < V; ++i) col_k[i] = dist[i][k];

        #pragma omp parallel for
        for (int i = 0; i < V; ++i) {
            // đọc col_k[i] là giá trị dist[i][k] sau bước k-1
            int dik = col_k[i];
            if (dik == INF) continue; // Không có đường i -> k thì bỏ qua cả hàng

            for (int j = 0; j < V; ++j) {
                int dkj = row_k[j];     // dist[k][j] sau bước k-1
                if (dkj == INF) continue;

                long long through_k = (long long)dik + dkj;
                if (through_k < dist[i][j]) {
                    dist[i][j] = (int)through_k;
                }
            }
        }
    }
}

// Function to implement the Floyd-Warshall algorithm cach 2 (song song với OpenMP, tối ưu bằng collapse(2))
void parallel_floydWarshall_2(const vector<vector<int>>& graph, vector<vector<int>>& dist) {
    int V = graph.size();
    dist = graph; // Copy graph vào dist

    // Update the solution matrix by considering all vertices
    for (int k = 0; k < V; ++k) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && 
                    (long long)dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

int main() {
    // Người dùng nhập kích thước ma trận V
    int V;
    cout << "Nhap kich thuoc ma tran V (vi du: 1000, phai > 0): ";
    cin >> V;
    if (V <= 0) {
        throw invalid_argument("V phai lon hon 0!");
    }

    // Tạo đầu vào ngẫu nhiên: Ma trận VxV, đồ thị sparse (xác suất có cạnh ~0.1), trọng số ngẫu nhiên từ 1 đến 100
    vector<vector<int>> graph(V, vector<int>(V, INF));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> edge_dis(0, 9); // Xác suất 1/10 có cạnh (sparse)
    uniform_int_distribution<> weight_dis(1, 100); // Trọng số từ 1 đến 100

    for (int i = 0; i < V; ++i) {
        graph[i][i] = 0; // Khoảng cách đến chính mình là 0
        for (int j = i + 1; j < V; ++j) { // Đồ thị undirected, chỉ set upper triangle
            if (edge_dis(gen) == 0) { // 10% chance có cạnh
                int weight = weight_dis(gen);
                graph[i][j] = weight;
                graph[j][i] = weight;
            }
        }
    }

    // Chạy Floyd-Warshall tuần tự
    vector<vector<int>> dist_seq;
    auto start_seq = chrono::high_resolution_clock::now();
    floydWarshall(graph, dist_seq);
    auto end_seq = chrono::high_resolution_clock::now();
    auto duration_seq = chrono::duration_cast<chrono::milliseconds>(end_seq - start_seq).count();
    cout << "[INFO] Thoi gian hoan thanh (tuan tu): " << duration_seq << " ms\n" << endl;

    // Cấu hình và chạy song song với số lượng threads khác nhau: 4, 8, 12
    vector<int> thread_counts = {4, 8, 12};

    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads); // Set số lượng threads

        // Kiểm tra số lượng threads thực tế (sử dụng một parallel region nhỏ riêng biệt)
        int actual_threads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            {
                actual_threads = omp_get_num_threads();
            }
        }

        // Chạy Cach 1
        vector<vector<int>> dist_par_1;
        auto start_par_1 = chrono::high_resolution_clock::now();
        parallel_floydWarshall_1(graph, dist_par_1);
        auto end_par_1 = chrono::high_resolution_clock::now();
        auto duration_par_1 = chrono::duration_cast<chrono::milliseconds>(end_par_1 - start_par_1).count();
        cout << "[INFO] Thoi gian hoan thanh (Cach 1 voi " << actual_threads << " threads): " << duration_par_1 << " ms" << endl;

        // Kiểm tra tính đúng đắn cho Cach 1
        bool correct_1 = true;
        uniform_int_distribution<> idx_dis(0, V-1);
        for (int check = 0; check < V/10; ++check) { // Kiểm tra V/10 cặp ngẫu nhiên
            int i = idx_dis(gen);
            int j = idx_dis(gen);
            if (dist_seq[i][j] != dist_par_1[i][j]) {
                correct_1 = false;
                break;
            }
        }
        cout << "[INFO] Ket qua trung khop voi tuan tu: " << (correct_1 ? "True" : "False") << "\n" << endl;

        // Chạy Cach 2
        vector<vector<int>> dist_par_2;
        auto start_par_2 = chrono::high_resolution_clock::now();
        parallel_floydWarshall_2(graph, dist_par_2);
        auto end_par_2 = chrono::high_resolution_clock::now();
        auto duration_par_2 = chrono::duration_cast<chrono::milliseconds>(end_par_2 - start_par_2).count();
        cout << "[INFO] Thoi gian hoan thanh (Cach 2 voi " << actual_threads << " threads): " << duration_par_2 << " ms" << endl;

        // Kiểm tra tính đúng đắn cho Cach 2
        bool correct_2 = true;
        for (int check = 0; check < 50; ++check) { // Kiểm tra 50 cặp ngẫu nhiên
            int i = idx_dis(gen);
            int j = idx_dis(gen);
            if (dist_seq[i][j] != dist_par_2[i][j]) {
                correct_2 = false;
                break;
            }
        }
        cout << "[INFO] Ket qua trung khop voi tuan tu: " << (correct_2 ? "True" : "False") << "\n" << endl;
    }

    return 0;
}