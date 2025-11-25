#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> // Để sử dụng setprecision (nếu cần, nhưng giờ dùng chrono)
#include <random>  // Để tạo đầu vào ngẫu nhiên
#include <limits.h> // Để sử dụng INT_MAX
#include <chrono>  // Để tính thời gian bằng chrono

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
                    (long long)dist[i][k] + dist[k][j] < dist[i][j]) { // Sử dụng long long để tránh overflow
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

// Function to implement the Floyd-Warshall algorithm (song song với OpenMP)
void parallel_floydWarshall(const vector<vector<int>>& graph, vector<vector<int>>& dist) {
    int V = graph.size();
    dist = graph; // Copy graph vào dist

    // Update the solution matrix by considering all vertices
    for (int k = 0; k < V; ++k) {
        #pragma omp parallel for
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && 
                    (long long)dist[i][k] + dist[k][j] < dist[i][j]) { // Sử dụng long long để tránh overflow
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

int main() {
    // Người dùng nhập kích thước ma trận V
    int V;
    cout << "Nhap kich thuoc ma tran V (vi du: 1000): ";
    cin >> V;

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
    cout << "[INFO] Dang chay Floyd-Warshall tuan tu..." << endl;
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

        // Kiểm tra số lượng threads thực tế (sử dụng một parallel region nhỏ riêng biệt, không nằm trong thời gian đo)
        int actual_threads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            {
                actual_threads = omp_get_num_threads();
            }
        }
        cout << "[INFO] Dang chay Floyd-Warshall song song voi " << num_threads << " threads..." << endl;
        cout << "[INFO] So luong threads thuc te: " << actual_threads << endl;

        // Chạy Floyd-Warshall song song
        vector<vector<int>> dist_par;
        auto start_par = chrono::high_resolution_clock::now();
        parallel_floydWarshall(graph, dist_par);
        auto end_par = chrono::high_resolution_clock::now();
        auto duration_par = chrono::duration_cast<chrono::milliseconds>(end_par - start_par).count();
        cout << "[INFO] Thoi gian hoan thanh (song song voi " << num_threads << " threads): " << duration_par << " ms" << endl;

        // Kiểm tra tính đúng đắn (so sánh một vài giá trị ngẫu nhiên giữa dist_seq và dist_par) - chỉ kiểm tra cho mỗi chạy
        bool correct = true;
        uniform_int_distribution<> idx_dis(0, V-1);
        for (int check = 0; check < V/10; ++check) { // Kiểm tra V/10 cặp ngẫu nhiên
            int i = idx_dis(gen);
            int j = idx_dis(gen);
            if (dist_seq[i][j] != dist_par[i][j]) {
                correct = false;
                break;
            }
        }
        cout << "[INFO] Ket qua song song trung khop voi tuan tu: " << (correct ? "True" : "False") << "\n" << endl;
    }

    // Lưu ý: Không in toàn bộ ma trận vì quá lớn; nếu cần, có thể in một phần nhỏ

    return 0;
}