#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> // Để sử dụng setprecision
#include <random>  // Để tạo đầu vào ngẫu nhiên
#include <limits.h> // Để sử dụng INT_MAX

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
    // Tạo đầu vào ngẫu nhiên đủ lớn: Ma trận VxV với V=1000 (O(V^3) ≈ 10^9 operations, đủ để thấy sự khác biệt thời gian)
    // Đồ thị sparse (xác suất có cạnh ~0.1), trọng số ngẫu nhiên từ 1 đến 100 (dương để tránh negative cycles)
    const int V = 1000;
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
    double start_seq = omp_get_wtime();
    floydWarshall(graph, dist_seq);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    cout << fixed << setprecision(4); // Hiển thị thời gian với 4 chữ số thập phân
    cout << "Thoi gian hoan thanh (tuan tu): " << time_seq << " giay" << endl;

    // Chạy Floyd-Warshall song song
    vector<vector<int>> dist_par;
    double start_par = omp_get_wtime();
    parallel_floydWarshall(graph, dist_par);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    cout << "Thoi gian hoan thanh (song song): " << time_par << " giay" << endl;

    // Kiểm tra số lượng threads cho song song (sử dụng một parallel region nhỏ riêng biệt, không nằm trong thời gian đo)
    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }
    cout << "So luong threads dang su dung cho song song: " << num_threads << endl;

    // Kiểm tra tính đúng đắn (so sánh một vài giá trị ngẫu nhiên giữa dist_seq và dist_par)
    bool correct = true;
    uniform_int_distribution<> idx_dis(0, V-1);
    for (int check = 0; check < 10; ++check) { // Kiểm tra 10 cặp ngẫu nhiên
        int i = idx_dis(gen);
        int j = idx_dis(gen);
        if (dist_seq[i][j] != dist_par[i][j]) {
            correct = false;
            break;
        }
    }
    cout << "Ket qua song song trung khop voi tuan tu (kiem tra ngau nhien): " << (correct ? "Yes" : "No") << endl;

    // Lưu ý: Không in toàn bộ ma trận vì quá lớn; nếu cần, có thể in một phần nhỏ

    return 0;
}